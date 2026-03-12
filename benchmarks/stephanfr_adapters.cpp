#include "benchmarks/stephanfr_adapters.hpp"

#include <array>
#include <cmath>
#include <cstdint>

#include <immintrin.h>

extern "C" __m256d _ZGVdN4v_log(__m256d) noexcept;

#include "SIMDInstructionSet.h"
#include "Xoshiro256Plus.h"

namespace {

constexpr std::uint64_t kSeed = 0x123456789abcdef0ULL;

inline auto u64x4_to_uniform01_52(const std::uint64_t (&bits)[4]) noexcept
    -> std::array<double, 4> {
    alignas(32) std::array<double, 4> out{};
    const __m256i raw = _mm256_load_si256(reinterpret_cast<const __m256i*>(bits));
    const __m256i exponent = _mm256_set1_epi64x(0x3ff0000000000000ULL);
    const __m256i mantissa = _mm256_srli_epi64(raw, 12);
    const __m256d values = _mm256_sub_pd(
        _mm256_castsi256_pd(_mm256_or_si256(mantissa, exponent)),
        _mm256_set1_pd(1.0));
    _mm256_store_pd(out.data(), values);
    return out;
}

inline auto u64x4_to_pm1_52(const std::uint64_t (&bits)[4]) noexcept
    -> std::array<double, 4> {
    const auto u = u64x4_to_uniform01_52(bits);
    return {2.0 * u[0] - 1.0, 2.0 * u[1] - 1.0, 2.0 * u[2] - 1.0,
            2.0 * u[3] - 1.0};
}

inline auto u64x4_to_pm1_52_vec(const std::uint64_t (&bits)[4]) noexcept
    -> __m256d {
    const __m256i raw = _mm256_load_si256(reinterpret_cast<const __m256i*>(bits));
    const __m256i exponent = _mm256_set1_epi64x(0x3ff0000000000000ULL);
    const __m256i mantissa = _mm256_srli_epi64(raw, 12);
    const __m256d u = _mm256_sub_pd(
        _mm256_castsi256_pd(_mm256_or_si256(mantissa, exponent)),
        _mm256_set1_pd(1.0));
    return _mm256_sub_pd(_mm256_add_pd(u, u), _mm256_set1_pd(1.0));
}

using StephanFRXoshiro = SEFUtility::RNG::Xoshiro256Plus<SIMDInstructionSet::AVX2>;

}  // namespace

namespace zorro_bench {

void fill_uniform_stephanfr_avx2_52(double* out, std::size_t count) {
    StephanFRXoshiro rng(kSeed);
    std::size_t i = 0;
    while (i + 4 <= count) {
        const auto bits = rng.next4();
        alignas(32) const std::uint64_t raw[4] = {bits[0], bits[1], bits[2], bits[3]};
        const auto values = u64x4_to_uniform01_52(raw);
        for (int lane = 0; lane < 4; ++lane)
            out[i + lane] = values[lane];
        i += 4;
    }
    if (i < count) {
        const auto bits = rng.next4();
        alignas(32) const std::uint64_t raw[4] = {bits[0], bits[1], bits[2], bits[3]};
        const auto values = u64x4_to_uniform01_52(raw);
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = values[lane];
    }
}

void fill_normals_stephanfr_avx2_52(double* out, std::size_t count) {
    StephanFRXoshiro rng(kSeed);
    std::size_t i = 0;
    while (i < count) {
        const auto r1_bits = rng.next4();
        const auto r2_bits = rng.next4();
        alignas(32) const std::uint64_t raw1[4] = {r1_bits[0], r1_bits[1], r1_bits[2],
                                                   r1_bits[3]};
        alignas(32) const std::uint64_t raw2[4] = {r2_bits[0], r2_bits[1], r2_bits[2],
                                                   r2_bits[3]};
        const auto r1 = u64x4_to_pm1_52(raw1);
        const auto r2 = u64x4_to_pm1_52(raw2);
        for (std::size_t lane = 0; lane < 4 && i < count; ++lane) {
            const double s = r1[lane] * r1[lane] + r2[lane] * r2[lane];
            if (s >= 1.0 || s == 0.0)
                continue;
            const double scale = std::sqrt(-2.0 * std::log(s) / s);
            out[i++] = r1[lane] * scale;
            if (i < count)
                out[i++] = r2[lane] * scale;
        }
    }
}

void fill_normals_stephanfr_avx2_52_batched(double* out, std::size_t count) {
    StephanFRXoshiro rng(kSeed);
    static constexpr int kBuf = 64;
    double buf_u1[kBuf], buf_u2[kBuf], buf_s[kBuf];

    std::size_t i = 0;
    while (i < count) {
        int n = 0;
        while (n < kBuf) {
            const auto r1_bits = rng.next4();
            const auto r2_bits = rng.next4();
            alignas(32) const std::uint64_t raw1[4] = {r1_bits[0], r1_bits[1], r1_bits[2],
                                                       r1_bits[3]};
            alignas(32) const std::uint64_t raw2[4] = {r2_bits[0], r2_bits[1], r2_bits[2],
                                                       r2_bits[3]};
            const auto r1 = u64x4_to_pm1_52(raw1);
            const auto r2 = u64x4_to_pm1_52(raw2);
            for (int lane = 0; lane < 4 && n < kBuf; ++lane) {
                const double s = r1[lane] * r1[lane] + r2[lane] * r2[lane];
                if (s >= 1.0 || s == 0.0)
                    continue;
                buf_u1[n] = r1[lane];
                buf_u2[n] = r2[lane];
                buf_s[n] = s;
                ++n;
            }
        }

        for (int k = 0; k < n && i < count; ++k) {
            const double scale = std::sqrt(-2.0 * std::log(buf_s[k]) / buf_s[k]);
            out[i++] = buf_u1[k] * scale;
            if (i < count)
                out[i++] = buf_u2[k] * scale;
        }
    }
}

void fill_normals_stephanfr_avx2_52_veclog(double* out, std::size_t count) {
    StephanFRXoshiro rng(kSeed);
    static constexpr int kBuf = 64;
    alignas(32) double buf_u1[kBuf], buf_u2[kBuf], buf_s[kBuf];
    const __m256d neg2 = _mm256_set1_pd(-2.0);

    std::size_t i = 0;
    while (i < count) {
        int n = 0;
        while (n < kBuf) {
            const auto r1_bits = rng.next4();
            const auto r2_bits = rng.next4();
            alignas(32) const std::uint64_t raw1[4] = {r1_bits[0], r1_bits[1], r1_bits[2],
                                                       r1_bits[3]};
            alignas(32) const std::uint64_t raw2[4] = {r2_bits[0], r2_bits[1], r2_bits[2],
                                                       r2_bits[3]};
            const auto r1 = u64x4_to_pm1_52(raw1);
            const auto r2 = u64x4_to_pm1_52(raw2);
            for (int lane = 0; lane < 4 && n < kBuf; ++lane) {
                const double s = r1[lane] * r1[lane] + r2[lane] * r2[lane];
                if (s >= 1.0 || s == 0.0)
                    continue;
                buf_u1[n] = r1[lane];
                buf_u2[n] = r2[lane];
                buf_s[n] = s;
                ++n;
            }
        }

        int k = 0;
        for (; k + 4 <= n && i < count; k += 4) {
            const __m256d s_vec = _mm256_load_pd(buf_s + k);
            const __m256d log_s = _ZGVdN4v_log(s_vec);
            const __m256d factor = _mm256_sqrt_pd(
                _mm256_div_pd(_mm256_mul_pd(neg2, log_s), s_vec));
            const __m256d n1 = _mm256_mul_pd(_mm256_load_pd(buf_u1 + k), factor);
            const __m256d n2 = _mm256_mul_pd(_mm256_load_pd(buf_u2 + k), factor);

            if (i + 8 <= count) {
                const __m256d lo = _mm256_unpacklo_pd(n1, n2);
                const __m256d hi = _mm256_unpackhi_pd(n1, n2);
                _mm256_storeu_pd(out + i, _mm256_permute2f128_pd(lo, hi, 0x20));
                _mm256_storeu_pd(out + i + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
                i += 8;
            } else {
                alignas(32) double n1_arr[4], n2_arr[4];
                _mm256_store_pd(n1_arr, n1);
                _mm256_store_pd(n2_arr, n2);
                for (int j = 0; j < 4 && i < count; ++j) {
                    out[i++] = n1_arr[j];
                    if (i < count)
                        out[i++] = n2_arr[j];
                }
            }
        }

        for (; k < n && i < count; ++k) {
            const double scale = std::sqrt(-2.0 * std::log(buf_s[k]) / buf_s[k]);
            out[i++] = buf_u1[k] * scale;
            if (i < count)
                out[i++] = buf_u2[k] * scale;
        }
    }
}

void fill_normals_stephanfr_avx2_52_vecpolar(double* out, std::size_t count) {
    StephanFRXoshiro rng(kSeed);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d zero = _mm256_setzero_pd();
    const __m256d neg2 = _mm256_set1_pd(-2.0);

    std::size_t i = 0;
    while (i < count) {
        const auto r1_bits = rng.next4();
        const auto r2_bits = rng.next4();
        alignas(32) const std::uint64_t raw1[4] = {r1_bits[0], r1_bits[1], r1_bits[2],
                                                   r1_bits[3]};
        alignas(32) const std::uint64_t raw2[4] = {r2_bits[0], r2_bits[1], r2_bits[2],
                                                   r2_bits[3]};
        const __m256d u1 = u64x4_to_pm1_52_vec(raw1);
        const __m256d u2 = u64x4_to_pm1_52_vec(raw2);
        const __m256d s = _mm256_add_pd(_mm256_mul_pd(u1, u1), _mm256_mul_pd(u2, u2));
        const __m256d accept = _mm256_and_pd(
            _mm256_cmp_pd(s, one, _CMP_LT_OQ),
            _mm256_cmp_pd(s, zero, _CMP_GT_OQ));
        const int mask_bits = _mm256_movemask_pd(accept);
        if (!mask_bits)
            continue;

        const __m256d safe_s = _mm256_blendv_pd(_mm256_set1_pd(0.5), s, accept);
        const __m256d log_s = _ZGVdN4v_log(safe_s);
        const __m256d factor = _mm256_sqrt_pd(
            _mm256_div_pd(_mm256_mul_pd(neg2, log_s), safe_s));
        const __m256d n1 = _mm256_mul_pd(u1, factor);
        const __m256d n2 = _mm256_mul_pd(u2, factor);

        alignas(32) double n1v[4], n2v[4];
        _mm256_store_pd(n1v, n1);
        _mm256_store_pd(n2v, n2);
        for (int lane = 0; lane < 4 && i < count; ++lane) {
            if (mask_bits & (1 << lane)) {
                out[i++] = n1v[lane];
                if (i < count)
                    out[i++] = n2v[lane];
            }
        }
    }
}

}  // namespace zorro_bench
