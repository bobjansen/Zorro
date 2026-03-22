#include "benchmarks/zorro_benchmark_kernels.hpp"

#include <cmath>
#include <cpuid.h>
#include <cstring>
#include <vector>
#ifdef __AVX2__
#include <immintrin.h>
// glibc libmvec: AVX2 4-wide double log
extern "C" __m256d _ZGVdN4v_log(__m256d) noexcept;
extern "C" __m256d _ZGVdN4v_sin(__m256d) noexcept;
extern "C" __m256d _ZGVdN4v_cos(__m256d) noexcept;
#endif
#ifdef __AVX512F__
#include <immintrin.h>
// glibc libmvec: AVX-512 8-wide double log
extern "C" __m512d _ZGVeN8v_log(__m512d) noexcept;
#endif

// Benchmark-only kernels. The public library surface remains in zorro/zorro.hpp.

namespace zorro_bench {

namespace {

constexpr double kBitsTo01Scale = 0x1.0p-53;
#ifndef __AVX2__
constexpr double kBitsToPm1Scale = 0x1.0p-52;
#endif

inline auto splitmix64(std::uint64_t& state) noexcept -> std::uint64_t {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

inline auto rotl64(std::uint64_t x, int k) noexcept -> std::uint64_t {
    return (x << k) | (x >> (64 - k));
}

// One step of the xoshiro256 state recurrence (output-function agnostic).
inline void state_advance(std::uint64_t (&s)[4]) noexcept {
    const std::uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl64(s[3], 45);
}

// Advance state by 2^192 steps. The long-jump polynomial is identical for all
// xoshiro256 variants (+, ++, **) because it depends only on the linear recurrence.
inline void long_jump(std::uint64_t (&s)[4]) noexcept {
    static constexpr std::uint64_t kCoeffs[4] = {
        0x76e15d3efefdcbbfULL, 0xc5004e441c522fb3ULL,
        0x77710069854ee241ULL, 0x39109bb02acbe635ULL,
    };
    std::uint64_t t[4] = {};
    for (auto coeff : kCoeffs) {
        for (int b = 0; b < 64; ++b) {
            if (coeff & (std::uint64_t{1} << b)) {
                t[0] ^= s[0]; t[1] ^= s[1]; t[2] ^= s[2]; t[3] ^= s[3];
            }
            state_advance(s);
        }
    }
    s[0] = t[0]; s[1] = t[1]; s[2] = t[2]; s[3] = t[3];
}

// Seed `n` independent lanes into SoA arrays. Lane 0 is seeded from splitmix64;
// each subsequent lane is 2^192 steps ahead of the previous via long_jump.
inline void seed_lanes(std::uint64_t seed, int n,
                       std::uint64_t* s0, std::uint64_t* s1,
                       std::uint64_t* s2, std::uint64_t* s3) noexcept {
    std::uint64_t state[4] = {
        splitmix64(seed), splitmix64(seed),
        splitmix64(seed), splitmix64(seed),
    };
    for (int lane = 0; lane < n; ++lane) {
        s0[lane] = state[0]; s1[lane] = state[1];
        s2[lane] = state[2]; s3[lane] = state[3];
        if (lane + 1 < n) long_jump(state);
    }
}

inline void next_x4(std::uint64_t (&s0)[4], std::uint64_t (&s1)[4],
                    std::uint64_t (&s2)[4], std::uint64_t (&s3)[4],
                    std::uint64_t* result) noexcept {
    std::uint64_t t[4];
    for (int lane = 0; lane < 4; ++lane)
        result[lane] = rotl64(s0[lane] + s3[lane], 23) + s0[lane];
    for (int lane = 0; lane < 4; ++lane)
        t[lane] = s1[lane] << 17;
    for (int lane = 0; lane < 4; ++lane)
        s2[lane] ^= s0[lane];
    for (int lane = 0; lane < 4; ++lane)
        s3[lane] ^= s1[lane];
    for (int lane = 0; lane < 4; ++lane)
        s1[lane] ^= s2[lane];
    for (int lane = 0; lane < 4; ++lane)
        s0[lane] ^= s3[lane];
    for (int lane = 0; lane < 4; ++lane)
        s2[lane] ^= t[lane];
    for (int lane = 0; lane < 4; ++lane)
        s3[lane] = rotl64(s3[lane], 45);
}

[[maybe_unused]] inline void next_x4_plus(std::uint64_t (&s0)[4],
                                          std::uint64_t (&s1)[4],
                                          std::uint64_t (&s2)[4],
                                          std::uint64_t (&s3)[4],
                                          std::uint64_t* result) noexcept {
    std::uint64_t t[4];
    for (int lane = 0; lane < 4; ++lane)
        result[lane] = s0[lane] + s3[lane];
    for (int lane = 0; lane < 4; ++lane)
        t[lane] = s1[lane] << 17;
    for (int lane = 0; lane < 4; ++lane)
        s2[lane] ^= s0[lane];
    for (int lane = 0; lane < 4; ++lane)
        s3[lane] ^= s1[lane];
    for (int lane = 0; lane < 4; ++lane)
        s1[lane] ^= s2[lane];
    for (int lane = 0; lane < 4; ++lane)
        s0[lane] ^= s3[lane];
    for (int lane = 0; lane < 4; ++lane)
        s2[lane] ^= t[lane];
    for (int lane = 0; lane < 4; ++lane)
        s3[lane] = rotl64(s3[lane], 45);
}

#ifdef __AVX2__
template <int k>
inline auto rotl64_avx2(__m256i x) noexcept -> __m256i {
#ifdef __AVX512VL__
    return _mm256_rol_epi64(x, k);
#else
    return _mm256_or_si256(_mm256_slli_epi64(x, k), _mm256_srli_epi64(x, 64 - k));
#endif
}

inline auto next_x4_avx2(__m256i& s0, __m256i& s1, __m256i& s2,
                         __m256i& s3) noexcept -> __m256i {
    const __m256i result =
        _mm256_add_epi64(rotl64_avx2<23>(_mm256_add_epi64(s0, s3)), s0);
    const __m256i t = _mm256_slli_epi64(s1, 17);
    s2 = _mm256_xor_si256(s2, s0);
    s3 = _mm256_xor_si256(s3, s1);
    s1 = _mm256_xor_si256(s1, s2);
    s0 = _mm256_xor_si256(s0, s3);
    s2 = _mm256_xor_si256(s2, t);
    s3 = rotl64_avx2<45>(s3);
    return result;
}

inline auto next_x4_plus_avx2(__m256i& s0, __m256i& s1, __m256i& s2,
                              __m256i& s3) noexcept -> __m256i {
    const __m256i result = _mm256_add_epi64(s0, s3);
    const __m256i t = _mm256_slli_epi64(s1, 17);
    s2 = _mm256_xor_si256(s2, s0);
    s3 = _mm256_xor_si256(s3, s1);
    s1 = _mm256_xor_si256(s1, s2);
    s0 = _mm256_xor_si256(s0, s3);
    s2 = _mm256_xor_si256(s2, t);
    s3 = rotl64_avx2<45>(s3);
    return result;
}

inline auto u64_to_uniform01_52_avx2(__m256i bits) noexcept -> __m256d {
    const __m256i exponent = _mm256_set1_epi64x(0x3ff0000000000000ULL);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256i mantissa = _mm256_srli_epi64(bits, 12);
    const __m256d one_to_two =
        _mm256_castsi256_pd(_mm256_or_si256(mantissa, exponent));
    return _mm256_sub_pd(one_to_two, one);
}

inline auto u64_to_pm1_52_avx2(__m256i bits) noexcept -> __m256d {
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d two = _mm256_set1_pd(2.0);
    return _mm256_sub_pd(_mm256_mul_pd(u64_to_uniform01_52_avx2(bits), two), one);
}

inline void seed_x4(std::uint64_t seed, std::uint64_t (&s0)[4],
                    std::uint64_t (&s1)[4], std::uint64_t (&s2)[4],
                    std::uint64_t (&s3)[4]) noexcept {
    seed_lanes(seed, 4, s0, s1, s2, s3);
}

inline void seed_x8(std::uint64_t seed,
                    std::uint64_t (&sa0)[4], std::uint64_t (&sa1)[4],
                    std::uint64_t (&sa2)[4], std::uint64_t (&sa3)[4],
                    std::uint64_t (&sb0)[4], std::uint64_t (&sb1)[4],
                    std::uint64_t (&sb2)[4], std::uint64_t (&sb3)[4]) noexcept {
    std::uint64_t all0[8], all1[8], all2[8], all3[8];
    seed_lanes(seed, 8, all0, all1, all2, all3);
    for (int lane = 0; lane < 4; ++lane) {
        sa0[lane] = all0[lane]; sa1[lane] = all1[lane];
        sa2[lane] = all2[lane]; sa3[lane] = all3[lane];
        sb0[lane] = all0[lane + 4]; sb1[lane] = all1[lane + 4];
        sb2[lane] = all2[lane + 4]; sb3[lane] = all3[lane + 4];
    }
}
#endif

// Scalar xoshiro256++ step. Used by scalar and slow-path gamma/student-t code.
inline auto next_pp(std::uint64_t (&s)[4]) noexcept -> std::uint64_t {
    const std::uint64_t r = rotl64(s[0] + s[3], 23) + s[0];
    state_advance(s);
    return r;
}

#ifdef __AVX512F__

// CPUID leaf 0: vendor string is in EBX-EDX-ECX (yes, that order).
// "AuthenticAMD" → EBX=0x68747541 EDX=0x69746e65 ECX=0x444d4163
inline auto cpu_is_amd() noexcept -> bool {
    static const bool is_amd = [] {
        unsigned eax, ebx, ecx, edx;
        __cpuid(0, eax, ebx, ecx, edx);
        return ebx == 0x68747541u && edx == 0x69746e65u && ecx == 0x444d4163u;
    }();
    return is_amd;
}

// ── AVX-512 helpers ──────────────────────────────────────────────────────────

// Rotate-left for 64-bit lanes in a 512-bit register.
// _mm512_rol_epi64 is available in base AVX-512F — no VL qualifier needed.
template <int k>
inline auto rotl64_avx512(__m512i x) noexcept -> __m512i {
    return _mm512_rol_epi64(x, k);
}

// xoshiro256++ step for 8 independent streams packed into four __m512i registers.
// Identical arithmetic to next_x4_avx2; every __m256i intrinsic becomes __m512i.
inline auto next_x8_avx512(__m512i& s0, __m512i& s1, __m512i& s2,
                            __m512i& s3) noexcept -> __m512i {
    const __m512i result =
        _mm512_add_epi64(rotl64_avx512<23>(_mm512_add_epi64(s0, s3)), s0);
    const __m512i t = _mm512_slli_epi64(s1, 17);
    s2 = _mm512_xor_si512(s2, s0);
    s3 = _mm512_xor_si512(s3, s1);
    s1 = _mm512_xor_si512(s1, s2);
    s0 = _mm512_xor_si512(s0, s3);
    s2 = _mm512_xor_si512(s2, t);
    s3 = rotl64_avx512<45>(s3);
    return result;
}

// Map 8 raw 64-bit values to doubles in [0, 1) using the same mantissa-force
// trick as u64_to_uniform01_52_avx2: discard top 12 bits, OR in exponent 0x3FF,
// reinterpret as doubles in [1.0, 2.0), subtract 1.0.
inline auto u64_to_uniform01_52_avx512(__m512i bits) noexcept -> __m512d {
    const __m512i exponent =
        _mm512_set1_epi64(static_cast<std::int64_t>(0x3ff0000000000000ULL));
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512i mantissa = _mm512_srli_epi64(bits, 12);
    const __m512d one_to_two =
        _mm512_castsi512_pd(_mm512_or_si512(mantissa, exponent));
    return _mm512_sub_pd(one_to_two, one);
}

// Map 8 raw 64-bit values to doubles in (-1, 1): [0,1)*2 - 1.
inline auto u64_to_pm1_52_avx512(__m512i bits) noexcept -> __m512d {
    const __m512d two = _mm512_set1_pd(2.0);
    const __m512d one = _mm512_set1_pd(1.0);
    return _mm512_sub_pd(_mm512_mul_pd(u64_to_uniform01_52_avx512(bits), two), one);
}

// Seed one group of 8 independent lanes into 64-byte-aligned SoA arrays.
inline void seed_x8_avx512(std::uint64_t seed,
                            std::uint64_t (&s0)[8], std::uint64_t (&s1)[8],
                            std::uint64_t (&s2)[8], std::uint64_t (&s3)[8]) noexcept {
    seed_lanes(seed, 8, s0, s1, s2, s3);
}

// Seed two groups of 8 lanes (16 total).  Each group gets 8 streams spaced
// 2^192 apart; the two groups are also separated by 2^192 from each other.
inline void seed_x16_avx512(std::uint64_t seed,
                             std::uint64_t (&sa0)[8], std::uint64_t (&sa1)[8],
                             std::uint64_t (&sa2)[8], std::uint64_t (&sa3)[8],
                             std::uint64_t (&sb0)[8], std::uint64_t (&sb1)[8],
                             std::uint64_t (&sb2)[8], std::uint64_t (&sb3)[8]) noexcept {
    std::uint64_t all0[16], all1[16], all2[16], all3[16];
    seed_lanes(seed, 16, all0, all1, all2, all3);
    for (int lane = 0; lane < 8; ++lane) {
        sa0[lane] = all0[lane];     sa1[lane] = all1[lane];
        sa2[lane] = all2[lane];     sa3[lane] = all3[lane];
        sb0[lane] = all0[lane + 8]; sb1[lane] = all1[lane + 8];
        sb2[lane] = all2[lane + 8]; sb3[lane] = all3[lane + 8];
    }
}

#endif  // __AVX512F__

}  // namespace

void generate_xoshiro256pp_x4_bits(std::uint64_t seed, std::uint64_t* out,
                                   std::size_t count) noexcept {
    std::uint64_t s0[4];
    std::uint64_t s1[4];
    std::uint64_t s2[4];
    std::uint64_t s3[4];

    seed_lanes(seed, 4, s0, s1, s2, s3);

    std::size_t i = 0;
    while (i + 64 <= count) {
        for (std::size_t batch = 0; batch < 64; batch += 4)
            next_x4(s0, s1, s2, s3, out + i + batch);
        i += 64;
    }
    while (i + 4 <= count) {
        next_x4(s0, s1, s2, s3, out + i);
        i += 4;
    }
    if (i < count) {
        std::uint64_t tail[4];
        next_x4(s0, s1, s2, s3, tail);
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = tail[lane];
    }
}

void fill_xoshiro256pp_x4_uniform01(std::uint64_t seed, double* out,
                                    std::size_t count) noexcept {
    std::uint64_t s0[4];
    std::uint64_t s1[4];
    std::uint64_t s2[4];
    std::uint64_t s3[4];

    seed_lanes(seed, 4, s0, s1, s2, s3);

    std::size_t i = 0;
    while (i + 64 <= count) {
        std::uint64_t bits[64];
        for (std::size_t batch = 0; batch < 64; batch += 4)
            next_x4(s0, s1, s2, s3, bits + batch);
        for (std::size_t lane = 0; lane < 64; ++lane)
            out[i + lane] = static_cast<double>(bits[lane] >> 11) * kBitsTo01Scale;
        i += 64;
    }
    while (i + 4 <= count) {
        std::uint64_t bits[4];
        next_x4(s0, s1, s2, s3, bits);
        for (int lane = 0; lane < 4; ++lane)
            out[i + lane] = static_cast<double>(bits[lane] >> 11) * kBitsTo01Scale;
        i += 4;
    }
    if (i < count) {
        std::uint64_t tail[4];
        next_x4(s0, s1, s2, s3, tail);
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = static_cast<double>(tail[lane] >> 11) * kBitsTo01Scale;
    }
}

void fill_xoshiro256pp_x4_uniform01_avx2(std::uint64_t seed, double* out,
                                         std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t state0[4];
    alignas(32) std::uint64_t state1[4];
    alignas(32) std::uint64_t state2[4];
    alignas(32) std::uint64_t state3[4];
    alignas(32) double tail[4];
    seed_x4(seed, state0, state1, state2, state3);

    __m256i s0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state0));
    __m256i s1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state1));
    __m256i s2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state2));
    __m256i s3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state3));

    std::size_t i = 0;
    while (i + 4 <= count) {
        const __m256d values = u64_to_uniform01_52_avx2(next_x4_avx2(s0, s1, s2, s3));
        _mm256_storeu_pd(out + i, values);
        i += 4;
    }
    if (i < count) {
        const __m256d values = u64_to_uniform01_52_avx2(next_x4_avx2(s0, s1, s2, s3));
        _mm256_store_pd(tail, values);
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = tail[lane];
    }
#else
    fill_xoshiro256pp_x4_uniform01(seed, out, count);
#endif
}

void fill_xoshiro256pp_x4_uniform01_avx2_unroll2(std::uint64_t seed, double* out,
                                                  std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t state0[4];
    alignas(32) std::uint64_t state1[4];
    alignas(32) std::uint64_t state2[4];
    alignas(32) std::uint64_t state3[4];
    alignas(32) double tail[4];
    seed_x4(seed, state0, state1, state2, state3);

    __m256i s0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state0));
    __m256i s1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state1));
    __m256i s2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state2));
    __m256i s3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state3));

    std::size_t i = 0;
    while (i + 8 <= count) {
        const __m256i r0 = next_x4_avx2(s0, s1, s2, s3);
        const __m256i r1 = next_x4_avx2(s0, s1, s2, s3);
        _mm256_storeu_pd(out + i,     u64_to_uniform01_52_avx2(r0));
        _mm256_storeu_pd(out + i + 4, u64_to_uniform01_52_avx2(r1));
        i += 8;
    }
    while (i + 4 <= count) {
        const __m256d values = u64_to_uniform01_52_avx2(next_x4_avx2(s0, s1, s2, s3));
        _mm256_storeu_pd(out + i, values);
        i += 4;
    }
    if (i < count) {
        const __m256d values = u64_to_uniform01_52_avx2(next_x4_avx2(s0, s1, s2, s3));
        _mm256_store_pd(tail, values);
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = tail[lane];
    }
#else
    fill_xoshiro256pp_x4_uniform01(seed, out, count);
#endif
}

void fill_xoshiro256pp_x8_uniform01_avx2(std::uint64_t seed, double* out,
                                         std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
    alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
    alignas(32) double tail[4];
    seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
    __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
    __m256i a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
    __m256i a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
    __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
    __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
    __m256i b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
    __m256i b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));

    std::size_t i = 0;
    while (i + 8 <= count) {
        const __m256i ra = next_x4_avx2(a0, a1, a2, a3);
        const __m256i rb = next_x4_avx2(b0, b1, b2, b3);
        _mm256_storeu_pd(out + i,     u64_to_uniform01_52_avx2(ra));
        _mm256_storeu_pd(out + i + 4, u64_to_uniform01_52_avx2(rb));
        i += 8;
    }
    while (i + 4 <= count) {
        const __m256d values = u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3));
        _mm256_storeu_pd(out + i, values);
        i += 4;
    }
    if (i < count) {
        const __m256d values = u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3));
        _mm256_store_pd(tail, values);
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = tail[lane];
    }
#else
    fill_xoshiro256pp_x4_uniform01(seed, out, count);
#endif
}

void fill_xoshiro256p_x4_uniform01_avx2(std::uint64_t seed, double* out,
                                        std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t state0[4];
    alignas(32) std::uint64_t state1[4];
    alignas(32) std::uint64_t state2[4];
    alignas(32) std::uint64_t state3[4];
    alignas(32) double tail[4];
    seed_x4(seed, state0, state1, state2, state3);

    __m256i s0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state0));
    __m256i s1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state1));
    __m256i s2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state2));
    __m256i s3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state3));

    std::size_t i = 0;
    while (i + 4 <= count) {
        const __m256d values =
            u64_to_uniform01_52_avx2(next_x4_plus_avx2(s0, s1, s2, s3));
        _mm256_storeu_pd(out + i, values);
        i += 4;
    }
    if (i < count) {
        const __m256d values =
            u64_to_uniform01_52_avx2(next_x4_plus_avx2(s0, s1, s2, s3));
        _mm256_store_pd(tail, values);
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = tail[lane];
    }
#else
    fill_xoshiro256pp_x4_uniform01(seed, out, count);
#endif
}

void fill_xoshiro256pp_x4_normal_polar_avx2(std::uint64_t seed, double* out,
                                            std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t state0[4];
    alignas(32) std::uint64_t state1[4];
    alignas(32) std::uint64_t state2[4];
    alignas(32) std::uint64_t state3[4];
    alignas(32) double r1[4];
    alignas(32) double r2[4];
    seed_x4(seed, state0, state1, state2, state3);

    __m256i s0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state0));
    __m256i s1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state1));
    __m256i s2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state2));
    __m256i s3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state3));

    std::size_t i = 0;
    while (i < count) {
        _mm256_store_pd(r1, u64_to_pm1_52_avx2(next_x4_avx2(s0, s1, s2, s3)));
        _mm256_store_pd(r2, u64_to_pm1_52_avx2(next_x4_avx2(s0, s1, s2, s3)));
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
#else
    std::uint64_t s0[4];
    std::uint64_t s1[4];
    std::uint64_t s2[4];
    std::uint64_t s3[4];
    seed_lanes(seed, 4, s0, s1, s2, s3);
    std::size_t i = 0;
    while (i < count) {
        std::uint64_t r1[4];
        std::uint64_t r2[4];
        next_x4(s0, s1, s2, s3, r1);
        next_x4(s0, s1, s2, s3, r2);
        for (std::size_t lane = 0; lane < 4 && i < count; ++lane) {
            const double u1 =
                static_cast<double>(static_cast<std::int64_t>(r1[lane]) >> 11) *
                kBitsToPm1Scale;
            const double u2 =
                static_cast<double>(static_cast<std::int64_t>(r2[lane]) >> 11) *
                kBitsToPm1Scale;
            const double s = u1 * u1 + u2 * u2;
            if (s >= 1.0 || s == 0.0)
                continue;
            const double scale = std::sqrt(-2.0 * std::log(s) / s);
            out[i++] = u1 * scale;
            if (i < count)
                out[i++] = u2 * scale;
        }
    }
#endif
}

void fill_xoshiro256p_x4_normal_polar_avx2(std::uint64_t seed, double* out,
                                           std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t state0[4];
    alignas(32) std::uint64_t state1[4];
    alignas(32) std::uint64_t state2[4];
    alignas(32) std::uint64_t state3[4];
    alignas(32) double r1[4];
    alignas(32) double r2[4];
    seed_x4(seed, state0, state1, state2, state3);

    __m256i s0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state0));
    __m256i s1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state1));
    __m256i s2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state2));
    __m256i s3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state3));

    std::size_t i = 0;
    while (i < count) {
        _mm256_store_pd(r1,
                        u64_to_pm1_52_avx2(next_x4_plus_avx2(s0, s1, s2, s3)));
        _mm256_store_pd(r2,
                        u64_to_pm1_52_avx2(next_x4_plus_avx2(s0, s1, s2, s3)));
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
#else
    std::uint64_t s0[4];
    std::uint64_t s1[4];
    std::uint64_t s2[4];
    std::uint64_t s3[4];
    seed_lanes(seed, 4, s0, s1, s2, s3);
    std::size_t i = 0;
    while (i < count) {
        std::uint64_t r1_bits[4];
        std::uint64_t r2_bits[4];
        next_x4_plus(s0, s1, s2, s3, r1_bits);
        next_x4_plus(s0, s1, s2, s3, r2_bits);
        for (std::size_t lane = 0; lane < 4 && i < count; ++lane) {
            const double u1 =
                static_cast<double>(static_cast<std::int64_t>(r1_bits[lane]) >> 11) *
                kBitsToPm1Scale;
            const double u2 =
                static_cast<double>(static_cast<std::int64_t>(r2_bits[lane]) >> 11) *
                kBitsToPm1Scale;
            const double s = u1 * u1 + u2 * u2;
            if (s >= 1.0 || s == 0.0)
                continue;
            const double scale = std::sqrt(-2.0 * std::log(s) / s);
            out[i++] = u1 * scale;
            if (i < count)
                out[i++] = u2 * scale;
        }
    }
#endif
}

// ─── Idea 4: Fix store-forwarding in polar (x8 ++ with extract) ─────────────
void fill_xoshiro256pp_x8_normal_polar_avx2(std::uint64_t seed, double* out,
                                             std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
    alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
    seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
    __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
    __m256i a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
    __m256i a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
    __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
    __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
    __m256i b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
    __m256i b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));

    // Process 8 lanes (a=4, b=4) per outer iteration.
    // Extract doubles via 128-bit halves to avoid store-forwarding stalls
    // (256-bit store → 64-bit scalar reload causes ~12cy penalty).
    std::size_t i = 0;
    while (i < count) {
        const __m256d v_r1a = u64_to_pm1_52_avx2(next_x4_avx2(a0, a1, a2, a3));
        const __m256d v_r2a = u64_to_pm1_52_avx2(next_x4_avx2(a0, a1, a2, a3));
        const __m256d v_r1b = u64_to_pm1_52_avx2(next_x4_avx2(b0, b1, b2, b3));
        const __m256d v_r2b = u64_to_pm1_52_avx2(next_x4_avx2(b0, b1, b2, b3));

        // Extract via 128-bit halves: no store-forwarding stall
        const __m128d r1a_lo = _mm256_castpd256_pd128(v_r1a);
        const __m128d r1a_hi = _mm256_extractf128_pd(v_r1a, 1);
        const __m128d r2a_lo = _mm256_castpd256_pd128(v_r2a);
        const __m128d r2a_hi = _mm256_extractf128_pd(v_r2a, 1);

        const double r1a[4] = {
            _mm_cvtsd_f64(r1a_lo), _mm_cvtsd_f64(_mm_unpackhi_pd(r1a_lo, r1a_lo)),
            _mm_cvtsd_f64(r1a_hi), _mm_cvtsd_f64(_mm_unpackhi_pd(r1a_hi, r1a_hi))};
        const double r2a[4] = {
            _mm_cvtsd_f64(r2a_lo), _mm_cvtsd_f64(_mm_unpackhi_pd(r2a_lo, r2a_lo)),
            _mm_cvtsd_f64(r2a_hi), _mm_cvtsd_f64(_mm_unpackhi_pd(r2a_hi, r2a_hi))};

        for (int lane = 0; lane < 4 && i < count; ++lane) {
            const double s = r1a[lane] * r1a[lane] + r2a[lane] * r2a[lane];
            if (s >= 1.0 || s == 0.0)
                continue;
            const double scale = std::sqrt(-2.0 * std::log(s) / s);
            out[i++] = r1a[lane] * scale;
            if (i < count)
                out[i++] = r2a[lane] * scale;
        }

        const __m128d r1b_lo = _mm256_castpd256_pd128(v_r1b);
        const __m128d r1b_hi = _mm256_extractf128_pd(v_r1b, 1);
        const __m128d r2b_lo = _mm256_castpd256_pd128(v_r2b);
        const __m128d r2b_hi = _mm256_extractf128_pd(v_r2b, 1);

        const double r1b[4] = {
            _mm_cvtsd_f64(r1b_lo), _mm_cvtsd_f64(_mm_unpackhi_pd(r1b_lo, r1b_lo)),
            _mm_cvtsd_f64(r1b_hi), _mm_cvtsd_f64(_mm_unpackhi_pd(r1b_hi, r1b_hi))};
        const double r2b[4] = {
            _mm_cvtsd_f64(r2b_lo), _mm_cvtsd_f64(_mm_unpackhi_pd(r2b_lo, r2b_lo)),
            _mm_cvtsd_f64(r2b_hi), _mm_cvtsd_f64(_mm_unpackhi_pd(r2b_hi, r2b_hi))};

        for (int lane = 0; lane < 4 && i < count; ++lane) {
            const double s = r1b[lane] * r1b[lane] + r2b[lane] * r2b[lane];
            if (s >= 1.0 || s == 0.0)
                continue;
            const double scale = std::sqrt(-2.0 * std::log(s) / s);
            out[i++] = r1b[lane] * scale;
            if (i < count)
                out[i++] = r2b[lane] * scale;
        }
    }
#else
    fill_xoshiro256pp_x4_normal_polar_avx2(seed, out, count);
#endif
}

// ─── Idea 3: Batched acceptance buffer ───────────────────────────────────────
// Separates the YMM-heavy RNG+rejection phase from the scalar log/sqrt phase.
// The 8 state registers (a0-a3, b0-b3) don't need to stay live during the
// transform pass, relieving spill pressure around the transcendental calls.
void fill_xoshiro256pp_x8_normal_polar_avx2_batched(std::uint64_t seed, double* out,
                                                     std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
    alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
    seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
    __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
    __m256i a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
    __m256i a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
    __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
    __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
    __m256i b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
    __m256i b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));

    // 64 accepted pairs = 3×512 bytes — comfortably within L1.
    // Store s alongside u1/u2 to avoid recomputing in the transform pass.
    static constexpr int kBuf = 64;
    double buf_u1[kBuf], buf_u2[kBuf], buf_s[kBuf];

    std::size_t i = 0;
    while (i < count) {
        // ── Phase 1: fill buffer (RNG + rejection only, no transcendentals) ──
        int n = 0;
        while (n < kBuf) {
            const __m256d v_r1a = u64_to_pm1_52_avx2(next_x4_avx2(a0, a1, a2, a3));
            const __m256d v_r2a = u64_to_pm1_52_avx2(next_x4_avx2(a0, a1, a2, a3));
            const __m256d v_r1b = u64_to_pm1_52_avx2(next_x4_avx2(b0, b1, b2, b3));
            const __m256d v_r2b = u64_to_pm1_52_avx2(next_x4_avx2(b0, b1, b2, b3));

            const __m128d r1a_lo = _mm256_castpd256_pd128(v_r1a);
            const __m128d r1a_hi = _mm256_extractf128_pd(v_r1a, 1);
            const __m128d r2a_lo = _mm256_castpd256_pd128(v_r2a);
            const __m128d r2a_hi = _mm256_extractf128_pd(v_r2a, 1);
            const double r1a[4] = {
                _mm_cvtsd_f64(r1a_lo), _mm_cvtsd_f64(_mm_unpackhi_pd(r1a_lo, r1a_lo)),
                _mm_cvtsd_f64(r1a_hi), _mm_cvtsd_f64(_mm_unpackhi_pd(r1a_hi, r1a_hi))};
            const double r2a[4] = {
                _mm_cvtsd_f64(r2a_lo), _mm_cvtsd_f64(_mm_unpackhi_pd(r2a_lo, r2a_lo)),
                _mm_cvtsd_f64(r2a_hi), _mm_cvtsd_f64(_mm_unpackhi_pd(r2a_hi, r2a_hi))};
            for (int lane = 0; lane < 4 && n < kBuf; ++lane) {
                const double s = r1a[lane] * r1a[lane] + r2a[lane] * r2a[lane];
                if (s >= 1.0 || s == 0.0) continue;
                buf_u1[n] = r1a[lane]; buf_u2[n] = r2a[lane]; buf_s[n] = s; ++n;
            }

            const __m128d r1b_lo = _mm256_castpd256_pd128(v_r1b);
            const __m128d r1b_hi = _mm256_extractf128_pd(v_r1b, 1);
            const __m128d r2b_lo = _mm256_castpd256_pd128(v_r2b);
            const __m128d r2b_hi = _mm256_extractf128_pd(v_r2b, 1);
            const double r1b[4] = {
                _mm_cvtsd_f64(r1b_lo), _mm_cvtsd_f64(_mm_unpackhi_pd(r1b_lo, r1b_lo)),
                _mm_cvtsd_f64(r1b_hi), _mm_cvtsd_f64(_mm_unpackhi_pd(r1b_hi, r1b_hi))};
            const double r2b[4] = {
                _mm_cvtsd_f64(r2b_lo), _mm_cvtsd_f64(_mm_unpackhi_pd(r2b_lo, r2b_lo)),
                _mm_cvtsd_f64(r2b_hi), _mm_cvtsd_f64(_mm_unpackhi_pd(r2b_hi, r2b_hi))};
            for (int lane = 0; lane < 4 && n < kBuf; ++lane) {
                const double s = r1b[lane] * r1b[lane] + r2b[lane] * r2b[lane];
                if (s >= 1.0 || s == 0.0) continue;
                buf_u1[n] = r1b[lane]; buf_u2[n] = r2b[lane]; buf_s[n] = s; ++n;
            }
        }

        // ── Phase 2: transform (no YMM state live, pure scalar) ──────────────
        for (int k = 0; k < n && i < count; ++k) {
            const double scale = std::sqrt(-2.0 * std::log(buf_s[k]) / buf_s[k]);
            out[i++] = buf_u1[k] * scale;
            if (i < count)
                out[i++] = buf_u2[k] * scale;
        }
    }
#else
    fill_xoshiro256pp_x4_normal_polar_avx2(seed, out, count);
#endif
}

// ─── Idea 6: Batched polar with vectorized log/sqrt via libmvec ──────────────
// Phase 1 is identical to the batched variant: collect accepted (u1, u2, s) triples.
// Phase 2 processes 4 triples at a time using _ZGVdN4v_log + _mm256_sqrt_pd.
void fill_xoshiro256pp_x8_normal_polar_avx2_veclog(std::uint64_t seed, double* out,
                                                     std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
    alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
    seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
    __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
    __m256i a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
    __m256i a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
    __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
    __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
    __m256i b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
    __m256i b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));

    static constexpr int kBuf = 64;
    alignas(32) double buf_u1[kBuf], buf_u2[kBuf], buf_s[kBuf];

    std::size_t i = 0;
    while (i < count) {
        // ── Phase 1: fill buffer (RNG + rejection only) ──
        int n = 0;
        while (n < kBuf) {
            const __m256d v_r1a = u64_to_pm1_52_avx2(next_x4_avx2(a0, a1, a2, a3));
            const __m256d v_r2a = u64_to_pm1_52_avx2(next_x4_avx2(a0, a1, a2, a3));
            const __m256d v_r1b = u64_to_pm1_52_avx2(next_x4_avx2(b0, b1, b2, b3));
            const __m256d v_r2b = u64_to_pm1_52_avx2(next_x4_avx2(b0, b1, b2, b3));

            // Compute s = u1² + u2² in SIMD for acceptance test
            const __m256d sa = _mm256_add_pd(_mm256_mul_pd(v_r1a, v_r1a),
                                              _mm256_mul_pd(v_r2a, v_r2a));
            const __m256d sb = _mm256_add_pd(_mm256_mul_pd(v_r1b, v_r1b),
                                              _mm256_mul_pd(v_r2b, v_r2b));

            // Accept mask: 0 < s < 1
            const __m256d one = _mm256_set1_pd(1.0);
            const __m256d zero = _mm256_setzero_pd();
            const __m256d mask_a = _mm256_and_pd(
                _mm256_cmp_pd(sa, one, _CMP_LT_OQ),
                _mm256_cmp_pd(sa, zero, _CMP_GT_OQ));
            const __m256d mask_b = _mm256_and_pd(
                _mm256_cmp_pd(sb, one, _CMP_LT_OQ),
                _mm256_cmp_pd(sb, zero, _CMP_GT_OQ));

            const int bits_a = _mm256_movemask_pd(mask_a);
            const int bits_b = _mm256_movemask_pd(mask_b);

            // Extract via 128-bit halves
            if (bits_a) {
                const __m128d r1a_lo = _mm256_castpd256_pd128(v_r1a);
                const __m128d r1a_hi = _mm256_extractf128_pd(v_r1a, 1);
                const __m128d r2a_lo = _mm256_castpd256_pd128(v_r2a);
                const __m128d r2a_hi = _mm256_extractf128_pd(v_r2a, 1);
                const __m128d sa_lo = _mm256_castpd256_pd128(sa);
                const __m128d sa_hi = _mm256_extractf128_pd(sa, 1);
                const double r1a[4] = {
                    _mm_cvtsd_f64(r1a_lo), _mm_cvtsd_f64(_mm_unpackhi_pd(r1a_lo, r1a_lo)),
                    _mm_cvtsd_f64(r1a_hi), _mm_cvtsd_f64(_mm_unpackhi_pd(r1a_hi, r1a_hi))};
                const double r2a[4] = {
                    _mm_cvtsd_f64(r2a_lo), _mm_cvtsd_f64(_mm_unpackhi_pd(r2a_lo, r2a_lo)),
                    _mm_cvtsd_f64(r2a_hi), _mm_cvtsd_f64(_mm_unpackhi_pd(r2a_hi, r2a_hi))};
                const double s_a[4] = {
                    _mm_cvtsd_f64(sa_lo), _mm_cvtsd_f64(_mm_unpackhi_pd(sa_lo, sa_lo)),
                    _mm_cvtsd_f64(sa_hi), _mm_cvtsd_f64(_mm_unpackhi_pd(sa_hi, sa_hi))};
                for (int lane = 0; lane < 4 && n < kBuf; ++lane) {
                    if (bits_a & (1 << lane)) {
                        buf_u1[n] = r1a[lane]; buf_u2[n] = r2a[lane]; buf_s[n] = s_a[lane]; ++n;
                    }
                }
            }
            if (bits_b) {
                const __m128d r1b_lo = _mm256_castpd256_pd128(v_r1b);
                const __m128d r1b_hi = _mm256_extractf128_pd(v_r1b, 1);
                const __m128d r2b_lo = _mm256_castpd256_pd128(v_r2b);
                const __m128d r2b_hi = _mm256_extractf128_pd(v_r2b, 1);
                const __m128d sb_lo = _mm256_castpd256_pd128(sb);
                const __m128d sb_hi = _mm256_extractf128_pd(sb, 1);
                const double r1b[4] = {
                    _mm_cvtsd_f64(r1b_lo), _mm_cvtsd_f64(_mm_unpackhi_pd(r1b_lo, r1b_lo)),
                    _mm_cvtsd_f64(r1b_hi), _mm_cvtsd_f64(_mm_unpackhi_pd(r1b_hi, r1b_hi))};
                const double r2b[4] = {
                    _mm_cvtsd_f64(r2b_lo), _mm_cvtsd_f64(_mm_unpackhi_pd(r2b_lo, r2b_lo)),
                    _mm_cvtsd_f64(r2b_hi), _mm_cvtsd_f64(_mm_unpackhi_pd(r2b_hi, r2b_hi))};
                const double s_b[4] = {
                    _mm_cvtsd_f64(sb_lo), _mm_cvtsd_f64(_mm_unpackhi_pd(sb_lo, sb_lo)),
                    _mm_cvtsd_f64(sb_hi), _mm_cvtsd_f64(_mm_unpackhi_pd(sb_hi, sb_hi))};
                for (int lane = 0; lane < 4 && n < kBuf; ++lane) {
                    if (bits_b & (1 << lane)) {
                        buf_u1[n] = r1b[lane]; buf_u2[n] = r2b[lane]; buf_s[n] = s_b[lane]; ++n;
                    }
                }
            }
        }

        // ── Phase 2: vectorized transform using libmvec log + hw sqrt ──
        const __m256d neg2 = _mm256_set1_pd(-2.0);
        int k = 0;
        for (; k + 4 <= n && i < count; k += 4) {
            const __m256d s_vec = _mm256_load_pd(buf_s + k);
            const __m256d log_s = _ZGVdN4v_log(s_vec);
            // factor = sqrt(-2 * log(s) / s)
            const __m256d factor = _mm256_sqrt_pd(
                _mm256_div_pd(_mm256_mul_pd(neg2, log_s), s_vec));
            const __m256d n1 = _mm256_mul_pd(_mm256_load_pd(buf_u1 + k), factor);
            const __m256d n2 = _mm256_mul_pd(_mm256_load_pd(buf_u2 + k), factor);

            // Interleave n1, n2 for paired output: [n1[0],n2[0],n1[1],n2[1],...]
            if (i + 8 <= count) {
                const __m256d lo = _mm256_unpacklo_pd(n1, n2);  // [n1[0],n2[0] | n1[2],n2[2]]
                const __m256d hi = _mm256_unpackhi_pd(n1, n2);  // [n1[1],n2[1] | n1[3],n2[3]]
                _mm256_storeu_pd(out + i,     _mm256_permute2f128_pd(lo, hi, 0x20));
                _mm256_storeu_pd(out + i + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
                i += 8;
            } else {
                // Scalar tail
                alignas(32) double n1_arr[4], n2_arr[4];
                _mm256_store_pd(n1_arr, n1);
                _mm256_store_pd(n2_arr, n2);
                for (int j = 0; j < 4 && i < count; ++j) {
                    out[i++] = n1_arr[j];
                    if (i < count) out[i++] = n2_arr[j];
                }
            }
        }
        // Scalar remainder (< 4 accepted pairs)
        for (; k < n && i < count; ++k) {
            const double scale = std::sqrt(-2.0 * std::log(buf_s[k]) / buf_s[k]);
            out[i++] = buf_u1[k] * scale;
            if (i < count) out[i++] = buf_u2[k] * scale;
        }
    }
#else
    fill_xoshiro256pp_x4_normal_polar_avx2(seed, out, count);
#endif
}

// ─── Idea 7: Fully vectorized polar (compute-and-mask, no batching) ──────────
// Compute the polar transform for ALL 4 lanes in SIMD — rejected lanes get
// blended to safe values before log. Then scatter only accepted results.
// No buffer, no phase separation — everything stays in YMM registers.
void fill_xoshiro256pp_x8_normal_vecpolar_avx2(std::uint64_t seed, double* out,
                                                 std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
    alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
    seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
    __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
    __m256i a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
    __m256i a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
    __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
    __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
    __m256i b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
    __m256i b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));

    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d zero = _mm256_setzero_pd();
    const __m256d neg2 = _mm256_set1_pd(-2.0);
    const __m256d safe_val = _mm256_set1_pd(0.5);  // safe for log when lane is rejected

    std::size_t i = 0;
    while (i < count) {
        // Process group a (4 lanes)
        {
            const __m256d u1 = u64_to_pm1_52_avx2(next_x4_avx2(a0, a1, a2, a3));
            const __m256d u2 = u64_to_pm1_52_avx2(next_x4_avx2(a0, a1, a2, a3));
            const __m256d s = _mm256_add_pd(_mm256_mul_pd(u1, u1), _mm256_mul_pd(u2, u2));

            const __m256d accept = _mm256_and_pd(
                _mm256_cmp_pd(s, one, _CMP_LT_OQ),
                _mm256_cmp_pd(s, zero, _CMP_GT_OQ));
            const int mask_bits = _mm256_movemask_pd(accept);

            if (mask_bits) {
                // Blend safe value for rejected lanes (avoids NaN/inf in log)
                const __m256d safe_s = _mm256_blendv_pd(safe_val, s, accept);
                const __m256d log_s = _ZGVdN4v_log(safe_s);
                const __m256d factor = _mm256_sqrt_pd(
                    _mm256_div_pd(_mm256_mul_pd(neg2, log_s), safe_s));
                const __m256d n1 = _mm256_mul_pd(u1, factor);
                const __m256d n2 = _mm256_mul_pd(u2, factor);

                // Extract via 128-bit halves and scatter accepted
                const __m128d n1_lo = _mm256_castpd256_pd128(n1);
                const __m128d n1_hi = _mm256_extractf128_pd(n1, 1);
                const __m128d n2_lo = _mm256_castpd256_pd128(n2);
                const __m128d n2_hi = _mm256_extractf128_pd(n2, 1);
                const double n1v[4] = {
                    _mm_cvtsd_f64(n1_lo), _mm_cvtsd_f64(_mm_unpackhi_pd(n1_lo, n1_lo)),
                    _mm_cvtsd_f64(n1_hi), _mm_cvtsd_f64(_mm_unpackhi_pd(n1_hi, n1_hi))};
                const double n2v[4] = {
                    _mm_cvtsd_f64(n2_lo), _mm_cvtsd_f64(_mm_unpackhi_pd(n2_lo, n2_lo)),
                    _mm_cvtsd_f64(n2_hi), _mm_cvtsd_f64(_mm_unpackhi_pd(n2_hi, n2_hi))};

                for (int lane = 0; lane < 4 && i < count; ++lane) {
                    if (mask_bits & (1 << lane)) {
                        out[i++] = n1v[lane];
                        if (i < count) out[i++] = n2v[lane];
                    }
                }
            }
        }

        if (i >= count) break;

        // Process group b (4 lanes)
        {
            const __m256d u1 = u64_to_pm1_52_avx2(next_x4_avx2(b0, b1, b2, b3));
            const __m256d u2 = u64_to_pm1_52_avx2(next_x4_avx2(b0, b1, b2, b3));
            const __m256d s = _mm256_add_pd(_mm256_mul_pd(u1, u1), _mm256_mul_pd(u2, u2));

            const __m256d accept = _mm256_and_pd(
                _mm256_cmp_pd(s, one, _CMP_LT_OQ),
                _mm256_cmp_pd(s, zero, _CMP_GT_OQ));
            const int mask_bits = _mm256_movemask_pd(accept);

            if (mask_bits) {
                const __m256d safe_s = _mm256_blendv_pd(safe_val, s, accept);
                const __m256d log_s = _ZGVdN4v_log(safe_s);
                const __m256d factor = _mm256_sqrt_pd(
                    _mm256_div_pd(_mm256_mul_pd(neg2, log_s), safe_s));
                const __m256d n1 = _mm256_mul_pd(u1, factor);
                const __m256d n2 = _mm256_mul_pd(u2, factor);

                const __m128d n1_lo = _mm256_castpd256_pd128(n1);
                const __m128d n1_hi = _mm256_extractf128_pd(n1, 1);
                const __m128d n2_lo = _mm256_castpd256_pd128(n2);
                const __m128d n2_hi = _mm256_extractf128_pd(n2, 1);
                const double n1v[4] = {
                    _mm_cvtsd_f64(n1_lo), _mm_cvtsd_f64(_mm_unpackhi_pd(n1_lo, n1_lo)),
                    _mm_cvtsd_f64(n1_hi), _mm_cvtsd_f64(_mm_unpackhi_pd(n1_hi, n1_hi))};
                const double n2v[4] = {
                    _mm_cvtsd_f64(n2_lo), _mm_cvtsd_f64(_mm_unpackhi_pd(n2_lo, n2_lo)),
                    _mm_cvtsd_f64(n2_hi), _mm_cvtsd_f64(_mm_unpackhi_pd(n2_hi, n2_hi))};

                for (int lane = 0; lane < 4 && i < count; ++lane) {
                    if (mask_bits & (1 << lane)) {
                        out[i++] = n1v[lane];
                        if (i < count) out[i++] = n2v[lane];
                    }
                }
            }
        }
    }
#else
    fill_xoshiro256pp_x4_normal_polar_avx2(seed, out, count);
#endif
}

// ─── Idea 8: x8 AVX2 Box-Muller with libmvec trig/log ───────────────────────
// Regular SIMD work per sample: no rejection, no lane compaction, and no
// masked scatter. This isolates whether a smooth transform beats polar's
// accept/reject machinery on this machine.
void fill_xoshiro256pp_x8_normal_box_muller_avx2(std::uint64_t seed, double* out,
                                                   std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
    alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
    seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
    __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
    __m256i a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
    __m256i a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
    __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
    __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
    __m256i b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
    __m256i b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));

    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d neg2 = _mm256_set1_pd(-2.0);
    const __m256d two_pi =
        _mm256_set1_pd(6.2831853071795864769252867665590058);

    std::size_t i = 0;
    while (i + 16 <= count) {
        const __m256d ur_a =
            _mm256_sub_pd(one, u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3)));
        const __m256d ut_a =
            u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3));
        const __m256d ur_b =
            _mm256_sub_pd(one, u64_to_uniform01_52_avx2(next_x4_avx2(b0, b1, b2, b3)));
        const __m256d ut_b =
            u64_to_uniform01_52_avx2(next_x4_avx2(b0, b1, b2, b3));

        const __m256d radius_a =
            _mm256_sqrt_pd(_mm256_mul_pd(neg2, _ZGVdN4v_log(ur_a)));
        const __m256d radius_b =
            _mm256_sqrt_pd(_mm256_mul_pd(neg2, _ZGVdN4v_log(ur_b)));
        const __m256d theta_a = _mm256_mul_pd(two_pi, ut_a);
        const __m256d theta_b = _mm256_mul_pd(two_pi, ut_b);

        const __m256d n1a = _mm256_mul_pd(radius_a, _ZGVdN4v_cos(theta_a));
        const __m256d n2a = _mm256_mul_pd(radius_a, _ZGVdN4v_sin(theta_a));
        const __m256d n1b = _mm256_mul_pd(radius_b, _ZGVdN4v_cos(theta_b));
        const __m256d n2b = _mm256_mul_pd(radius_b, _ZGVdN4v_sin(theta_b));

        _mm256_storeu_pd(out + i,      n1a);
        _mm256_storeu_pd(out + i + 4,  n2a);
        _mm256_storeu_pd(out + i + 8,  n1b);
        _mm256_storeu_pd(out + i + 12, n2b);
        i += 16;
    }

    while (i < count) {
        alignas(32) double n1[4], n2[4];

        const __m256d ur_a =
            _mm256_sub_pd(one, u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3)));
        const __m256d ut_a =
            u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3));
        const __m256d radius_a =
            _mm256_sqrt_pd(_mm256_mul_pd(neg2, _ZGVdN4v_log(ur_a)));
        const __m256d theta_a = _mm256_mul_pd(two_pi, ut_a);
        _mm256_store_pd(n1, _mm256_mul_pd(radius_a, _ZGVdN4v_cos(theta_a)));
        _mm256_store_pd(n2, _mm256_mul_pd(radius_a, _ZGVdN4v_sin(theta_a)));
        for (int lane = 0; lane < 4 && i < count; ++lane) out[i++] = n1[lane];
        for (int lane = 0; lane < 4 && i < count; ++lane) out[i++] = n2[lane];
        if (i >= count) break;

        const __m256d ur_b =
            _mm256_sub_pd(one, u64_to_uniform01_52_avx2(next_x4_avx2(b0, b1, b2, b3)));
        const __m256d ut_b =
            u64_to_uniform01_52_avx2(next_x4_avx2(b0, b1, b2, b3));
        const __m256d radius_b =
            _mm256_sqrt_pd(_mm256_mul_pd(neg2, _ZGVdN4v_log(ur_b)));
        const __m256d theta_b = _mm256_mul_pd(two_pi, ut_b);
        _mm256_store_pd(n1, _mm256_mul_pd(radius_b, _ZGVdN4v_cos(theta_b)));
        _mm256_store_pd(n2, _mm256_mul_pd(radius_b, _ZGVdN4v_sin(theta_b)));
        for (int lane = 0; lane < 4 && i < count; ++lane) out[i++] = n1[lane];
        for (int lane = 0; lane < 4 && i < count; ++lane) out[i++] = n2[lane];
    }
#else
    fill_xoshiro256pp_x4_normal_polar_avx2(seed, out, count);
#endif
}

// ─── Idea 5: Fused RNG+conversion for portable x4 ───────────────────────────
void fill_xoshiro256pp_x4_uniform01_fused(std::uint64_t seed, double* out,
                                           std::size_t count) noexcept {
    // Instead of: rng() → std::array → bits_to_01(arr[i]),
    // we inline the RNG step and convert directly, avoiding the
    // intermediate array materialization.
    std::uint64_t s0[4], s1[4], s2[4], s3[4];
    seed_lanes(seed, 4, s0, s1, s2, s3);

    std::size_t i = 0;
    while (i + 4 <= count) {
        // Compute result and convert in one fused loop
        for (int lane = 0; lane < 4; ++lane)
            out[i + lane] = static_cast<double>(
                (rotl64(s0[lane] + s3[lane], 23) + s0[lane]) >> 11) * kBitsTo01Scale;
        // State update
        std::uint64_t t[4];
        for (int lane = 0; lane < 4; ++lane)
            t[lane] = s1[lane] << 17;
        for (int lane = 0; lane < 4; ++lane)
            s2[lane] ^= s0[lane];
        for (int lane = 0; lane < 4; ++lane)
            s3[lane] ^= s1[lane];
        for (int lane = 0; lane < 4; ++lane)
            s1[lane] ^= s2[lane];
        for (int lane = 0; lane < 4; ++lane)
            s0[lane] ^= s3[lane];
        for (int lane = 0; lane < 4; ++lane)
            s2[lane] ^= t[lane];
        for (int lane = 0; lane < 4; ++lane)
            s3[lane] = rotl64(s3[lane], 45);
        i += 4;
    }
    if (i < count) {
        std::uint64_t result[4];
        next_x4(s0, s1, s2, s3, result);
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = static_cast<double>(result[lane] >> 11) * kBitsTo01Scale;
    }
}

// ─── Exponential: naive (generate uniforms, then scalar -log) ────────────────
void fill_xoshiro256pp_x8_exponential_naive(std::uint64_t seed, double* out,
                                             std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
    alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
    alignas(32) double tmp[4];
    seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
    __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
    __m256i a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
    __m256i a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
    __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
    __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
    __m256i b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
    __m256i b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));

    std::size_t i = 0;
    while (i + 8 <= count) {
        _mm256_store_pd(tmp, u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3)));
        out[i + 0] = -std::log(tmp[0]);
        out[i + 1] = -std::log(tmp[1]);
        out[i + 2] = -std::log(tmp[2]);
        out[i + 3] = -std::log(tmp[3]);
        _mm256_store_pd(tmp, u64_to_uniform01_52_avx2(next_x4_avx2(b0, b1, b2, b3)));
        out[i + 4] = -std::log(tmp[0]);
        out[i + 5] = -std::log(tmp[1]);
        out[i + 6] = -std::log(tmp[2]);
        out[i + 7] = -std::log(tmp[3]);
        i += 8;
    }
    while (i + 4 <= count) {
        _mm256_store_pd(tmp, u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3)));
        out[i + 0] = -std::log(tmp[0]);
        out[i + 1] = -std::log(tmp[1]);
        out[i + 2] = -std::log(tmp[2]);
        out[i + 3] = -std::log(tmp[3]);
        i += 4;
    }
    if (i < count) {
        _mm256_store_pd(tmp, u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3)));
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = -std::log(tmp[lane]);
    }
#else
    (void)seed; (void)out; (void)count;
#endif
}

// ─── Exponential: vectorized -log via libmvec ────────────────────────────────
void fill_xoshiro256pp_x8_exponential_avx2(std::uint64_t seed, double* out,
                                            std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
    alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
    seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
    __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
    __m256i a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
    __m256i a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
    __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
    __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
    __m256i b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
    __m256i b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));

    const __m256d neg1 = _mm256_set1_pd(-1.0);

    std::size_t i = 0;
    while (i + 8 <= count) {
        const __m256d ua = u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3));
        const __m256d ub = u64_to_uniform01_52_avx2(next_x4_avx2(b0, b1, b2, b3));
        // -log(u) = (-1) * log(u)
        _mm256_storeu_pd(out + i,     _mm256_mul_pd(neg1, _ZGVdN4v_log(ua)));
        _mm256_storeu_pd(out + i + 4, _mm256_mul_pd(neg1, _ZGVdN4v_log(ub)));
        i += 8;
    }
    while (i + 4 <= count) {
        const __m256d u = u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3));
        _mm256_storeu_pd(out + i, _mm256_mul_pd(neg1, _ZGVdN4v_log(u)));
        i += 4;
    }
    if (i < count) {
        alignas(32) double tmp[4];
        const __m256d u = u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3));
        _mm256_store_pd(tmp, _mm256_mul_pd(neg1, _ZGVdN4v_log(u)));
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = tmp[lane];
    }
#else
    (void)seed; (void)out; (void)count;
#endif
}

// ─── Bernoulli: naive (generate uniform doubles, SIMD compare against p) ─────
// This is what a user would naturally write: generate [0,1) doubles, then compare.
// The conversion to double (shift + OR + subtract) is the overhead we want to skip.
void fill_xoshiro256pp_x8_bernoulli_naive(std::uint64_t seed, double p,
                                           double* out,
                                           std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
    alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
    seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
    __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
    __m256i a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
    __m256i a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
    __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
    __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
    __m256i b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
    __m256i b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));

    const __m256d p_vec = _mm256_set1_pd(p);
    const __m256d one_d = _mm256_set1_pd(1.0);

    std::size_t i = 0;
    while (i + 8 <= count) {
        const __m256d ua = u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3));
        const __m256d ub = u64_to_uniform01_52_avx2(next_x4_avx2(b0, b1, b2, b3));
        // u < p → mask of all-ones, AND with 1.0 → 1.0, else 0.0
        _mm256_storeu_pd(out + i,
            _mm256_and_pd(_mm256_cmp_pd(ua, p_vec, _CMP_LT_OQ), one_d));
        _mm256_storeu_pd(out + i + 4,
            _mm256_and_pd(_mm256_cmp_pd(ub, p_vec, _CMP_LT_OQ), one_d));
        i += 8;
    }
    while (i + 4 <= count) {
        const __m256d ua = u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3));
        _mm256_storeu_pd(out + i,
            _mm256_and_pd(_mm256_cmp_pd(ua, p_vec, _CMP_LT_OQ), one_d));
        i += 4;
    }
    if (i < count) {
        alignas(32) double tmp[4];
        const __m256d ua = u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3));
        _mm256_store_pd(tmp,
            _mm256_and_pd(_mm256_cmp_pd(ua, p_vec, _CMP_LT_OQ), one_d));
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = tmp[lane];
    }
#else
    (void)seed; (void)p; (void)out; (void)count;
#endif
}

// ─── Bernoulli: fast (integer threshold, no float conversion) ────────────────
// Compare raw uint64 output directly against a precomputed threshold.
// The RNG output is uniform over [0, 2^64), so threshold = p * 2^64.
// We use unsigned comparison: raw < threshold ↔ Bernoulli(p) success.
// AVX2 only has signed cmpgt_epi64, so we flip the sign bit of both operands.
void fill_xoshiro256pp_x8_bernoulli_fast(std::uint64_t seed, double p,
                                          double* out,
                                          std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
    alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
    seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
    __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
    __m256i a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
    __m256i a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
    __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
    __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
    __m256i b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
    __m256i b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));

    const auto threshold = static_cast<std::uint64_t>(p * 0x1.0p64);
    const __m256i sign_flip = _mm256_set1_epi64x(
        static_cast<std::int64_t>(0x8000000000000000ULL));
    const __m256i thresh_vec = _mm256_set1_epi64x(
        static_cast<std::int64_t>(threshold ^ 0x8000000000000000ULL));
    const __m256d one_d = _mm256_set1_pd(1.0);

    std::size_t i = 0;
    while (i + 8 <= count) {
        const __m256i ra = next_x4_avx2(a0, a1, a2, a3);
        const __m256i rb = next_x4_avx2(b0, b1, b2, b3);
        const __m256i cmp_a = _mm256_cmpgt_epi64(
            thresh_vec, _mm256_xor_si256(ra, sign_flip));
        const __m256i cmp_b = _mm256_cmpgt_epi64(
            thresh_vec, _mm256_xor_si256(rb, sign_flip));
        _mm256_storeu_pd(out + i,
            _mm256_and_pd(_mm256_castsi256_pd(cmp_a), one_d));
        _mm256_storeu_pd(out + i + 4,
            _mm256_and_pd(_mm256_castsi256_pd(cmp_b), one_d));
        i += 8;
    }
    while (i + 4 <= count) {
        const __m256i ra = next_x4_avx2(a0, a1, a2, a3);
        const __m256i cmp_a = _mm256_cmpgt_epi64(
            thresh_vec, _mm256_xor_si256(ra, sign_flip));
        _mm256_storeu_pd(out + i,
            _mm256_and_pd(_mm256_castsi256_pd(cmp_a), one_d));
        i += 4;
    }
    if (i < count) {
        alignas(32) double tmp[4];
        const __m256i ra = next_x4_avx2(a0, a1, a2, a3);
        const __m256i cmp_a = _mm256_cmpgt_epi64(
            thresh_vec, _mm256_xor_si256(ra, sign_flip));
        _mm256_store_pd(tmp, _mm256_and_pd(_mm256_castsi256_pd(cmp_a), one_d));
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = tmp[lane];
    }
#else
    (void)seed; (void)p; (void)out; (void)count;
#endif
}

// ─── Bernoulli(0.5): bit-unpacking — each bit is an independent trial ────────
// We discard the low 12 bits of each uint64 (matching the uniform path's >> 12)
// because xoshiro256++'s lowest bits have the weakest linear complexity.
// That gives 52 usable bits per uint64, × 4 lanes = 208 samples per x4 call.
static constexpr int kBernoulliSkipBits = 12;
static constexpr int kBernoulliBitsPerLane = 64 - kBernoulliSkipBits;  // 52

void fill_xoshiro256pp_x8_bernoulli_half(std::uint64_t seed, double* out,
                                          std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
    alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
    seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
    __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
    __m256i a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
    __m256i a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
    __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
    __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
    __m256i b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
    __m256i b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));

    const __m256i one_i = _mm256_set1_epi64x(1);
    const __m256d one_d = _mm256_set1_pd(1.0);

    std::size_t i = 0;
    while (i < count) {
        // Discard low 12 bits, then extract bits [12..63] (52 bits per lane)
        __m256i bits = _mm256_srli_epi64(next_x4_avx2(a0, a1, a2, a3), kBernoulliSkipBits);

        for (int bit = 0; bit < kBernoulliBitsPerLane && i + 4 <= count; ++bit) {
            const __m256i cmp = _mm256_cmpeq_epi64(
                _mm256_and_si256(bits, one_i), one_i);
            _mm256_storeu_pd(out + i,
                _mm256_and_pd(_mm256_castsi256_pd(cmp), one_d));
            i += 4;
            bits = _mm256_srli_epi64(bits, 1);
        }
        if (i >= count) break;

        bits = _mm256_srli_epi64(next_x4_avx2(b0, b1, b2, b3), kBernoulliSkipBits);
        for (int bit = 0; bit < kBernoulliBitsPerLane && i + 4 <= count; ++bit) {
            const __m256i cmp = _mm256_cmpeq_epi64(
                _mm256_and_si256(bits, one_i), one_i);
            _mm256_storeu_pd(out + i,
                _mm256_and_pd(_mm256_castsi256_pd(cmp), one_d));
            i += 4;
            bits = _mm256_srli_epi64(bits, 1);
        }
    }
    // Handle tail (< 4 remaining)
    if (i < count) {
        alignas(32) double tmp[4];
        __m256i bits = _mm256_srli_epi64(next_x4_avx2(a0, a1, a2, a3), kBernoulliSkipBits);
        const __m256i cmp = _mm256_cmpeq_epi64(
            _mm256_and_si256(bits, one_i), one_i);
        _mm256_store_pd(tmp, _mm256_and_pd(_mm256_castsi256_pd(cmp), one_d));
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = tmp[lane];
    }
#else
    (void)seed; (void)out; (void)count;
#endif
}

// ─── Bernoulli → uint8_t output ──────────────────────────────────────────────

// Naive: generate uniform doubles, compare, store 0/1 as uint8_t.
void fill_xoshiro256pp_x8_bernoulli_u8_naive(std::uint64_t seed, double p,
                                              std::uint8_t* out,
                                              std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
    alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
    alignas(32) double tmp[4];
    seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
    __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
    __m256i a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
    __m256i a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
    __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
    __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
    __m256i b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
    __m256i b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));

    std::size_t i = 0;
    while (i + 8 <= count) {
        _mm256_store_pd(tmp, u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3)));
        out[i + 0] = tmp[0] < p ? 1 : 0;
        out[i + 1] = tmp[1] < p ? 1 : 0;
        out[i + 2] = tmp[2] < p ? 1 : 0;
        out[i + 3] = tmp[3] < p ? 1 : 0;
        _mm256_store_pd(tmp, u64_to_uniform01_52_avx2(next_x4_avx2(b0, b1, b2, b3)));
        out[i + 4] = tmp[0] < p ? 1 : 0;
        out[i + 5] = tmp[1] < p ? 1 : 0;
        out[i + 6] = tmp[2] < p ? 1 : 0;
        out[i + 7] = tmp[3] < p ? 1 : 0;
        i += 8;
    }
    while (i + 4 <= count) {
        _mm256_store_pd(tmp, u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3)));
        for (int lane = 0; lane < 4; ++lane)
            out[i + lane] = tmp[lane] < p ? 1 : 0;
        i += 4;
    }
    if (i < count) {
        _mm256_store_pd(tmp, u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3)));
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = tmp[lane] < p ? 1 : 0;
    }
#else
    (void)seed; (void)p; (void)out; (void)count;
#endif
}

// Fast: integer threshold compare → pack 4 results into low bytes of __m256i,
// then narrow to uint8_t via _mm256_packs_epi32 + _mm256_packs_epi16.
void fill_xoshiro256pp_x8_bernoulli_u8_fast(std::uint64_t seed, double p,
                                             std::uint8_t* out,
                                             std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
    alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
    seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
    __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
    __m256i a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
    __m256i a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
    __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
    __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
    __m256i b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
    __m256i b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));

    const auto threshold = static_cast<std::uint64_t>(p * 0x1.0p64);
    const __m256i sign_flip = _mm256_set1_epi64x(
        static_cast<std::int64_t>(0x8000000000000000ULL));
    const __m256i thresh_vec = _mm256_set1_epi64x(
        static_cast<std::int64_t>(threshold ^ 0x8000000000000000ULL));
    const __m256i one_i64 = _mm256_set1_epi64x(1);

    // cmpgt produces 0 or -1 (0xFFFF...) per 64-bit lane.
    // We AND with 1 to get 0 or 1, then pack 4×i64 → movemask or narrow.
    // 8 RNG calls → 32 results, packed into 32 bytes via shuffle.
    std::size_t i = 0;
    while (i + 32 <= count) {
        // 8 x4 calls = 32 samples
        __m256i r0 = _mm256_and_si256(one_i64, _mm256_cmpgt_epi64(thresh_vec, _mm256_xor_si256(next_x4_avx2(a0, a1, a2, a3), sign_flip)));
        __m256i r1 = _mm256_and_si256(one_i64, _mm256_cmpgt_epi64(thresh_vec, _mm256_xor_si256(next_x4_avx2(b0, b1, b2, b3), sign_flip)));
        __m256i r2 = _mm256_and_si256(one_i64, _mm256_cmpgt_epi64(thresh_vec, _mm256_xor_si256(next_x4_avx2(a0, a1, a2, a3), sign_flip)));
        __m256i r3 = _mm256_and_si256(one_i64, _mm256_cmpgt_epi64(thresh_vec, _mm256_xor_si256(next_x4_avx2(b0, b1, b2, b3), sign_flip)));
        __m256i r4 = _mm256_and_si256(one_i64, _mm256_cmpgt_epi64(thresh_vec, _mm256_xor_si256(next_x4_avx2(a0, a1, a2, a3), sign_flip)));
        __m256i r5 = _mm256_and_si256(one_i64, _mm256_cmpgt_epi64(thresh_vec, _mm256_xor_si256(next_x4_avx2(b0, b1, b2, b3), sign_flip)));
        __m256i r6 = _mm256_and_si256(one_i64, _mm256_cmpgt_epi64(thresh_vec, _mm256_xor_si256(next_x4_avx2(a0, a1, a2, a3), sign_flip)));
        __m256i r7 = _mm256_and_si256(one_i64, _mm256_cmpgt_epi64(thresh_vec, _mm256_xor_si256(next_x4_avx2(b0, b1, b2, b3), sign_flip)));

        // Pack 4×i64{0,1} → 4×i32 → 4×i16 → 4×i8
        // _mm256_packs_epi32 works on 128-bit halves independently (lane-crossing).
        // Simpler: use shuffle bytes to gather the low byte of each 64-bit element.
        // Each 256-bit reg has 4 uint64 values (0 or 1). The '1' is in byte 0
        // of each 8-byte group. We want bytes [0, 8, 16, 24] → 4 output bytes.
        // Use _mm256_shuffle_epi8 to collect them, then extract.

        // Shuffle mask: grab byte 0 of each 64-bit element within each 128-bit lane
        // Lane0: bytes [0, 8] → positions [0, 1]  (within 128-bit half)
        // Lane1: bytes [0, 8] → positions [0, 1]
        // Then permute to combine.
        // Easier approach: just use _mm256_extract_epi8 or store + memcpy.
        // Actually, the fastest for 32 bytes: store all 8 regs via movemask trick.

        // Extract low byte of each 64-bit element via shuffle.
        const __m256i shuf = _mm256_set_epi8(
            -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, 24,16,8,0,
            -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, 24,16,8,0);
        // This puts the 4 low bytes into positions [0..3] of each 128-bit lane.
        __m256i s0 = _mm256_shuffle_epi8(r0, shuf);  // [B0,B1,B2,B3, 0..0 | B0,B1,B2,B3, 0..0]
        __m256i s1 = _mm256_shuffle_epi8(r1, shuf);
        __m256i s2 = _mm256_shuffle_epi8(r2, shuf);
        __m256i s3 = _mm256_shuffle_epi8(r3, shuf);
        __m256i s4 = _mm256_shuffle_epi8(r4, shuf);
        __m256i s5 = _mm256_shuffle_epi8(r5, shuf);
        __m256i s6 = _mm256_shuffle_epi8(r6, shuf);
        __m256i s7 = _mm256_shuffle_epi8(r7, shuf);

        // Each sx has 4 useful bytes in low 128-bit half, positions [0..3].
        // Combine: s0[0..3] + s1[0..3] → 8 bytes, etc.
        // Use unpacklo to interleave i32s, building up to 32 bytes.
        // s0 low lane: [b0 b1 b2 b3 0 0 0 0 0 0 0 0 0 0 0 0]
        // s1 low lane: [b4 b5 b6 b7 0 0 0 0 0 0 0 0 0 0 0 0]
        // We want: [b0 b1 b2 b3 b4 b5 b6 b7 ...]

        // Extract low 32-bits of each as scalar and build output.
        // _mm256_extract_epi32(sx, 0) gives us the 4 packed bytes as one int32.
        const auto w0 = static_cast<std::uint32_t>(_mm256_extract_epi32(s0, 0));
        const auto w1 = static_cast<std::uint32_t>(_mm256_extract_epi32(s1, 0));
        const auto w2 = static_cast<std::uint32_t>(_mm256_extract_epi32(s2, 0));
        const auto w3 = static_cast<std::uint32_t>(_mm256_extract_epi32(s3, 0));
        const auto w4 = static_cast<std::uint32_t>(_mm256_extract_epi32(s4, 0));
        const auto w5 = static_cast<std::uint32_t>(_mm256_extract_epi32(s5, 0));
        const auto w6 = static_cast<std::uint32_t>(_mm256_extract_epi32(s6, 0));
        const auto w7 = static_cast<std::uint32_t>(_mm256_extract_epi32(s7, 0));

        std::memcpy(out + i +  0, &w0, 4);
        std::memcpy(out + i +  4, &w1, 4);
        std::memcpy(out + i +  8, &w2, 4);
        std::memcpy(out + i + 12, &w3, 4);
        std::memcpy(out + i + 16, &w4, 4);
        std::memcpy(out + i + 20, &w5, 4);
        std::memcpy(out + i + 24, &w6, 4);
        std::memcpy(out + i + 28, &w7, 4);
        i += 32;
    }
    // Tail: scalar fallback
    while (i + 4 <= count) {
        const __m256i ra = next_x4_avx2(a0, a1, a2, a3);
        const int bits = _mm256_movemask_pd(_mm256_castsi256_pd(
            _mm256_cmpgt_epi64(thresh_vec, _mm256_xor_si256(ra, sign_flip))));
        for (int lane = 0; lane < 4; ++lane)
            out[i + lane] = (bits >> lane) & 1;
        i += 4;
    }
    if (i < count) {
        const __m256i ra = next_x4_avx2(a0, a1, a2, a3);
        const int bits = _mm256_movemask_pd(_mm256_castsi256_pd(
            _mm256_cmpgt_epi64(thresh_vec, _mm256_xor_si256(ra, sign_flip))));
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = (bits >> lane) & 1;
    }
#else
    (void)seed; (void)p; (void)out; (void)count;
#endif
}

// p=0.5 bit-unpack to uint8_t: each bit of raw uint64 → one byte (0 or 1).
// Discards low 12 bits for quality (matching uniform path). 52 usable bits per
// uint64, × 4 lanes = 208 samples per x4 call. Uses byte-broadcast + bit-mask
// SIMD to unpack 8 bits → 8 bytes in one shot.
// Templated on SkipBits so we can benchmark skip=12 vs skip=16.
// skip=12 → 52 usable bits → 6 full bytes + 4-bit scalar tail
// skip=16 → 48 usable bits → 6 full bytes, no tail (exact multiple of 8)
template <int SkipBits>
void fill_bernoulli_u8_half_impl(std::uint64_t seed,
                                  std::uint8_t* out,
                                  std::size_t count) noexcept {
#ifdef __AVX2__
    static constexpr int kUsableBits = 64 - SkipBits;
    static constexpr int kFullBytes = kUsableBits / 8;
    static constexpr int kRemainBits = kUsableBits % 8;

    alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
    alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
    seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
    __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
    __m256i a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
    __m256i a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
    __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
    __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
    __m256i b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
    __m256i b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));

    const __m256i bit_mask = _mm256_set_epi8(
        -128, 64, 32, 16, 8, 4, 2, 1,  -128, 64, 32, 16, 8, 4, 2, 1,
        -128, 64, 32, 16, 8, 4, 2, 1,  -128, 64, 32, 16, 8, 4, 2, 1);
    const __m256i one_byte = _mm256_set1_epi8(1);

    alignas(32) std::uint64_t raw[4];
    std::size_t i = 0;

    auto unpack_one_call = [&](__m256i& s0, __m256i& s1, __m256i& s2, __m256i& s3)
        __attribute__((always_inline)) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(raw),
                           _mm256_srli_epi64(next_x4_avx2(s0, s1, s2, s3), SkipBits));

        for (int lane = 0; lane < 4 && i < count; ++lane) {
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(&raw[lane]);
            for (int b = 0; b < kFullBytes && i + 8 <= count; ++b) {
                const __m256i bcast = _mm256_set1_epi8(static_cast<char>(bytes[b]));
                const __m256i isolated = _mm256_and_si256(bcast, bit_mask);
                const __m256i nonzero = _mm256_min_epu8(isolated, one_byte);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(out + i),
                    _mm256_castsi256_si128(nonzero));
                i += 8;
            }
            if constexpr (kRemainBits > 0) {
                if (i < count) {
                    std::uint8_t tail_byte = bytes[kFullBytes];
                    for (int bit = 0; bit < kRemainBits && i < count; ++bit) {
                        out[i++] = tail_byte & 1;
                        tail_byte >>= 1;
                    }
                }
            }
        }
    };

    while (i < count) {
        unpack_one_call(a0, a1, a2, a3);
        if (i >= count) break;
        unpack_one_call(b0, b1, b2, b3);
    }
#else
    (void)seed; (void)out; (void)count;
#endif
}

void fill_xoshiro256pp_x8_bernoulli_u8_half(std::uint64_t seed,
                                             std::uint8_t* out,
                                             std::size_t count) noexcept {
    fill_bernoulli_u8_half_impl<12>(seed, out, count);
}

void fill_xoshiro256pp_x8_bernoulli_u8_half_skip16(std::uint64_t seed,
                                                    std::uint8_t* out,
                                                    std::size_t count) noexcept {
    fill_bernoulli_u8_half_impl<16>(seed, out, count);
}

// ─── Bernoulli(0.5) → packed bitmask ─────────────────────────────────────────
// Raw RNG output >> 12, stored directly. Each uint64_t has 52 quality bits.
// count is in samples (bits). We fill ceil(count/208) × 8 words, each shifted.
void fill_xoshiro256pp_x8_bernoulli_bits(std::uint64_t seed,
                                          std::uint64_t* out,
                                          std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
    alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
    seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
    __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
    __m256i a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
    __m256i a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
    __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
    __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
    __m256i b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
    __m256i b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));

    // Each x4 call produces 4 uint64_t, each with 52 usable bits = 208 samples.
    // We need ceil(count / 208) calls. Write 4 words per call.
    static constexpr int kBitsPerWord = 52;
    static constexpr int kWordsPerCall = 4;
    static constexpr int kSamplesPerCall = kBitsPerWord * kWordsPerCall;  // 208

    std::size_t samples_done = 0;
    std::size_t wi = 0;

    while (samples_done + 2 * kSamplesPerCall <= count) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + wi),
            _mm256_srli_epi64(next_x4_avx2(a0, a1, a2, a3), kBernoulliSkipBits));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + wi + 4),
            _mm256_srli_epi64(next_x4_avx2(b0, b1, b2, b3), kBernoulliSkipBits));
        wi += 8;
        samples_done += 2 * kSamplesPerCall;
    }
    if (samples_done + kSamplesPerCall <= count) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + wi),
            _mm256_srli_epi64(next_x4_avx2(a0, a1, a2, a3), kBernoulliSkipBits));
        wi += 4;
        samples_done += kSamplesPerCall;
    }
    // Tail: partial call
    if (samples_done < count) {
        alignas(32) std::uint64_t tmp[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(tmp),
            _mm256_srli_epi64(next_x4_avx2(a0, a1, a2, a3), kBernoulliSkipBits));
        for (int lane = 0; lane < 4 && samples_done < count; ++lane) {
            out[wi++] = tmp[lane];
            samples_done += kBitsPerWord;
        }
    }
#else
    (void)seed; (void)out; (void)count;
#endif
}

// ─── Gamma(alpha, 1) — Marsaglia-Tsang algorithm ─────────────────────────────
//
// For shape alpha >= 1:  d = alpha − 1/3,  c = 1/√(9d)
//   repeat:
//     x  = N(0,1)  [polar method]
//     v  = (1 + c·x)³;  reject if v ≤ 0
//     u  = Uniform(0,1)
//     fast-accept if u < 1 − 0.0331·x⁴               (~80 % of trials)
//     slow-accept if log(u) < x²/2 + d·(1 − v + log v)
//
// Benchmarked at alpha = 2 → d = 5/3, c ≈ 0.447, MT acceptance ≈ 97 %.
//
// The thesis: "fused" keeps all RNG state in registers / L1; "decoupled"
// writes intermediate normal and uniform buffers to heap, then reads them back,
// adding ≈ 3 × count × 8 bytes of extra memory traffic.

// ── scalar fused ─────────────────────────────────────────────────────────────
// Single xoshiro256++ stream: polar N(0,1) and MT uniform drawn back-to-back.
void fill_gamma_scalar_fused(std::uint64_t seed, double alpha, double* out,
                              std::size_t count) noexcept {
    std::uint64_t s[4] = {
        splitmix64(seed), splitmix64(seed), splitmix64(seed), splitmix64(seed)};
    const double d = alpha - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    std::size_t i = 0;
    while (i < count) {
        double x;
        for (;;) {
            const double u1 = static_cast<double>(
                static_cast<std::int64_t>(next_pp(s)) >> 11) * 0x1.0p-52;
            const double u2 = static_cast<double>(
                static_cast<std::int64_t>(next_pp(s)) >> 11) * 0x1.0p-52;
            const double sq = u1 * u1 + u2 * u2;
            if (sq >= 1.0 || sq == 0.0) continue;
            x = u1 * std::sqrt(-2.0 * std::log(sq) / sq);
            break;
        }
        const double vr = 1.0 + c * x;
        if (vr <= 0.0) continue;
        const double v  = vr * vr * vr;
        const double u  = static_cast<double>(next_pp(s) >> 11) * 0x1.0p-53;
        const double x2 = x * x;
        if (u < 1.0 - 0.0331 * x2 * x2) { out[i++] = d * v; continue; }
        if (std::log(u) < 0.5 * x2 + d * (1.0 - v + std::log(v))) { out[i++] = d * v; }
    }
}

// ── x8 AVX2 fused ────────────────────────────────────────────────────────────
// Dual x4 streams (a, b) drive vecpolar for N(0,1); a 9th independent scalar
// stream draws the MT acceptance uniform.  All three streams advance in the
// same outer loop — buf_x (64 doubles = 512 B) stays in L1 throughout.
void fill_gamma_x8_avx2_fused(std::uint64_t seed, double alpha, double* out,
                               std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
    alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
    seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
    __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
    __m256i a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
    __m256i a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
    __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
    __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
    __m256i b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
    __m256i b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));

    // 9th independent stream: same base seed, then 8 long-jumps past a/b.
    std::uint64_t sc[4];
    {
        std::uint64_t cseed = seed;
        sc[0] = splitmix64(cseed); sc[1] = splitmix64(cseed);
        sc[2] = splitmix64(cseed); sc[3] = splitmix64(cseed);
        for (int j = 0; j < 8; ++j) long_jump(sc);
    }

    const double d = alpha - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);

    static constexpr int kBuf = 64;
    alignas(32) double buf_x[kBuf];  // accepted N(0,1) values; fits in L1

    const __m256d one  = _mm256_set1_pd(1.0);
    const __m256d zero = _mm256_setzero_pd();
    const __m256d neg2 = _mm256_set1_pd(-2.0);
    const __m256d safe = _mm256_set1_pd(0.5);  // safe substitute for rejected lanes

    std::size_t i = 0;
    while (i < count) {
        // ── Phase 1: fill buf_x with kBuf N(0,1) values via x8 vecpolar ──────
        int n = 0;
        while (n < kBuf) {
            // Group A (4 lanes)
            const __m256d u1a = u64_to_pm1_52_avx2(next_x4_avx2(a0, a1, a2, a3));
            const __m256d u2a = u64_to_pm1_52_avx2(next_x4_avx2(a0, a1, a2, a3));
            const __m256d sq_a = _mm256_add_pd(_mm256_mul_pd(u1a, u1a),
                                               _mm256_mul_pd(u2a, u2a));
            const __m256d acc_a = _mm256_and_pd(_mm256_cmp_pd(sq_a, one, _CMP_LT_OQ),
                                                _mm256_cmp_pd(sq_a, zero, _CMP_GT_OQ));
            const int bits_a = _mm256_movemask_pd(acc_a);
            if (bits_a) {
                const __m256d sf_a = _mm256_blendv_pd(safe, sq_a, acc_a);
                const __m256d fa   = _mm256_sqrt_pd(
                    _mm256_div_pd(_mm256_mul_pd(neg2, _ZGVdN4v_log(sf_a)), sf_a));
                alignas(32) double xa[4], xa2[4];
                _mm256_store_pd(xa,  _mm256_mul_pd(u1a, fa));
                _mm256_store_pd(xa2, _mm256_mul_pd(u2a, fa));
                for (int lane = 0; lane < 4 && n < kBuf; ++lane) {
                    if (bits_a & (1 << lane)) {
                        buf_x[n++] = xa[lane];
                        if (n < kBuf) buf_x[n++] = xa2[lane];
                    }
                }
            }
            // Group B (4 lanes)
            const __m256d u1b = u64_to_pm1_52_avx2(next_x4_avx2(b0, b1, b2, b3));
            const __m256d u2b = u64_to_pm1_52_avx2(next_x4_avx2(b0, b1, b2, b3));
            const __m256d sq_b = _mm256_add_pd(_mm256_mul_pd(u1b, u1b),
                                               _mm256_mul_pd(u2b, u2b));
            const __m256d acc_b = _mm256_and_pd(_mm256_cmp_pd(sq_b, one, _CMP_LT_OQ),
                                                _mm256_cmp_pd(sq_b, zero, _CMP_GT_OQ));
            const int bits_b = _mm256_movemask_pd(acc_b);
            if (bits_b) {
                const __m256d sf_b = _mm256_blendv_pd(safe, sq_b, acc_b);
                const __m256d fb   = _mm256_sqrt_pd(
                    _mm256_div_pd(_mm256_mul_pd(neg2, _ZGVdN4v_log(sf_b)), sf_b));
                alignas(32) double xb[4], xb2[4];
                _mm256_store_pd(xb,  _mm256_mul_pd(u1b, fb));
                _mm256_store_pd(xb2, _mm256_mul_pd(u2b, fb));
                for (int lane = 0; lane < 4 && n < kBuf; ++lane) {
                    if (bits_b & (1 << lane)) {
                        buf_x[n++] = xb[lane];
                        if (n < kBuf) buf_x[n++] = xb2[lane];
                    }
                }
            }
        }
        // ── Phase 2: MT acceptance; uniform drawn from scalar stream sc ───────
        for (int k = 0; k < n && i < count; ++k) {
            const double x  = buf_x[k];
            const double vr = 1.0 + c * x;
            if (vr <= 0.0) continue;
            const double v  = vr * vr * vr;
            const double u  = static_cast<double>(next_pp(sc) >> 11) * 0x1.0p-53;
            const double x2 = x * x;
            if (u < 1.0 - 0.0331 * x2 * x2) { out[i++] = d * v; continue; }
            if (std::log(u) < 0.5 * x2 + d * (1.0 - v + std::log(v))) { out[i++] = d * v; }
        }
    }
#else
    fill_gamma_scalar_fused(seed, alpha, out, count);
#endif
}

// ── x8 AVX2 decoupled: pre-fill normal and uniform buffers, then apply MT ────
// Pass 1 writes ~1.25×count normals to heap.
// Pass 2 writes ~1.25×count uniforms to heap.
// Pass 3 reads both back and applies MT.
// Demonstrates the memory-bandwidth cost of separating generation from transform.
void fill_gamma_x8_avx2_decoupled(std::uint64_t seed, double alpha, double* out,
                                   std::size_t count) noexcept {
    const std::size_t n_over = count + (count >> 2) + 16;  // 1.25× + headroom
    std::vector<double> norm_buf(n_over), unif_buf(n_over);

    fill_xoshiro256pp_x8_normal_vecpolar_avx2(seed, norm_buf.data(), n_over);
    fill_xoshiro256pp_x8_uniform01_avx2(seed ^ 0x9e3779b97f4a7c15ULL,
                                        unif_buf.data(), n_over);

    const double d = alpha - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    std::size_t j = 0, k = 0, i = 0;
    while (i < count) {
        const double x  = norm_buf[j++];
        const double vr = 1.0 + c * x;
        if (vr <= 0.0) continue;
        const double v  = vr * vr * vr;
        const double u  = unif_buf[k++];
        const double x2 = x * x;
        if (u < 1.0 - 0.0331 * x2 * x2) { out[i++] = d * v; continue; }
        if (std::log(u) < 0.5 * x2 + d * (1.0 - v + std::log(v))) { out[i++] = d * v; }
    }
}

// ── x8+x4 AVX2 full: vectorized MT phase via a 4-wide AVX2 uniform stream ────
// Phase 1: x8 vecpolar → buf_x (N(0,1) normals), same as fused above.
// Phase 2: 4-wide AVX2 stream C draws uniforms; compute v^3, fast test, and
//          both log(u)+log(v) via veclog unconditionally — avoids scalar loop.
// Calling veclog for all 4 lanes regardless of fast-accept wastes ~25 % log
// work but eliminates serial dependency chains and scalar loop overhead.
void fill_gamma_x8_avx2_full(std::uint64_t seed, double alpha, double* out,
                              std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
    alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
    seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
    __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
    __m256i a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
    __m256i a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
    __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
    __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
    __m256i b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
    __m256i b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));

    // 9th–12th streams (one x4 AVX2 state) for MT acceptance uniforms.
    alignas(32) std::uint64_t sc0[4], sc1[4], sc2[4], sc3[4];
    {
        std::uint64_t all0[12], all1[12], all2[12], all3[12];
        seed_lanes(seed, 12, all0, all1, all2, all3);
        for (int lane = 0; lane < 4; ++lane) {
            sc0[lane] = all0[lane + 8]; sc1[lane] = all1[lane + 8];
            sc2[lane] = all2[lane + 8]; sc3[lane] = all3[lane + 8];
        }
    }
    __m256i c0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sc0));
    __m256i c1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sc1));
    __m256i c2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sc2));
    __m256i c3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sc3));

    const double d = alpha - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);

    const __m256d one      = _mm256_set1_pd(1.0);
    const __m256d zero     = _mm256_setzero_pd();
    const __m256d neg2     = _mm256_set1_pd(-2.0);
    const __m256d half     = _mm256_set1_pd(0.5);
    const __m256d safe     = _mm256_set1_pd(0.5);
    const __m256d d_vec    = _mm256_set1_pd(d);
    const __m256d c_vec    = _mm256_set1_pd(c);
    const __m256d mt_coeff = _mm256_set1_pd(0.0331);

    static constexpr int kBuf = 64;
    alignas(32) double buf_x[kBuf];

    std::size_t i = 0;
    while (i < count) {
        // ── Phase 1: fill buf_x with kBuf N(0,1) via x8 vecpolar ─────────────
        int n = 0;
        while (n < kBuf) {
            const __m256d u1a = u64_to_pm1_52_avx2(next_x4_avx2(a0, a1, a2, a3));
            const __m256d u2a = u64_to_pm1_52_avx2(next_x4_avx2(a0, a1, a2, a3));
            const __m256d sq_a = _mm256_add_pd(_mm256_mul_pd(u1a, u1a),
                                               _mm256_mul_pd(u2a, u2a));
            const __m256d acc_a = _mm256_and_pd(_mm256_cmp_pd(sq_a, one, _CMP_LT_OQ),
                                                _mm256_cmp_pd(sq_a, zero, _CMP_GT_OQ));
            const int bits_a = _mm256_movemask_pd(acc_a);
            if (bits_a) {
                const __m256d sf_a = _mm256_blendv_pd(safe, sq_a, acc_a);
                const __m256d fa   = _mm256_sqrt_pd(
                    _mm256_div_pd(_mm256_mul_pd(neg2, _ZGVdN4v_log(sf_a)), sf_a));
                alignas(32) double xa[4], xa2[4];
                _mm256_store_pd(xa,  _mm256_mul_pd(u1a, fa));
                _mm256_store_pd(xa2, _mm256_mul_pd(u2a, fa));
                for (int lane = 0; lane < 4 && n < kBuf; ++lane) {
                    if (bits_a & (1 << lane)) {
                        buf_x[n++] = xa[lane];
                        if (n < kBuf) buf_x[n++] = xa2[lane];
                    }
                }
            }
            const __m256d u1b = u64_to_pm1_52_avx2(next_x4_avx2(b0, b1, b2, b3));
            const __m256d u2b = u64_to_pm1_52_avx2(next_x4_avx2(b0, b1, b2, b3));
            const __m256d sq_b = _mm256_add_pd(_mm256_mul_pd(u1b, u1b),
                                               _mm256_mul_pd(u2b, u2b));
            const __m256d acc_b = _mm256_and_pd(_mm256_cmp_pd(sq_b, one, _CMP_LT_OQ),
                                                _mm256_cmp_pd(sq_b, zero, _CMP_GT_OQ));
            const int bits_b = _mm256_movemask_pd(acc_b);
            if (bits_b) {
                const __m256d sf_b = _mm256_blendv_pd(safe, sq_b, acc_b);
                const __m256d fb   = _mm256_sqrt_pd(
                    _mm256_div_pd(_mm256_mul_pd(neg2, _ZGVdN4v_log(sf_b)), sf_b));
                alignas(32) double xb[4], xb2[4];
                _mm256_store_pd(xb,  _mm256_mul_pd(u1b, fb));
                _mm256_store_pd(xb2, _mm256_mul_pd(u2b, fb));
                for (int lane = 0; lane < 4 && n < kBuf; ++lane) {
                    if (bits_b & (1 << lane)) {
                        buf_x[n++] = xb[lane];
                        if (n < kBuf) buf_x[n++] = xb2[lane];
                    }
                }
            }
        }

        // ── Phase 2: vectorized MT — 4-wide AVX2 uniform stream C ─────────────
        int k = 0;
        for (; k + 4 <= n && i < count; k += 4) {
            const __m256d x  = _mm256_load_pd(buf_x + k);
            const __m256d u  = u64_to_uniform01_52_avx2(next_x4_avx2(c0, c1, c2, c3));

            // v = (1 + c*x)^3; blend safe value (1.0) for lanes with vr ≤ 0
            const __m256d vr      = _mm256_add_pd(one, _mm256_mul_pd(c_vec, x));
            const __m256d vr_pos  = _mm256_cmp_pd(vr, zero, _CMP_GT_OQ);
            const __m256d safe_vr = _mm256_blendv_pd(one, vr, vr_pos);
            const __m256d v       = _mm256_mul_pd(safe_vr, _mm256_mul_pd(safe_vr, safe_vr));

            // Fast test (no log): u < 1 − 0.0331·x⁴
            const __m256d x2          = _mm256_mul_pd(x, x);
            const __m256d x4          = _mm256_mul_pd(x2, x2);
            const __m256d fast_thresh = _mm256_sub_pd(one, _mm256_mul_pd(mt_coeff, x4));
            const __m256d fast_acc    = _mm256_cmp_pd(u, fast_thresh, _CMP_LT_OQ);

            // Slow test: log(u) < x²/2 + d·(1 − v + log v)
            // Call veclog unconditionally; rejected lanes' results are masked out.
            const __m256d log_u   = _ZGVdN4v_log(u);          // u ∈ (0,1): always valid
            const __m256d log_v   = _ZGVdN4v_log(v);          // safe_vr ≥ 1: always valid
            const __m256d slow_rhs = _mm256_add_pd(
                _mm256_mul_pd(half, x2),
                _mm256_mul_pd(d_vec, _mm256_add_pd(_mm256_sub_pd(one, v), log_v)));
            const __m256d slow_acc = _mm256_cmp_pd(log_u, slow_rhs, _CMP_LT_OQ);

            // Accept = (fast OR slow) AND vr > 0
            const __m256d accept     = _mm256_and_pd(vr_pos, _mm256_or_pd(fast_acc, slow_acc));
            const int     accept_bits = _mm256_movemask_pd(accept);

            if (accept_bits) {
                alignas(32) double gv[4];
                _mm256_store_pd(gv, _mm256_mul_pd(d_vec, v));
                for (int lane = 0; lane < 4 && i < count; ++lane) {
                    if (accept_bits & (1 << lane))
                        out[i++] = gv[lane];
                }
            }
        }
        // Scalar tail for any remaining < 4 entries in buf_x this batch
        for (; k < n && i < count; ++k) {
            const double x  = buf_x[k];
            const double vr = 1.0 + c * x;
            if (vr <= 0.0) continue;
            const double v  = vr * vr * vr;
            // Reuse stream c0..c3 for the scalar tail via next_pp on sc
            alignas(32) std::uint64_t sc_snap[4];
            _mm256_store_si256(reinterpret_cast<__m256i*>(sc_snap), c0);
            const double u  = static_cast<double>(
                (rotl64(sc_snap[0] + sc_snap[3], 23) + sc_snap[0]) >> 11) * 0x1.0p-53;
            c0 = next_x4_avx2(c0, c1, c2, c3);  // advance one step
            const double x2 = x * x;
            if (u < 1.0 - 0.0331 * x2 * x2) { out[i++] = d * v; continue; }
            if (std::log(u) < 0.5 * x2 + d * (1.0 - v + std::log(v))) { out[i++] = d * v; }
        }
    }
#else
    fill_gamma_scalar_fused(seed, alpha, out, count);
#endif
}

// ─── Student's t(nu) — t = N(0,1) / sqrt(2·Gamma(nu/2, 1)/nu) ───────────────
//
// Benchmarked at nu = 5  →  Gamma shape = 2.5, d = 13/6, acceptance ≈ 98 %.
// Each output sample needs: one N(0,1) for Z plus one Gamma(nu/2) for V.
//
// Fused: Z and V generated in the same loop from the same stream(s).
// Decoupled: pre-fill a Z buffer and a V buffer, then combine — extra 2 passes.

// ── scalar fused ─────────────────────────────────────────────────────────────
void fill_student_t_scalar_fused(std::uint64_t seed, double nu, double* out,
                                  std::size_t count) noexcept {
    std::uint64_t s[4] = {
        splitmix64(seed), splitmix64(seed), splitmix64(seed), splitmix64(seed)};
    const double shape = nu / 2.0;
    const double d = shape - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);

    std::size_t i = 0;
    while (i < count) {
        // Z ~ N(0,1) for numerator
        double z;
        for (;;) {
            const double r1 = static_cast<double>(
                static_cast<std::int64_t>(next_pp(s)) >> 11) * 0x1.0p-52;
            const double r2 = static_cast<double>(
                static_cast<std::int64_t>(next_pp(s)) >> 11) * 0x1.0p-52;
            const double sq = r1 * r1 + r2 * r2;
            if (sq >= 1.0 || sq == 0.0) continue;
            z = r1 * std::sqrt(-2.0 * std::log(sq) / sq);
            break;
        }
        // V ~ Gamma(shape, 1) for denominator; same stream s
        double gamma_v;
        for (;;) {
            double x;
            for (;;) {
                const double r1 = static_cast<double>(
                    static_cast<std::int64_t>(next_pp(s)) >> 11) * 0x1.0p-52;
                const double r2 = static_cast<double>(
                    static_cast<std::int64_t>(next_pp(s)) >> 11) * 0x1.0p-52;
                const double sq = r1 * r1 + r2 * r2;
                if (sq >= 1.0 || sq == 0.0) continue;
                x = r1 * std::sqrt(-2.0 * std::log(sq) / sq);
                break;
            }
            const double vr = 1.0 + c * x;
            if (vr <= 0.0) continue;
            const double v  = vr * vr * vr;
            const double u  = static_cast<double>(next_pp(s) >> 11) * 0x1.0p-53;
            const double x2 = x * x;
            if (u < 1.0 - 0.0331 * x2 * x2) { gamma_v = d * v; break; }
            if (std::log(u) < 0.5 * x2 + d * (1.0 - v + std::log(v))) { gamma_v = d * v; break; }
        }
        out[i++] = z / std::sqrt(2.0 * gamma_v / nu);
    }
}

// ── x8 AVX2 fused ────────────────────────────────────────────────────────────
// Streams a+b (x8 AVX2) produce N(0,1) numerators via vecpolar into buf_z.
// Scalar stream sc generates the Gamma denominator for each z immediately.
// Both streams advance in the same outer loop; buf_z (64 doubles) stays in L1.
void fill_student_t_x8_avx2_fused(std::uint64_t seed, double nu, double* out,
                                   std::size_t count) noexcept {
#ifdef __AVX2__
    alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
    alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
    seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
    __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
    __m256i a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
    __m256i a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
    __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
    __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
    __m256i b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
    __m256i b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));

    std::uint64_t sc[4];
    {
        std::uint64_t cseed = seed;
        sc[0] = splitmix64(cseed); sc[1] = splitmix64(cseed);
        sc[2] = splitmix64(cseed); sc[3] = splitmix64(cseed);
        for (int j = 0; j < 8; ++j) long_jump(sc);
    }

    const double shape = nu / 2.0;
    const double d = shape - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);

    static constexpr int kBuf = 64;
    alignas(32) double buf_z[kBuf];

    const __m256d one  = _mm256_set1_pd(1.0);
    const __m256d zero = _mm256_setzero_pd();
    const __m256d neg2 = _mm256_set1_pd(-2.0);
    const __m256d safe = _mm256_set1_pd(0.5);

    std::size_t i = 0;
    while (i < count) {
        // ── Phase 1: fill buf_z with kBuf N(0,1) numerators ──────────────────
        int n = 0;
        while (n < kBuf) {
            const __m256d u1a = u64_to_pm1_52_avx2(next_x4_avx2(a0, a1, a2, a3));
            const __m256d u2a = u64_to_pm1_52_avx2(next_x4_avx2(a0, a1, a2, a3));
            const __m256d sq_a = _mm256_add_pd(_mm256_mul_pd(u1a, u1a),
                                               _mm256_mul_pd(u2a, u2a));
            const __m256d acc_a = _mm256_and_pd(_mm256_cmp_pd(sq_a, one, _CMP_LT_OQ),
                                                _mm256_cmp_pd(sq_a, zero, _CMP_GT_OQ));
            const int bits_a = _mm256_movemask_pd(acc_a);
            if (bits_a) {
                const __m256d sf = _mm256_blendv_pd(safe, sq_a, acc_a);
                const __m256d fa = _mm256_sqrt_pd(
                    _mm256_div_pd(_mm256_mul_pd(neg2, _ZGVdN4v_log(sf)), sf));
                alignas(32) double za[4], za2[4];
                _mm256_store_pd(za,  _mm256_mul_pd(u1a, fa));
                _mm256_store_pd(za2, _mm256_mul_pd(u2a, fa));
                for (int lane = 0; lane < 4 && n < kBuf; ++lane) {
                    if (bits_a & (1 << lane)) {
                        buf_z[n++] = za[lane];
                        if (n < kBuf) buf_z[n++] = za2[lane];
                    }
                }
            }
            const __m256d u1b = u64_to_pm1_52_avx2(next_x4_avx2(b0, b1, b2, b3));
            const __m256d u2b = u64_to_pm1_52_avx2(next_x4_avx2(b0, b1, b2, b3));
            const __m256d sq_b = _mm256_add_pd(_mm256_mul_pd(u1b, u1b),
                                               _mm256_mul_pd(u2b, u2b));
            const __m256d acc_b = _mm256_and_pd(_mm256_cmp_pd(sq_b, one, _CMP_LT_OQ),
                                                _mm256_cmp_pd(sq_b, zero, _CMP_GT_OQ));
            const int bits_b = _mm256_movemask_pd(acc_b);
            if (bits_b) {
                const __m256d sf = _mm256_blendv_pd(safe, sq_b, acc_b);
                const __m256d fb = _mm256_sqrt_pd(
                    _mm256_div_pd(_mm256_mul_pd(neg2, _ZGVdN4v_log(sf)), sf));
                alignas(32) double zb[4], zb2[4];
                _mm256_store_pd(zb,  _mm256_mul_pd(u1b, fb));
                _mm256_store_pd(zb2, _mm256_mul_pd(u2b, fb));
                for (int lane = 0; lane < 4 && n < kBuf; ++lane) {
                    if (bits_b & (1 << lane)) {
                        buf_z[n++] = zb[lane];
                        if (n < kBuf) buf_z[n++] = zb2[lane];
                    }
                }
            }
        }
        // ── Phase 2: for each z, generate Gamma(shape) from stream sc ─────────
        for (int k = 0; k < n && i < count; ++k) {
            const double z = buf_z[k];
            double gamma_v;
            for (;;) {
                double x;
                for (;;) {
                    const double r1 = static_cast<double>(
                        static_cast<std::int64_t>(next_pp(sc)) >> 11) * 0x1.0p-52;
                    const double r2 = static_cast<double>(
                        static_cast<std::int64_t>(next_pp(sc)) >> 11) * 0x1.0p-52;
                    const double sq = r1 * r1 + r2 * r2;
                    if (sq >= 1.0 || sq == 0.0) continue;
                    x = r1 * std::sqrt(-2.0 * std::log(sq) / sq);
                    break;
                }
                const double vr = 1.0 + c * x;
                if (vr <= 0.0) continue;
                const double v  = vr * vr * vr;
                const double u  = static_cast<double>(next_pp(sc) >> 11) * 0x1.0p-53;
                const double x2 = x * x;
                if (u < 1.0 - 0.0331 * x2 * x2) { gamma_v = d * v; break; }
                if (std::log(u) < 0.5 * x2 + d * (1.0 - v + std::log(v))) { gamma_v = d * v; break; }
            }
            out[i++] = z / std::sqrt(2.0 * gamma_v / nu);
        }
    }
#else
    fill_student_t_scalar_fused(seed, nu, out, count);
#endif
}

// ── x8 AVX2 decoupled ────────────────────────────────────────────────────────
// Pre-fills a Z buffer and a Gamma buffer independently, then combines.
// Adds 2 full heap-write passes + 2 full heap-read passes relative to fused.
void fill_student_t_x8_avx2_decoupled(std::uint64_t seed, double nu, double* out,
                                       std::size_t count) noexcept {
    std::vector<double> z_buf(count), g_buf(count);
    fill_xoshiro256pp_x8_normal_vecpolar_avx2(seed, z_buf.data(), count);
    fill_gamma_x8_avx2_fused(seed ^ 0x9e3779b97f4a7c15ULL, nu / 2.0, g_buf.data(), count);
    for (std::size_t i = 0; i < count; ++i)
        out[i] = z_buf[i] / std::sqrt(2.0 * g_buf[i] / nu);
}

// ── x8 AVX2 fast ─────────────────────────────────────────────────────────────
// Uses the best available kernels for each sub-distribution independently:
// vecpolar for Z ~ N(0,1); fill_gamma_x8_avx2_full (vectorized MT phase) for
// V ~ Gamma(nu/2). The Gamma "full" kernel replaces the scalar MT loop with a
// 4-wide AVX2 uniform stream, eliminating the scalar bottleneck.
void fill_student_t_x8_avx2_fast(std::uint64_t seed, double nu, double* out,
                                  std::size_t count) noexcept {
    std::vector<double> z_buf(count), g_buf(count);
    fill_xoshiro256pp_x8_normal_vecpolar_avx2(seed, z_buf.data(), count);
    fill_gamma_x8_avx2_full(seed ^ 0x9e3779b97f4a7c15ULL, nu / 2.0, g_buf.data(), count);
    for (std::size_t i = 0; i < count; ++i)
        out[i] = z_buf[i] / std::sqrt(2.0 * g_buf[i] / nu);
}

// ─── AVX-512: uniform, 8 lanes (single __m512i group) ────────────────────────
void fill_xoshiro256pp_x8_uniform01_avx512(std::uint64_t seed, double* out,
                                            std::size_t count) noexcept {
#ifdef __AVX512F__
    alignas(64) std::uint64_t state0[8], state1[8], state2[8], state3[8];
    alignas(64) double tail[8];
    seed_x8_avx512(seed, state0, state1, state2, state3);

    __m512i s0 = _mm512_load_si512(reinterpret_cast<const __m512i*>(state0));
    __m512i s1 = _mm512_load_si512(reinterpret_cast<const __m512i*>(state1));
    __m512i s2 = _mm512_load_si512(reinterpret_cast<const __m512i*>(state2));
    __m512i s3 = _mm512_load_si512(reinterpret_cast<const __m512i*>(state3));

    std::size_t i = 0;
    while (i + 8 <= count) {
        _mm512_storeu_pd(out + i, u64_to_uniform01_52_avx512(next_x8_avx512(s0, s1, s2, s3)));
        i += 8;
    }
    if (i < count) {
        _mm512_store_pd(tail, u64_to_uniform01_52_avx512(next_x8_avx512(s0, s1, s2, s3)));
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = tail[lane];
    }
#else
    fill_xoshiro256pp_x8_uniform01_avx2(seed, out, count);
#endif
}

// ─── AVX-512: uniform, 16 lanes (two __m512i groups, 8×2) ────────────────────
// Two independent 8-lane groups advance in lockstep, producing 16 outputs per
// iteration.  The two chains share no registers, so the CPU can pipeline them.
void fill_xoshiro256pp_x16_uniform01_avx512(std::uint64_t seed, double* out,
                                             std::size_t count) noexcept {
#ifdef __AVX512F__
    alignas(64) std::uint64_t sa0[8], sa1[8], sa2[8], sa3[8];
    alignas(64) std::uint64_t sb0[8], sb1[8], sb2[8], sb3[8];
    alignas(64) double tail[8];
    seed_x16_avx512(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m512i a0 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sa0));
    __m512i a1 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sa1));
    __m512i a2 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sa2));
    __m512i a3 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sa3));
    __m512i b0 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sb0));
    __m512i b1 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sb1));
    __m512i b2 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sb2));
    __m512i b3 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sb3));

    std::size_t i = 0;
    while (i + 16 <= count) {
        _mm512_storeu_pd(out + i,     u64_to_uniform01_52_avx512(next_x8_avx512(a0, a1, a2, a3)));
        _mm512_storeu_pd(out + i + 8, u64_to_uniform01_52_avx512(next_x8_avx512(b0, b1, b2, b3)));
        i += 16;
    }
    // Partial last iteration: remaining < 16.
    // Even group (positions i..i+7): chain a.  Odd group (i+8..count-1): chain b.
    if (i < count) {
        if (i + 8 <= count) {
            _mm512_storeu_pd(out + i, u64_to_uniform01_52_avx512(next_x8_avx512(a0, a1, a2, a3)));
            i += 8;
        } else {
            _mm512_store_pd(tail, u64_to_uniform01_52_avx512(next_x8_avx512(a0, a1, a2, a3)));
            for (std::size_t lane = 0; i < count; ++lane, ++i)
                out[i] = tail[lane];
        }
        if (i < count) {
            _mm512_store_pd(tail, u64_to_uniform01_52_avx512(next_x8_avx512(b0, b1, b2, b3)));
            for (std::size_t lane = 0; i < count; ++lane, ++i)
                out[i] = tail[lane];
        }
    }
#else
    fill_xoshiro256pp_x8_uniform01_avx2(seed, out, count);
#endif
}

// ─── AVX-512: normal, vecpolar with mask_compressstoreu ──────────────────────
// Two 8-lane groups.  Rejection test returns an 8-bit predicate mask (__mmask8)
// that drives _mm512_mask_compressstoreu_pd — no 128-bit extraction loop needed.
// Accepted pairs are packed into a small buffer then written to output.
//
// AMD Zen 4 implements 512-bit sqrt/div/compress as pairs of 256-bit micro-ops
// serialised on a single divider, making this kernel ~1.6× slower than the AVX2
// vecpolar on AMD hardware.  Fall back to AVX2 when running on AMD.
void fill_xoshiro256pp_x16_normal_vecpolar_avx512(std::uint64_t seed, double* out,
                                                   std::size_t count) noexcept {
#ifdef __AVX512F__
    if (cpu_is_amd()) {
        fill_xoshiro256pp_x8_normal_vecpolar_avx2(seed, out, count);
        return;
    }
    alignas(64) std::uint64_t sa0[8], sa1[8], sa2[8], sa3[8];
    alignas(64) std::uint64_t sb0[8], sb1[8], sb2[8], sb3[8];
    seed_x16_avx512(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m512i a0 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sa0));
    __m512i a1 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sa1));
    __m512i a2 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sa2));
    __m512i a3 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sa3));
    __m512i b0 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sb0));
    __m512i b1 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sb1));
    __m512i b2 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sb2));
    __m512i b3 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sb3));

    const __m512d one     = _mm512_set1_pd(1.0);
    const __m512d zero    = _mm512_setzero_pd();
    const __m512d neg2    = _mm512_set1_pd(-2.0);
    const __m512d safe_val = _mm512_set1_pd(0.5);

    std::size_t i = 0;
    while (i < count) {
        // ── Group a (8 lanes) ─────────────────────────────────────────────────
        {
            const __m512d u1 = u64_to_pm1_52_avx512(next_x8_avx512(a0, a1, a2, a3));
            const __m512d u2 = u64_to_pm1_52_avx512(next_x8_avx512(a0, a1, a2, a3));
            const __m512d s  = _mm512_add_pd(_mm512_mul_pd(u1, u1),
                                             _mm512_mul_pd(u2, u2));
            // Native unsigned-style mask: 0 < s < 1 in one expression
            const __mmask8 accept =
                _mm512_cmp_pd_mask(s, one,  _CMP_LT_OQ) &
                _mm512_cmp_pd_mask(s, zero, _CMP_GT_OQ);
            if (accept) {
                // Blend safe value for rejected lanes before log (no NaN/inf)
                const __m512d safe_s = _mm512_mask_blend_pd(accept, safe_val, s);
                const __m512d log_s  = _ZGVeN8v_log(safe_s);
                const __m512d factor = _mm512_sqrt_pd(
                    _mm512_div_pd(_mm512_mul_pd(neg2, log_s), safe_s));
                const __m512d n1 = _mm512_mul_pd(u1, factor);
                const __m512d n2 = _mm512_mul_pd(u2, factor);

                // Compress accepted lanes to a contiguous buffer, then write pairs
                alignas(64) double tmp1[8], tmp2[8];
                const int naccepted = __builtin_popcount(accept);
                _mm512_mask_compressstoreu_pd(tmp1, accept, n1);
                _mm512_mask_compressstoreu_pd(tmp2, accept, n2);
                for (int k = 0; k < naccepted && i < count; ++k) {
                    out[i++] = tmp1[k];
                    if (i < count) out[i++] = tmp2[k];
                }
            }
        }

        if (i >= count) break;

        // ── Group b (8 lanes) ─────────────────────────────────────────────────
        {
            const __m512d u1 = u64_to_pm1_52_avx512(next_x8_avx512(b0, b1, b2, b3));
            const __m512d u2 = u64_to_pm1_52_avx512(next_x8_avx512(b0, b1, b2, b3));
            const __m512d s  = _mm512_add_pd(_mm512_mul_pd(u1, u1),
                                             _mm512_mul_pd(u2, u2));
            const __mmask8 accept =
                _mm512_cmp_pd_mask(s, one,  _CMP_LT_OQ) &
                _mm512_cmp_pd_mask(s, zero, _CMP_GT_OQ);
            if (accept) {
                const __m512d safe_s = _mm512_mask_blend_pd(accept, safe_val, s);
                const __m512d log_s  = _ZGVeN8v_log(safe_s);
                const __m512d factor = _mm512_sqrt_pd(
                    _mm512_div_pd(_mm512_mul_pd(neg2, log_s), safe_s));
                const __m512d n1 = _mm512_mul_pd(u1, factor);
                const __m512d n2 = _mm512_mul_pd(u2, factor);

                alignas(64) double tmp1[8], tmp2[8];
                const int naccepted = __builtin_popcount(accept);
                _mm512_mask_compressstoreu_pd(tmp1, accept, n1);
                _mm512_mask_compressstoreu_pd(tmp2, accept, n2);
                for (int k = 0; k < naccepted && i < count; ++k) {
                    out[i++] = tmp1[k];
                    if (i < count) out[i++] = tmp2[k];
                }
            }
        }
    }
#else
    fill_xoshiro256pp_x8_normal_vecpolar_avx2(seed, out, count);
#endif
}

// ─── AVX-512: exponential, 8-wide log via libmvec _ZGVeN8v_log ───────────────
// Two 8-lane groups drive _ZGVeN8v_log, producing 16 exponential samples per
// iteration vs the AVX2 variant's 8.
void fill_xoshiro256pp_x16_exponential_avx512(std::uint64_t seed, double* out,
                                               std::size_t count) noexcept {
#ifdef __AVX512F__
    alignas(64) std::uint64_t sa0[8], sa1[8], sa2[8], sa3[8];
    alignas(64) std::uint64_t sb0[8], sb1[8], sb2[8], sb3[8];
    alignas(64) double tail[8];
    seed_x16_avx512(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m512i a0 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sa0));
    __m512i a1 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sa1));
    __m512i a2 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sa2));
    __m512i a3 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sa3));
    __m512i b0 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sb0));
    __m512i b1 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sb1));
    __m512i b2 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sb2));
    __m512i b3 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sb3));

    const __m512d neg1 = _mm512_set1_pd(-1.0);

    std::size_t i = 0;
    while (i + 16 <= count) {
        const __m512d ua = u64_to_uniform01_52_avx512(next_x8_avx512(a0, a1, a2, a3));
        const __m512d ub = u64_to_uniform01_52_avx512(next_x8_avx512(b0, b1, b2, b3));
        _mm512_storeu_pd(out + i,      _mm512_mul_pd(neg1, _ZGVeN8v_log(ua)));
        _mm512_storeu_pd(out + i + 8,  _mm512_mul_pd(neg1, _ZGVeN8v_log(ub)));
        i += 16;
    }
    if (i < count) {
        const __m512d va = _mm512_mul_pd(neg1, _ZGVeN8v_log(
            u64_to_uniform01_52_avx512(next_x8_avx512(a0, a1, a2, a3))));
        if (i + 8 <= count) {
            _mm512_storeu_pd(out + i, va);
            i += 8;
        } else {
            _mm512_store_pd(tail, va);
            for (std::size_t lane = 0; i < count; ++lane, ++i)
                out[i] = tail[lane];
        }
        if (i < count) {
            const __m512d vb = _mm512_mul_pd(neg1, _ZGVeN8v_log(
                u64_to_uniform01_52_avx512(next_x8_avx512(b0, b1, b2, b3))));
            _mm512_store_pd(tail, vb);
            for (std::size_t lane = 0; i < count; ++lane, ++i)
                out[i] = tail[lane];
        }
    }
#else
    fill_xoshiro256pp_x8_exponential_avx2(seed, out, count);
#endif
}

// ─── AVX-512: Bernoulli with native unsigned 64-bit comparison ───────────────
// AVX-512F provides _mm512_cmp_epu64_mask for true unsigned compare, eliminating
// the sign-flip workaround required by AVX2's signed-only _mm256_cmpgt_epi64.
// The result is a compact __mmask8 used directly with _mm512_maskz_mov_pd.
void fill_xoshiro256pp_x16_bernoulli_avx512(std::uint64_t seed, double p,
                                             double* out,
                                             std::size_t count) noexcept {
#ifdef __AVX512F__
    alignas(64) std::uint64_t sa0[8], sa1[8], sa2[8], sa3[8];
    alignas(64) std::uint64_t sb0[8], sb1[8], sb2[8], sb3[8];
    seed_x16_avx512(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m512i a0 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sa0));
    __m512i a1 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sa1));
    __m512i a2 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sa2));
    __m512i a3 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sa3));
    __m512i b0 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sb0));
    __m512i b1 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sb1));
    __m512i b2 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sb2));
    __m512i b3 = _mm512_load_si512(reinterpret_cast<const __m512i*>(sb3));

    const auto threshold = static_cast<std::uint64_t>(p * 0x1.0p64);
    const __m512i thresh_vec = _mm512_set1_epi64(
        static_cast<std::int64_t>(threshold));
    const __m512d one_d = _mm512_set1_pd(1.0);

    std::size_t i = 0;
    while (i + 16 <= count) {
        const __m512i ra = next_x8_avx512(a0, a1, a2, a3);
        const __m512i rb = next_x8_avx512(b0, b1, b2, b3);
        // Native unsigned compare: mask bit set ↔ ra[lane] < threshold
        const __mmask8 mask_a = _mm512_cmp_epu64_mask(ra, thresh_vec, _MM_CMPINT_LT);
        const __mmask8 mask_b = _mm512_cmp_epu64_mask(rb, thresh_vec, _MM_CMPINT_LT);
        // 1.0 where mask bit is set, 0.0 elsewhere
        _mm512_storeu_pd(out + i,      _mm512_maskz_mov_pd(mask_a, one_d));
        _mm512_storeu_pd(out + i + 8,  _mm512_maskz_mov_pd(mask_b, one_d));
        i += 16;
    }
    if (i < count) {
        alignas(64) double tmp[8];
        const __mmask8 mask_a = _mm512_cmp_epu64_mask(
            next_x8_avx512(a0, a1, a2, a3), thresh_vec, _MM_CMPINT_LT);
        if (i + 8 <= count) {
            _mm512_storeu_pd(out + i, _mm512_maskz_mov_pd(mask_a, one_d));
            i += 8;
        } else {
            _mm512_store_pd(tmp, _mm512_maskz_mov_pd(mask_a, one_d));
            for (std::size_t lane = 0; i < count; ++lane, ++i)
                out[i] = tmp[lane];
        }
        if (i < count) {
            const __mmask8 mask_b = _mm512_cmp_epu64_mask(
                next_x8_avx512(b0, b1, b2, b3), thresh_vec, _MM_CMPINT_LT);
            _mm512_store_pd(tmp, _mm512_maskz_mov_pd(mask_b, one_d));
            for (std::size_t lane = 0; i < count; ++lane, ++i)
                out[i] = tmp[lane];
        }
    }
#else
    fill_xoshiro256pp_x8_bernoulli_fast(seed, p, out, count);
#endif
}

}  // namespace zorro_bench
