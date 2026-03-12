#include "benchmarks/zorro_benchmark_kernels.hpp"

#include <cmath>
#ifdef __AVX2__
#include <immintrin.h>
// glibc libmvec: AVX2 4-wide double log
extern "C" __m256d _ZGVdN4v_log(__m256d) noexcept;
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

#ifdef __AVX512F__

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
    while (i + 8 <= count) {
        _mm512_storeu_pd(out + i, u64_to_uniform01_52_avx512(next_x8_avx512(a0, a1, a2, a3)));
        i += 8;
    }
    if (i < count) {
        _mm512_store_pd(tail, u64_to_uniform01_52_avx512(next_x8_avx512(a0, a1, a2, a3)));
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = tail[lane];
    }
#else
    fill_xoshiro256pp_x16_uniform01_avx512(seed, out, count);
#endif
}

// ─── AVX-512: normal, vecpolar with mask_compressstoreu ──────────────────────
// Two 8-lane groups.  Rejection test returns an 8-bit predicate mask (__mmask8)
// that drives _mm512_mask_compressstoreu_pd — no 128-bit extraction loop needed.
// Accepted pairs are packed into a small buffer then written to output.
void fill_xoshiro256pp_x16_normal_vecpolar_avx512(std::uint64_t seed, double* out,
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
    while (i + 8 <= count) {
        const __m512d ua = u64_to_uniform01_52_avx512(next_x8_avx512(a0, a1, a2, a3));
        _mm512_storeu_pd(out + i, _mm512_mul_pd(neg1, _ZGVeN8v_log(ua)));
        i += 8;
    }
    if (i < count) {
        const __m512d ua = u64_to_uniform01_52_avx512(next_x8_avx512(a0, a1, a2, a3));
        _mm512_store_pd(tail, _mm512_mul_pd(neg1, _ZGVeN8v_log(ua)));
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = tail[lane];
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
    while (i + 8 <= count) {
        const __m512i ra = next_x8_avx512(a0, a1, a2, a3);
        const __mmask8 mask_a = _mm512_cmp_epu64_mask(ra, thresh_vec, _MM_CMPINT_LT);
        _mm512_storeu_pd(out + i, _mm512_maskz_mov_pd(mask_a, one_d));
        i += 8;
    }
    if (i < count) {
        alignas(64) double tmp[8];
        const __m512i ra = next_x8_avx512(a0, a1, a2, a3);
        const __mmask8 mask_a = _mm512_cmp_epu64_mask(ra, thresh_vec, _MM_CMPINT_LT);
        _mm512_store_pd(tmp, _mm512_maskz_mov_pd(mask_a, one_d));
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = tmp[lane];
    }
#else
    fill_xoshiro256pp_x8_bernoulli_fast(seed, p, out, count);
#endif
}

}  // namespace zorro_bench
