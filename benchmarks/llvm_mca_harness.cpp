#include <bit>
#include <cstddef>
#include <cstdint>

#include "zorro/zorro.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// This file is not a benchmark and not a unit test.  It is a small collection
// of hand-picked kernels meant to be compiled to assembly and analyzed with
// LLVM-MCA.
//
// Design goals:
// - keep each region small enough that the generated assembly is easy to read
// - expose the important comparison points for this repo's RNG core
// - return a value so the compiler cannot delete the loop body
// - keep setup outside the hot loop so MCA focuses on steady-state behavior
//
// The harness uses `extern "C"` plus `noinline` so the symbols have stable,
// human-readable names and show up as isolated functions in the assembly.

namespace {

// Any non-pathological nonzero states are fine here because LLVM-MCA models the
// instruction stream, not the statistical quality of the sequence. The purpose
// is to keep the hot loops small and value-independent.
constexpr std::uint64_t kScalarS0 = 0x417cb9a826d831dfULL;
constexpr std::uint64_t kScalarS1 = 0x161922c645ce50e8ULL;
constexpr std::uint64_t kScalarS2 = 0x3501ff44902ca50dULL;
constexpr std::uint64_t kScalarS3 = 0xad760cafa1697b60ULL;

constexpr std::uint64_t kA0[4] = {
    0x417cb9a826d831dfULL, 0x3501ff44902ca50dULL,
    0xad760cafa1697b60ULL, 0x161922c645ce50e8ULL,
};
constexpr std::uint64_t kA1[4] = {
    0x6c3c7d6f1f2a4b59ULL, 0x93e1a7b41d2c8f05ULL,
    0x4fa9ce27b6135d8aULL, 0xd2b74407c91e3fa1ULL,
};
constexpr std::uint64_t kA2[4] = {
    0xb5d7c13a6e2f9041ULL, 0x28f16ab4cd907e35ULL,
    0xe17c45f920ab638dULL, 0x5a0dd38ec471b29fULL,
};
constexpr std::uint64_t kA3[4] = {
    0x9f13cb8475ae201dULL, 0x3c62f0a179de54b8ULL,
    0xc8ab6123ef9047d2ULL, 0x74d1950b2ec36af4ULL,
};

constexpr std::uint64_t kB0[4] = {
    0x2c17d9e45f80ab31ULL, 0xa4f06bc219de3578ULL,
    0x58b3941dea670fc2ULL, 0xe10f52a73bc4896dULL,
};
constexpr std::uint64_t kB1[4] = {
    0x7d245b90ce31a864ULL, 0xcf918a52d40e7b13ULL,
    0x14be63dca92f058eULL, 0xb83ad7516f0c42e9ULL,
};
constexpr std::uint64_t kB2[4] = {
    0x49c370de1285af76ULL, 0xf2ad18c46b9035e1ULL,
    0x06e954abfd217c38ULL, 0x9b7f2dc18564e0a5ULL,
};
constexpr std::uint64_t kB3[4] = {
    0xd6ac41f75b90238cULL, 0x31f089c2de764ab7ULL,
    0x8efc257134db9012ULL, 0x57a1d8ce20f64b69ULL,
};

constexpr std::uint64_t kC0[8] = {
    0x417cb9a826d831dfULL, 0x3501ff44902ca50dULL,
    0xad760cafa1697b60ULL, 0x161922c645ce50e8ULL,
    0x2c17d9e45f80ab31ULL, 0xa4f06bc219de3578ULL,
    0x58b3941dea670fc2ULL, 0xe10f52a73bc4896dULL,
};
constexpr std::uint64_t kC1[8] = {
    0x6c3c7d6f1f2a4b59ULL, 0x93e1a7b41d2c8f05ULL,
    0x4fa9ce27b6135d8aULL, 0xd2b74407c91e3fa1ULL,
    0x7d245b90ce31a864ULL, 0xcf918a52d40e7b13ULL,
    0x14be63dca92f058eULL, 0xb83ad7516f0c42e9ULL,
};
constexpr std::uint64_t kC2[8] = {
    0xb5d7c13a6e2f9041ULL, 0x28f16ab4cd907e35ULL,
    0xe17c45f920ab638dULL, 0x5a0dd38ec471b29fULL,
    0x49c370de1285af76ULL, 0xf2ad18c46b9035e1ULL,
    0x06e954abfd217c38ULL, 0x9b7f2dc18564e0a5ULL,
};
constexpr std::uint64_t kC3[8] = {
    0x9f13cb8475ae201dULL, 0x3c62f0a179de54b8ULL,
    0xc8ab6123ef9047d2ULL, 0x74d1950b2ec36af4ULL,
    0xd6ac41f75b90238cULL, 0x31f089c2de764ab7ULL,
    0x8efc257134db9012ULL, 0x57a1d8ce20f64b69ULL,
};

constexpr std::uint64_t kD0[8] = {
    0x93f1ab26d4805c17ULL, 0x4eb809d2a51f37ccULL,
    0xc1275d8af06b349eULL, 0x2874e1bc9d53fa60ULL,
    0x6fa23158cde98472ULL, 0xba7054d9132ec681ULL,
    0x15dfe8a4b76920ccULL, 0xe38421cf507ab69dULL,
};
constexpr std::uint64_t kD1[8] = {
    0x5c18a72d349ef1b0ULL, 0xd28f04be713ca96dULL,
    0x3ab561ec820fd497ULL, 0x84df2970c13b5ea2ULL,
    0xaf3d70c128e94b56ULL, 0x11c7e8ad64b2903fULL,
    0xc67a15439df80e21ULL, 0x72b0dc8e4af5319bULL,
};
constexpr std::uint64_t kD2[8] = {
    0xe048c39fd7125ab6ULL, 0x39bd6512ac8ef470ULL,
    0x7ac103e65fb9248dULL, 0x12e57cb084d36af1ULL,
    0x9db40f71ce28a563ULL, 0x44f68923b17dc0aeULL,
    0xb18cde54f0327b19ULL, 0x2605a73fd9eb468cULL,
};
constexpr std::uint64_t kD3[8] = {
    0x31fa5ce49076b82dULL, 0xc8b4172de3a05f61ULL,
    0x0fe1937cb4628ad4ULL, 0x9670ad12f85ec349ULL,
    0x58ce2417db90f6a3ULL, 0xa274f95bc1e83d70ULL,
    0x6de18ab4305fc927ULL, 0xf34b20c79a16458eULL,
};

inline auto rotl64(std::uint64_t x, int k) noexcept -> std::uint64_t {
  return std::rotl(x, k);
}

#if defined(__AVX2__)
inline auto sum_lanes(__m256d value) noexcept -> double {
  alignas(32) double lanes[4];
  _mm256_store_pd(lanes, value);
  return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}
#endif

#if defined(__AVX512F__)
inline auto sum_lanes(__m512d value) noexcept -> double {
  alignas(64) double lanes[8];
  _mm512_store_pd(lanes, value);
  return lanes[0] + lanes[1] + lanes[2] + lanes[3] +
         lanes[4] + lanes[5] + lanes[6] + lanes[7];
}
#endif

} // namespace

extern "C" __attribute__((noinline)) auto
zorro_mca_scalar_step(std::size_t iterations) -> std::uint64_t {
  // Scalar recurrence identical to zorro::Xoshiro256pp::operator(), but with
  // the state already scalarized into local registers so LLVM-MCA sees the hot
  // dependency chain rather than object setup.
  std::uint64_t s0 = kScalarS0;
  std::uint64_t s1 = kScalarS1;
  std::uint64_t s2 = kScalarS2;
  std::uint64_t s3 = kScalarS3;
  std::uint64_t acc = 0;
  for (std::size_t i = 0; i < iterations; ++i) {
    const std::uint64_t result = rotl64(s0 + s3, 23) + s0;
    const std::uint64_t t = s1 << 17;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = rotl64(s3, 45);
    acc ^= result;
  }
  return acc;
}

extern "C" __attribute__((noinline)) auto
zorro_mca_scalar_step_unroll2(std::size_t iterations) -> std::uint64_t {
  // Same recurrence, but two steps per loop iteration. This helps answer "is
  // the loop branch mattering, or are we fully limited by the recurrence?".
  std::uint64_t s0 = kScalarS0;
  std::uint64_t s1 = kScalarS1;
  std::uint64_t s2 = kScalarS2;
  std::uint64_t s3 = kScalarS3;
  std::uint64_t acc = 0;
  std::size_t i = 0;
  for (; i + 1 < iterations; i += 2) {
    std::uint64_t result = rotl64(s0 + s3, 23) + s0;
    std::uint64_t t = s1 << 17;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = rotl64(s3, 45);
    acc ^= result;

    result = rotl64(s0 + s3, 23) + s0;
    t = s1 << 17;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = rotl64(s3, 45);
    acc ^= result;
  }
  if (i < iterations) {
    const std::uint64_t result = rotl64(s0 + s3, 23) + s0;
    const std::uint64_t t = s1 << 17;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = rotl64(s3, 45);
    acc ^= result;
  }
  return acc;
}

#if defined(__AVX2__)

extern "C" __attribute__((noinline)) auto
zorro_mca_avx2_next_x4(std::size_t iterations) -> std::uint64_t {
  // Raw AVX2 xoshiro256++ step on one 4-lane state group. This isolates the
  // SIMD recurrence itself without float conversion or memory stores.
  const __m256i v_init0 =
      _mm256_set_epi64x(kA0[3], kA0[2], kA0[1], kA0[0]);
  const __m256i v_init1 =
      _mm256_set_epi64x(kA1[3], kA1[2], kA1[1], kA1[0]);
  const __m256i v_init2 =
      _mm256_set_epi64x(kA2[3], kA2[2], kA2[1], kA2[0]);
  const __m256i v_init3 =
      _mm256_set_epi64x(kA3[3], kA3[2], kA3[1], kA3[0]);
  __m256i v0 = v_init0;
  __m256i v1 = v_init1;
  __m256i v2 = v_init2;
  __m256i v3 = v_init3;
  __m256i acc = _mm256_setzero_si256();

  for (std::size_t i = 0; i < iterations; ++i)
    acc = _mm256_xor_si256(acc, zorro::detail::next_x4_avx2(v0, v1, v2, v3));

  alignas(32) std::uint64_t out[4];
  _mm256_store_si256(reinterpret_cast<__m256i *>(out), acc);
  return out[0] ^ out[1] ^ out[2] ^ out[3];
}

extern "C" __attribute__((noinline)) auto
zorro_mca_avx2_next_x4_uniform01(std::size_t iterations) -> double {
  // Raw AVX2 recurrence plus the integer-to-double conversion used by uniform
  // generation. This tells you whether the throughput bottleneck is still the
  // RNG core or has shifted to the floating-point conversion path.
  //
  // Use two independent accumulators so LLVM-MCA sees the recurrence and
  // conversion work, not a single loop-carried floating-point reduction chain
  // that the real fill kernels do not have.
  const __m256i v_init0 =
      _mm256_set_epi64x(kA0[3], kA0[2], kA0[1], kA0[0]);
  const __m256i v_init1 =
      _mm256_set_epi64x(kA1[3], kA1[2], kA1[1], kA1[0]);
  const __m256i v_init2 =
      _mm256_set_epi64x(kA2[3], kA2[2], kA2[1], kA2[0]);
  const __m256i v_init3 =
      _mm256_set_epi64x(kA3[3], kA3[2], kA3[1], kA3[0]);
  __m256i v0 = v_init0;
  __m256i v1 = v_init1;
  __m256i v2 = v_init2;
  __m256i v3 = v_init3;
  __m256d acc0 = _mm256_setzero_pd();
  __m256d acc1 = _mm256_setzero_pd();
  std::size_t i = 0;

  for (; i + 1 < iterations; i += 2) {
    const __m256i bits0 = zorro::detail::next_x4_avx2(v0, v1, v2, v3);
    const __m256i bits1 = zorro::detail::next_x4_avx2(v0, v1, v2, v3);
    acc0 = _mm256_add_pd(acc0, zorro::detail::u64_to_uniform01_avx2(bits0));
    acc1 = _mm256_add_pd(acc1, zorro::detail::u64_to_uniform01_avx2(bits1));
  }

  if (i < iterations) {
    const __m256i bits = zorro::detail::next_x4_avx2(v0, v1, v2, v3);
    acc0 = _mm256_add_pd(acc0, zorro::detail::u64_to_uniform01_avx2(bits));
  }

  return sum_lanes(_mm256_add_pd(acc0, acc1));
}

extern "C" __attribute__((noinline)) auto
zorro_mca_avx2_interleaved_uniform_x8(std::size_t iterations) -> double {
  // Two independent 4-lane groups interleaved in one loop. This mirrors the
  // repo's x8 AVX2 uniform-fill structure more closely than the raw x4 kernel.
  //
  // This is the most useful MCA target if you want to test the claim that a
  // single xoshiro stream is latency-bound while two independent streams are
  // enough to keep the backend closer to throughput-bound.
  __m256i va0 = _mm256_set_epi64x(kA0[3], kA0[2], kA0[1], kA0[0]);
  __m256i va1 = _mm256_set_epi64x(kA1[3], kA1[2], kA1[1], kA1[0]);
  __m256i va2 = _mm256_set_epi64x(kA2[3], kA2[2], kA2[1], kA2[0]);
  __m256i va3 = _mm256_set_epi64x(kA3[3], kA3[2], kA3[1], kA3[0]);
  __m256i vb0 = _mm256_set_epi64x(kB0[3], kB0[2], kB0[1], kB0[0]);
  __m256i vb1 = _mm256_set_epi64x(kB1[3], kB1[2], kB1[1], kB1[0]);
  __m256i vb2 = _mm256_set_epi64x(kB2[3], kB2[2], kB2[1], kB2[0]);
  __m256i vb3 = _mm256_set_epi64x(kB3[3], kB3[2], kB3[1], kB3[0]);
  // Keep one accumulator per independent stream so the only loop-carried
  // dependencies are the actual RNG recurrences.
  __m256d acc_a = _mm256_setzero_pd();
  __m256d acc_b = _mm256_setzero_pd();

  for (std::size_t i = 0; i < iterations; ++i) {
    const __m256i ra = zorro::detail::next_x4_avx2(va0, va1, va2, va3);
    const __m256i rb = zorro::detail::next_x4_avx2(vb0, vb1, vb2, vb3);
    acc_a = _mm256_add_pd(acc_a, zorro::detail::u64_to_uniform01_avx2(ra));
    acc_b = _mm256_add_pd(acc_b, zorro::detail::u64_to_uniform01_avx2(rb));
  }

  return sum_lanes(_mm256_add_pd(acc_a, acc_b));
}

#endif

#if defined(__AVX512F__)

extern "C" __attribute__((noinline)) auto
zorro_mca_avx512_next_x8(std::size_t iterations) -> std::uint64_t {
  // Raw AVX-512 xoshiro256++ step on one 8-lane state group. This is the real
  // AVX-512 analogue of the AVX2 x4 kernel: same recurrence, wider state, and
  // native vector rotates via _mm512_rol_epi64.
  __m512i v0 =
      _mm512_set_epi64(kC0[7], kC0[6], kC0[5], kC0[4],
                       kC0[3], kC0[2], kC0[1], kC0[0]);
  __m512i v1 =
      _mm512_set_epi64(kC1[7], kC1[6], kC1[5], kC1[4],
                       kC1[3], kC1[2], kC1[1], kC1[0]);
  __m512i v2 =
      _mm512_set_epi64(kC2[7], kC2[6], kC2[5], kC2[4],
                       kC2[3], kC2[2], kC2[1], kC2[0]);
  __m512i v3 =
      _mm512_set_epi64(kC3[7], kC3[6], kC3[5], kC3[4],
                       kC3[3], kC3[2], kC3[1], kC3[0]);
  __m512i acc = _mm512_setzero_si512();

  for (std::size_t i = 0; i < iterations; ++i)
    acc = _mm512_xor_si512(acc, zorro::detail::next_x8_avx512(v0, v1, v2, v3));

  alignas(64) std::uint64_t out[8];
  _mm512_store_si512(out, acc);
  return out[0] ^ out[1] ^ out[2] ^ out[3] ^
         out[4] ^ out[5] ^ out[6] ^ out[7];
}

extern "C" __attribute__((noinline)) auto
zorro_mca_avx512_next_x8_uniform01(std::size_t iterations) -> double {
  // AVX-512 recurrence plus the integer-to-double conversion used by the x8
  // AVX-512 uniform path. This isolates the "true" 8-lane AVX-512 uniform core
  // without involving stores or tail handling.
  //
  // As in the AVX2 version, keep multiple partial sums so the report is not
  // dominated by an artificial scalarized reduction dependency.
  __m512i v0 =
      _mm512_set_epi64(kC0[7], kC0[6], kC0[5], kC0[4],
                       kC0[3], kC0[2], kC0[1], kC0[0]);
  __m512i v1 =
      _mm512_set_epi64(kC1[7], kC1[6], kC1[5], kC1[4],
                       kC1[3], kC1[2], kC1[1], kC1[0]);
  __m512i v2 =
      _mm512_set_epi64(kC2[7], kC2[6], kC2[5], kC2[4],
                       kC2[3], kC2[2], kC2[1], kC2[0]);
  __m512i v3 =
      _mm512_set_epi64(kC3[7], kC3[6], kC3[5], kC3[4],
                       kC3[3], kC3[2], kC3[1], kC3[0]);
  __m512d acc0 = _mm512_setzero_pd();
  __m512d acc1 = _mm512_setzero_pd();
  std::size_t i = 0;

  for (; i + 1 < iterations; i += 2) {
    const __m512i bits0 = zorro::detail::next_x8_avx512(v0, v1, v2, v3);
    const __m512i bits1 = zorro::detail::next_x8_avx512(v0, v1, v2, v3);
    acc0 = _mm512_add_pd(acc0, zorro::detail::u64_to_uniform01_avx512(bits0));
    acc1 = _mm512_add_pd(acc1, zorro::detail::u64_to_uniform01_avx512(bits1));
  }

  if (i < iterations) {
    const __m512i bits = zorro::detail::next_x8_avx512(v0, v1, v2, v3);
    acc0 = _mm512_add_pd(acc0, zorro::detail::u64_to_uniform01_avx512(bits));
  }

  return sum_lanes(_mm512_add_pd(acc0, acc1));
}

extern "C" __attribute__((noinline)) auto
zorro_mca_avx512_interleaved_uniform_x16(std::size_t iterations) -> double {
  // Two independent 8-lane AVX-512 groups interleaved in one loop. This is the
  // AVX-512 counterpart of the x8 AVX2 kernel and mirrors the library's 2x8
  // structure for the x16 uniform path.
  __m512i va0 =
      _mm512_set_epi64(kC0[7], kC0[6], kC0[5], kC0[4],
                       kC0[3], kC0[2], kC0[1], kC0[0]);
  __m512i va1 =
      _mm512_set_epi64(kC1[7], kC1[6], kC1[5], kC1[4],
                       kC1[3], kC1[2], kC1[1], kC1[0]);
  __m512i va2 =
      _mm512_set_epi64(kC2[7], kC2[6], kC2[5], kC2[4],
                       kC2[3], kC2[2], kC2[1], kC2[0]);
  __m512i va3 =
      _mm512_set_epi64(kC3[7], kC3[6], kC3[5], kC3[4],
                       kC3[3], kC3[2], kC3[1], kC3[0]);
  __m512i vb0 =
      _mm512_set_epi64(kD0[7], kD0[6], kD0[5], kD0[4],
                       kD0[3], kD0[2], kD0[1], kD0[0]);
  __m512i vb1 =
      _mm512_set_epi64(kD1[7], kD1[6], kD1[5], kD1[4],
                       kD1[3], kD1[2], kD1[1], kD1[0]);
  __m512i vb2 =
      _mm512_set_epi64(kD2[7], kD2[6], kD2[5], kD2[4],
                       kD2[3], kD2[2], kD2[1], kD2[0]);
  __m512i vb3 =
      _mm512_set_epi64(kD3[7], kD3[6], kD3[5], kD3[4],
                       kD3[3], kD3[2], kD3[1], kD3[0]);
  __m512d acc_a = _mm512_setzero_pd();
  __m512d acc_b = _mm512_setzero_pd();

  for (std::size_t i = 0; i < iterations; ++i) {
    const __m512i ra = zorro::detail::next_x8_avx512(va0, va1, va2, va3);
    const __m512i rb = zorro::detail::next_x8_avx512(vb0, vb1, vb2, vb3);
    acc_a = _mm512_add_pd(acc_a, zorro::detail::u64_to_uniform01_avx512(ra));
    acc_b = _mm512_add_pd(acc_b, zorro::detail::u64_to_uniform01_avx512(rb));
  }

  return sum_lanes(_mm512_add_pd(acc_a, acc_b));
}

#endif
