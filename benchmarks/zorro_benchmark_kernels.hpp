#pragma once

// Benchmark-only generator and transform kernels used by the distribution
// harness. These are not part of the stable public Zorro API.

#include <cstddef>
#include <cstdint>

namespace zorro_bench {

void generate_xoshiro256pp_x4_bits(std::uint64_t seed, std::uint64_t* out,
                                   std::size_t count) noexcept;
void fill_xoshiro256pp_x4_uniform01(std::uint64_t seed, double* out,
                                    std::size_t count) noexcept;
void fill_xoshiro256pp_x4_uniform01_avx2(std::uint64_t seed, double* out,
                                         std::size_t count) noexcept;
void fill_xoshiro256p_x4_uniform01_avx2(std::uint64_t seed, double* out,
                                        std::size_t count) noexcept;
void fill_xoshiro256pp_x4_normal_polar_avx2(std::uint64_t seed, double* out,
                                            std::size_t count) noexcept;
void fill_xoshiro256p_x4_normal_polar_avx2(std::uint64_t seed, double* out,
                                           std::size_t count) noexcept;
void fill_xoshiro256pp_x4_uniform01_avx2_unroll2(std::uint64_t seed, double* out,
                                                  std::size_t count) noexcept;
void fill_xoshiro256pp_x8_uniform01_avx2(std::uint64_t seed, double* out,
                                          std::size_t count) noexcept;
void fill_xoshiro256pp_x8_normal_polar_avx2(std::uint64_t seed, double* out,
                                             std::size_t count) noexcept;
void fill_xoshiro256pp_x8_normal_polar_avx2_batched(std::uint64_t seed, double* out,
                                                     std::size_t count) noexcept;
void fill_xoshiro256pp_x4_uniform01_fused(std::uint64_t seed, double* out,
                                           std::size_t count) noexcept;
void fill_xoshiro256pp_x8_normal_polar_avx2_veclog(std::uint64_t seed, double* out,
                                                     std::size_t count) noexcept;
void fill_xoshiro256pp_x8_normal_vecpolar_avx2(std::uint64_t seed, double* out,
                                                 std::size_t count) noexcept;

// ── Exponential distribution ────────────────────────────────────────────────
// Naive: generate uniform doubles, then -log(u) scalar.
void fill_xoshiro256pp_x8_exponential_naive(std::uint64_t seed, double* out,
                                             std::size_t count) noexcept;
// Fast: generate raw bits, convert to uniform in SIMD, vectorized -log via libmvec.
void fill_xoshiro256pp_x8_exponential_avx2(std::uint64_t seed, double* out,
                                            std::size_t count) noexcept;

// ── Bernoulli distribution ──────────────────────────────────────────────────
// Naive: generate uniform doubles, compare against p.
void fill_xoshiro256pp_x8_bernoulli_naive(std::uint64_t seed, double p,
                                           double* out,
                                           std::size_t count) noexcept;
// Fast: compare raw uint64 against precomputed threshold, no float conversion.
void fill_xoshiro256pp_x8_bernoulli_fast(std::uint64_t seed, double p,
                                          double* out,
                                          std::size_t count) noexcept;

}  // namespace zorro_bench
