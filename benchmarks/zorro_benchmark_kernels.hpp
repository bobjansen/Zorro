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
void fill_xoshiro256pp_x8_normal_box_muller_avx2(std::uint64_t seed, double* out,
                                                  std::size_t count) noexcept;
void fill_xoshiro256pp_x8_normal_box_muller_avx2_fastlog(std::uint64_t seed, double* out,
                                                          std::size_t count) noexcept;
void fill_xoshiro256pp_x8_normal_box_muller_avx2_approxsincos(std::uint64_t seed, double* out,
                                                               std::size_t count) noexcept;
void fill_xoshiro256pp_x8_normal_box_muller_avx2_fullapprox(std::uint64_t seed, double* out,
                                                             std::size_t count) noexcept;
void fill_xoshiro256pp_x8_normal_vecpolar_avx2(std::uint64_t seed, double* out,
                                                 std::size_t count) noexcept;
void fill_xoshiro256pp_x8_exponential_naive(std::uint64_t seed, double* out,
                                             std::size_t count) noexcept;
void fill_xoshiro256pp_x8_exponential_avx2(std::uint64_t seed, double* out,
                                            std::size_t count) noexcept;
void fill_xoshiro256pp_x8_exponential_avx2_fastlog(std::uint64_t seed, double* out,
                                                    std::size_t count) noexcept;
void fill_xoshiro256pp_x8_bernoulli_naive(std::uint64_t seed, double p,
                                           double* out,
                                           std::size_t count) noexcept;
void fill_xoshiro256pp_x8_bernoulli_fast(std::uint64_t seed, double p,
                                          double* out,
                                          std::size_t count) noexcept;
// Bernoulli(0.5) special case: each raw bit is an independent trial.
// One uint64 → 64 samples. 16x throughput vs per-lane threshold compare.
void fill_xoshiro256pp_x8_bernoulli_half(std::uint64_t seed, double* out,
                                          std::size_t count) noexcept;

// ── Bernoulli → uint8_t output (0/1 per byte) ──────────────────────────────
// 8x less memory traffic than double output.
void fill_xoshiro256pp_x8_bernoulli_u8_naive(std::uint64_t seed, double p,
                                              std::uint8_t* out,
                                              std::size_t count) noexcept;
void fill_xoshiro256pp_x8_bernoulli_u8_fast(std::uint64_t seed, double p,
                                             std::uint8_t* out,
                                             std::size_t count) noexcept;
// p=0.5 bit-unpack to uint8_t: unpack each bit of raw uint64 into a byte.
void fill_xoshiro256pp_x8_bernoulli_u8_half(std::uint64_t seed,
                                             std::uint8_t* out,
                                             std::size_t count) noexcept;
// p=0.5 bit-unpack, skip 16 bits (48 usable = exact multiple of 8, no tail).
void fill_xoshiro256pp_x8_bernoulli_u8_half_skip16(std::uint64_t seed,
                                                    std::uint8_t* out,
                                                    std::size_t count) noexcept;

// ── Bernoulli(0.5) → packed bitmask (1 bit per sample) ──────────────────────
// Each uint64_t contains 52 independent trials in bits [0..51].
// Bits [52..63] are zero (discarded weak low bits after >> 12).
// count is in *samples* (bits), not words. Output buffer must hold
// at least ceil(count / 52) * 8 bytes (since each call produces
// 4 × 52 = 208 bits = 4 words, buffer needs ceil(count/208)*32 bytes).
void fill_xoshiro256pp_x8_bernoulli_bits(std::uint64_t seed,
                                          std::uint64_t* out,
                                          std::size_t count) noexcept;

// Gamma(alpha, 1) — Marsaglia-Tsang
void fill_gamma_scalar_fused(std::uint64_t seed, double alpha, double* out,
                              std::size_t count) noexcept;
void fill_gamma_x8_avx2_fused(std::uint64_t seed, double alpha, double* out,
                               std::size_t count) noexcept;
void fill_gamma_x8_avx2_decoupled(std::uint64_t seed, double alpha, double* out,
                                   std::size_t count) noexcept;
void fill_gamma_x8_avx2_full(std::uint64_t seed, double alpha, double* out,
                              std::size_t count) noexcept;

// Student's t(nu) — t = N(0,1) / sqrt(2·Gamma(nu/2)/nu)
void fill_student_t_scalar_fused(std::uint64_t seed, double nu, double* out,
                                  std::size_t count) noexcept;
void fill_student_t_x8_avx2_fused(std::uint64_t seed, double nu, double* out,
                                   std::size_t count) noexcept;
void fill_student_t_x8_avx2_decoupled(std::uint64_t seed, double nu, double* out,
                                       std::size_t count) noexcept;
void fill_student_t_x8_avx2_fast(std::uint64_t seed, double nu, double* out,
                                  std::size_t count) noexcept;

// AVX-512 kernels
void fill_xoshiro256pp_x8_uniform01_avx512(std::uint64_t seed, double* out,
                                            std::size_t count) noexcept;
void fill_xoshiro256pp_x16_uniform01_avx512(std::uint64_t seed, double* out,
                                             std::size_t count) noexcept;
void fill_xoshiro256pp_x16_normal_box_muller_avx512_fullapprox(std::uint64_t seed, double* out,
                                                                std::size_t count) noexcept;
void fill_xoshiro256pp_x16_normal_vecpolar_avx512(std::uint64_t seed, double* out,
                                                   std::size_t count) noexcept;
void fill_xoshiro256pp_x16_exponential_avx512(std::uint64_t seed, double* out,
                                               std::size_t count) noexcept;
void fill_xoshiro256pp_x16_bernoulli_avx512(std::uint64_t seed, double p,
                                             double* out,
                                             std::size_t count) noexcept;

}  // namespace zorro_bench
