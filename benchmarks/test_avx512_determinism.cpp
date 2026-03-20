// test_avx512_determinism.cpp
//
// Verifies that every AVX-512 kernel is *prefix-consistent*: the first N values
// produced with count=M are bit-identical to those produced with count=N, for
// every N in [1, kMaxCheck].
//
// This specifically exercises all tail paths (count % 8 == 1..7 for x8 kernels,
// count % 16 == 1..15 for x16 kernels) as well as the exact-multiple paths.
//
// A sentinel region immediately after the requested buffer is also checked to
// ensure no kernel writes beyond the requested count.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "benchmarks/zorro_benchmark_kernels.hpp"

namespace {

static constexpr std::uint64_t kSeed = 0xdeadbeef01234567ULL;

// Reference buffer large enough to cover at least two full iterations of the
// widest kernel (x16 = 16 per iteration → 32 covers 2 × 16).
static constexpr int kRefSize  = 32;
static constexpr int kMaxCheck = kRefSize - 1;  // test every count in [1, kMaxCheck]

// Number of sentinel doubles placed immediately after the requested region.
static constexpr int kSentinelCount = 16;
static constexpr double kSentinel = -3.141592653589793;  // unlikely natural output

using KernelFn = std::function<void(std::uint64_t, double*, std::size_t)>;

int run_prefix_test(const std::string& name, const KernelFn& fn) {
    // Generate the reference sequence with the full buffer.
    std::vector<double> ref(kRefSize);
    fn(kSeed, ref.data(), kRefSize);

    int failures = 0;

    for (int n = 1; n <= kMaxCheck; ++n) {
        // Fill a buffer slightly larger than n with the sentinel so we can
        // detect any over-write.
        std::vector<double> buf(n + kSentinelCount, kSentinel);

        fn(kSeed, buf.data(), static_cast<std::size_t>(n));

        // 1. Prefix values must match the reference exactly (bit-for-bit).
        for (int i = 0; i < n; ++i) {
            std::uint64_t ref_bits, buf_bits;
            std::memcpy(&ref_bits, &ref[i], 8);
            std::memcpy(&buf_bits, &buf[i], 8);
            if (ref_bits != buf_bits) {
                std::printf("FAIL  %-48s  count=%2d  idx=%2d  ref=%+.17g  got=%+.17g\n",
                            name.c_str(), n, i, ref[i], buf[i]);
                ++failures;
                break;  // one mismatch per count is diagnostic enough
            }
        }

        // 2. No write must occur beyond position n-1.
        for (int i = n; i < n + kSentinelCount; ++i) {
            std::uint64_t sentinel_bits, buf_bits;
            std::memcpy(&sentinel_bits, &kSentinel, 8);
            std::memcpy(&buf_bits, &buf[i], 8);
            if (sentinel_bits != buf_bits) {
                std::printf("FAIL  %-48s  count=%2d  overwrite at idx=%2d\n",
                            name.c_str(), n, i);
                ++failures;
                break;
            }
        }
    }

    if (failures == 0)
        std::printf("PASS  %s\n", name.c_str());

    return failures;
}

}  // namespace

int main() {
    int total_failures = 0;

#ifdef __AVX512F__
    total_failures += run_prefix_test(
        "fill_xoshiro256pp_x8_uniform01_avx512",
        [](std::uint64_t seed, double* out, std::size_t count) {
            zorro_bench::fill_xoshiro256pp_x8_uniform01_avx512(seed, out, count);
        });

    total_failures += run_prefix_test(
        "fill_xoshiro256pp_x16_uniform01_avx512",
        [](std::uint64_t seed, double* out, std::size_t count) {
            zorro_bench::fill_xoshiro256pp_x16_uniform01_avx512(seed, out, count);
        });

    // Normal (rejection sampling): prefix-consistent because accepted pairs are
    // emitted in deterministic order — the same RNG draws are consumed up to
    // position i regardless of the total requested count.
    total_failures += run_prefix_test(
        "fill_xoshiro256pp_x16_normal_vecpolar_avx512",
        [](std::uint64_t seed, double* out, std::size_t count) {
            zorro_bench::fill_xoshiro256pp_x16_normal_vecpolar_avx512(seed, out, count);
        });

    total_failures += run_prefix_test(
        "fill_xoshiro256pp_x16_exponential_avx512",
        [](std::uint64_t seed, double* out, std::size_t count) {
            zorro_bench::fill_xoshiro256pp_x16_exponential_avx512(seed, out, count);
        });

    total_failures += run_prefix_test(
        "fill_xoshiro256pp_x16_bernoulli_avx512 (p=0.3)",
        [](std::uint64_t seed, double* out, std::size_t count) {
            zorro_bench::fill_xoshiro256pp_x16_bernoulli_avx512(seed, 0.3, out, count);
        });

    total_failures += run_prefix_test(
        "fill_xoshiro256pp_x16_bernoulli_avx512 (p=0.0)",
        [](std::uint64_t seed, double* out, std::size_t count) {
            zorro_bench::fill_xoshiro256pp_x16_bernoulli_avx512(seed, 0.0, out, count);
        });

    total_failures += run_prefix_test(
        "fill_xoshiro256pp_x16_bernoulli_avx512 (p=1.0)",
        [](std::uint64_t seed, double* out, std::size_t count) {
            zorro_bench::fill_xoshiro256pp_x16_bernoulli_avx512(seed, 1.0, out, count);
        });
#else
    std::printf("SKIP  AVX-512 not available at compile time\n");
#endif

    if (total_failures == 0)
        std::printf("\nAll tests passed.\n");
    else
        std::printf("\n%d test(s) FAILED.\n", total_failures);

    return total_failures != 0 ? 1 : 0;
}
