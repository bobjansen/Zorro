#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "zorro/zorro.hpp"
#include "benchmarks/zorro_benchmark_kernels.hpp"

#ifdef RNG_BENCH_ENABLE_STEPHANFR_AVX2
#include "benchmarks/stephanfr_adapters.hpp"
#endif

namespace {

using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;

constexpr std::size_t kSampleCount = 1u << 20;
constexpr std::uint64_t kSeed = 0x123456789abcdef0ULL;
constexpr int kWarmupIterations = 3;
constexpr int kMeasureIterations = 21;
constexpr double kBitsToPm1Scale = 0x1.0p-52;

volatile double g_checksum_sink = 0.0;

struct BenchmarkResult {
    std::string name;
    double best_ms;
    double median_ms;
    double mean_ms;
    double samples_per_second;
    double checksum;
};

auto checksum_buffer(const std::vector<double>& samples) -> double {
    return std::accumulate(samples.begin(), samples.end(), 0.0);
}

void fill_uniform_local(zorro::Xoshiro256pp& rng, double* out,
                        std::size_t count) {
    for (std::size_t i = 0; i < count; ++i)
        out[i] = zorro::bits_to_01(rng());
}

void fill_uniform_local(zorro::Xoshiro256pp_x2& rng, double* out,
                        std::size_t count) {
    std::size_t i = 0;
    while (i + 2 <= count) {
        const auto bits = rng();
        out[i] = zorro::bits_to_01(bits[0]);
        out[i + 1] = zorro::bits_to_01(bits[1]);
        i += 2;
    }
    if (i < count) {
        const auto bits = rng();
        out[i] = zorro::bits_to_01(bits[0]);
    }
}

void fill_normal_local(zorro::Xoshiro256pp& rng, double* out,
                       std::size_t count) {
    std::size_t i = 0;
    while (i < count) {
        const double u1 = zorro::bits_to_pm1(rng());
        const double u2 = zorro::bits_to_pm1(rng());
        const double s = u1 * u1 + u2 * u2;
        if (s >= 1.0 || s == 0.0)
            continue;
        const double scale = std::sqrt(-2.0 * std::log(s) / s);
        out[i++] = u1 * scale;
        if (i < count) [[likely]]
            out[i++] = u2 * scale;
    }
}

void fill_normal_local(zorro::Xoshiro256pp_x2& rng, double* out,
                       std::size_t count) {
    std::size_t i = 0;
    while (i < count) {
        const auto r1 = rng();
        const auto r2 = rng();
        for (std::size_t lane = 0; lane < 2 && i < count; ++lane) {
            const double u1 = zorro::bits_to_pm1(r1[lane]);
            const double u2 = zorro::bits_to_pm1(r2[lane]);
            const double s = u1 * u1 + u2 * u2;
            if (s >= 1.0 || s == 0.0)
                continue;
            const double scale = std::sqrt(-2.0 * std::log(s) / s);
            out[i++] = u1 * scale;
            if (i < count) [[likely]]
                out[i++] = u2 * scale;
        }
    }
}

void fill_normal_local(zorro::Xoshiro256pp_x4_portable& rng, double* out,
                       std::size_t count) {
    std::size_t i = 0;
    while (i < count) {
        const auto r1 = rng();
        const auto r2 = rng();
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
            if (i < count) [[likely]]
                out[i++] = u2 * scale;
        }
    }
}

template <typename FillFn>
auto run_benchmark(std::string name, FillFn&& fill) -> BenchmarkResult {
    std::vector<double> samples(kSampleCount);

    for (int i = 0; i < kWarmupIterations; ++i)
        fill(samples.data(), samples.size());

    std::vector<double> times_ms;
    times_ms.reserve(kMeasureIterations);
    double checksum = 0.0;

    for (int i = 0; i < kMeasureIterations; ++i) {
        const auto start = Clock::now();
        fill(samples.data(), samples.size());
        const auto stop = Clock::now();
        times_ms.push_back(Milliseconds(stop - start).count());
        checksum += checksum_buffer(samples);
    }

    std::vector<double> sorted = times_ms;
    std::sort(sorted.begin(), sorted.end());

    const double best_ms = sorted.front();
    const double median_ms = sorted[sorted.size() / 2];
    const double mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) /
                           static_cast<double>(times_ms.size());
    const double samples_per_second =
        static_cast<double>(kSampleCount) / (best_ms / 1000.0);

    g_checksum_sink += checksum;

    return BenchmarkResult{
        .name = std::move(name),
        .best_ms = best_ms,
        .median_ms = median_ms,
        .mean_ms = mean_ms,
        .samples_per_second = samples_per_second,
        .checksum = checksum,
    };
}

void fill_uniform_xoshiro_scalar(double* out, std::size_t count) {
    zorro::Xoshiro256pp rng(kSeed);
    fill_uniform_local(rng, out, count);
}

void fill_uniform_xoshiro_x2(double* out, std::size_t count) {
    zorro::Xoshiro256pp_x2 rng(kSeed);
    fill_uniform_local(rng, out, count);
}

void fill_uniform_xoshiro_x4(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x4_uniform01(kSeed, out, count);
}

#ifdef __AVX2__
void fill_uniform_xoshiro_x4_avx2(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x4_uniform01_avx2(kSeed, out, count);
}

void fill_uniform_xoshiro_plus_x4_avx2(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256p_x4_uniform01_avx2(kSeed, out, count);
}

void fill_uniform_xoshiro_x4_avx2_unroll2(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x4_uniform01_avx2_unroll2(kSeed, out, count);
}

void fill_uniform_xoshiro_x8_avx2(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_uniform01_avx2(kSeed, out, count);
}

void fill_normals_xoshiro_x8_avx2(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_normal_polar_avx2(kSeed, out, count);
}

void fill_normals_xoshiro_x8_avx2_batched(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_normal_polar_avx2_batched(kSeed, out, count);
}

void fill_normals_xoshiro_x8_avx2_veclog(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_normal_polar_avx2_veclog(kSeed, out, count);
}

void fill_normals_xoshiro_x8_avx2_box_muller(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_normal_box_muller_avx2(kSeed, out, count);
}

void fill_normals_xoshiro_x8_avx2_box_muller_fastlog(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_normal_box_muller_avx2_fastlog(kSeed, out, count);
}

void fill_normals_xoshiro_x8_avx2_box_muller_approxsincos(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_normal_box_muller_avx2_approxsincos(
        kSeed, out, count);
}

void fill_normals_xoshiro_x8_avx2_box_muller_fullapprox(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_normal_box_muller_avx2_fullapprox(
        kSeed, out, count);
}

void fill_normals_xoshiro_x8_avx2_vecpolar(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_normal_vecpolar_avx2(kSeed, out, count);
}
#endif

void fill_uniform_xoshiro_x4_fused(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x4_uniform01_fused(kSeed, out, count);
}

void fill_normals_xoshiro_scalar_polar(double* out, std::size_t count) {
    zorro::Xoshiro256pp rng(kSeed);
    fill_normal_local(rng, out, count);
}

void fill_normals_xoshiro_x2(double* out, std::size_t count) {
    zorro::Xoshiro256pp_x2 rng(kSeed);
    fill_normal_local(rng, out, count);
}

void fill_normals_xoshiro_x4(double* out, std::size_t count) {
    zorro::Xoshiro256pp_x4_portable rng(kSeed);
    fill_normal_local(rng, out, count);
}

#ifdef __AVX2__
void fill_normals_xoshiro_x4_avx2(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x4_normal_polar_avx2(kSeed, out, count);
}

void fill_normals_xoshiro_plus_x4_avx2(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256p_x4_normal_polar_avx2(kSeed, out, count);
}

void fill_exponential_x8_naive(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_exponential_naive(kSeed, out, count);
}

void fill_exponential_x8_veclog(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_exponential_avx2(kSeed, out, count);
}

void fill_exponential_x8_fastlog(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_exponential_avx2_fastlog(kSeed, out, count);
}

void fill_bernoulli_x8_naive(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_bernoulli_naive(kSeed, 0.3, out, count);
}

void fill_bernoulli_x8_fast(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_bernoulli_fast(kSeed, 0.3, out, count);
}

void fill_bernoulli_x8_half_naive(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_bernoulli_naive(kSeed, 0.5, out, count);
}

void fill_bernoulli_x8_half_fast(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_bernoulli_fast(kSeed, 0.5, out, count);
}

void fill_bernoulli_x8_half_bits(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_bernoulli_half(kSeed, out, count);
}
#endif

// ── uint8_t Bernoulli benchmarks ─────────────────────────────────────────────

auto checksum_buffer_u8(const std::vector<std::uint8_t>& samples) -> double {
    std::uint64_t sum = 0;
    for (auto v : samples) sum += v;
    return static_cast<double>(sum);
}

template <typename FillFn>
auto run_benchmark_u8(std::string name, FillFn&& fill) -> BenchmarkResult {
    std::vector<std::uint8_t> samples(kSampleCount);

    for (int i = 0; i < kWarmupIterations; ++i)
        fill(samples.data(), samples.size());

    std::vector<double> times_ms;
    times_ms.reserve(kMeasureIterations);
    double checksum = 0.0;

    for (int i = 0; i < kMeasureIterations; ++i) {
        const auto start = Clock::now();
        fill(samples.data(), samples.size());
        const auto stop = Clock::now();
        times_ms.push_back(Milliseconds(stop - start).count());
        checksum += checksum_buffer_u8(samples);
    }

    std::vector<double> sorted = times_ms;
    std::sort(sorted.begin(), sorted.end());

    const double best_ms = sorted.front();
    const double median_ms = sorted[sorted.size() / 2];
    const double mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) /
                           static_cast<double>(times_ms.size());
    const double samples_per_second =
        static_cast<double>(kSampleCount) / (best_ms / 1000.0);

    g_checksum_sink += checksum;

    return BenchmarkResult{
        .name = std::move(name),
        .best_ms = best_ms,
        .median_ms = median_ms,
        .mean_ms = mean_ms,
        .samples_per_second = samples_per_second,
        .checksum = checksum,
    };
}

#ifdef __AVX2__
void fill_bernoulli_u8_naive(std::uint8_t* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_bernoulli_u8_naive(kSeed, 0.3, out, count);
}

void fill_bernoulli_u8_fast(std::uint8_t* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_bernoulli_u8_fast(kSeed, 0.3, out, count);
}

void fill_bernoulli_u8_half_naive(std::uint8_t* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_bernoulli_u8_naive(kSeed, 0.5, out, count);
}

void fill_bernoulli_u8_half_fast(std::uint8_t* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_bernoulli_u8_fast(kSeed, 0.5, out, count);
}

void fill_bernoulli_u8_half_bits(std::uint8_t* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_bernoulli_u8_half(kSeed, out, count);
}

void fill_bernoulli_u8_half_bits_skip16(std::uint8_t* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_bernoulli_u8_half_skip16(kSeed, out, count);
}

void fill_bernoulli_bits(std::uint64_t* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_bernoulli_bits(kSeed, out, count);
}
#endif

// run_benchmark_bits: bitmask output benchmark. count is in samples (bits).
// Buffer is uint64_t, checksum via popcount for comparability with uint8_t path.
template <typename FillFn>
auto run_benchmark_bits(std::string name, FillFn&& fill) -> BenchmarkResult {
    // 52 usable bits per word → need ceil(kSampleCount/52) words
    static constexpr std::size_t kBitsPerWord = 52;
    static constexpr std::size_t kNumWords =
        (kSampleCount + kBitsPerWord - 1) / kBitsPerWord;
    std::vector<std::uint64_t> words(kNumWords + 4);  // +4 for tail overwrite

    for (int i = 0; i < kWarmupIterations; ++i)
        fill(words.data(), kSampleCount);

    std::vector<double> times_ms;
    times_ms.reserve(kMeasureIterations);
    double checksum = 0.0;

    for (int i = 0; i < kMeasureIterations; ++i) {
        const auto start = Clock::now();
        fill(words.data(), kSampleCount);
        const auto stop = Clock::now();
        times_ms.push_back(Milliseconds(stop - start).count());
        std::uint64_t popsum = 0;
        for (std::size_t w = 0; w < kNumWords; ++w)
            popsum += __builtin_popcountll(words[w]);
        checksum += static_cast<double>(popsum);
    }

    std::vector<double> sorted = times_ms;
    std::sort(sorted.begin(), sorted.end());

    const double best_ms = sorted.front();
    const double median_ms = sorted[sorted.size() / 2];
    const double mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) /
                           static_cast<double>(times_ms.size());
    const double samples_per_second =
        static_cast<double>(kSampleCount) / (best_ms / 1000.0);

    g_checksum_sink += checksum;

    return BenchmarkResult{
        .name = std::move(name),
        .best_ms = best_ms,
        .median_ms = median_ms,
        .mean_ms = mean_ms,
        .samples_per_second = samples_per_second,
        .checksum = checksum,
    };
}

// ── Gamma(2, 1) ──────────────────────────────────────────────────────────────
void fill_gamma_scalar(double* out, std::size_t count) {
    zorro_bench::fill_gamma_scalar_fused(kSeed, 2.0, out, count);
}
#ifdef __AVX2__
void fill_gamma_avx2_fused(double* out, std::size_t count) {
    zorro_bench::fill_gamma_x8_avx2_fused(kSeed, 2.0, out, count);
}
void fill_gamma_avx2_decoupled(double* out, std::size_t count) {
    zorro_bench::fill_gamma_x8_avx2_decoupled(kSeed, 2.0, out, count);
}
void fill_gamma_avx2_full(double* out, std::size_t count) {
    zorro_bench::fill_gamma_x8_avx2_full(kSeed, 2.0, out, count);
}
#endif

// ── Student's t(5) ───────────────────────────────────────────────────────────
void fill_student_t_scalar(double* out, std::size_t count) {
    zorro_bench::fill_student_t_scalar_fused(kSeed, 5.0, out, count);
}
#ifdef __AVX2__
void fill_student_t_avx2_fused(double* out, std::size_t count) {
    zorro_bench::fill_student_t_x8_avx2_fused(kSeed, 5.0, out, count);
}
void fill_student_t_avx2_decoupled(double* out, std::size_t count) {
    zorro_bench::fill_student_t_x8_avx2_decoupled(kSeed, 5.0, out, count);
}
void fill_student_t_avx2_fast(double* out, std::size_t count) {
    zorro_bench::fill_student_t_x8_avx2_fast(kSeed, 5.0, out, count);
}
#endif

#ifdef __AVX512F__
void fill_uniform_xoshiro_x8_avx512(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x8_uniform01_avx512(kSeed, out, count);
}

void fill_uniform_xoshiro_x16_avx512(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x16_uniform01_avx512(kSeed, out, count);
}

void fill_normals_xoshiro_x16_avx512_vecpolar(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x16_normal_vecpolar_avx512(kSeed, out, count);
}

void fill_normals_xoshiro_x16_avx512_fullapprox(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x16_normal_box_muller_avx512_fullapprox(
        kSeed, out, count);
}

void fill_exponential_x16_avx512(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x16_exponential_avx512(kSeed, out, count);
}

void fill_bernoulli_x16_avx512(double* out, std::size_t count) {
    zorro_bench::fill_xoshiro256pp_x16_bernoulli_avx512(kSeed, 0.3, out, count);
}
#endif

void print_header() {
    std::cout << "Benchmark: 2^20 samples\n";
    std::cout << "Samples:   " << kSampleCount << '\n';
    std::cout << "Seed:      0x" << std::hex << kSeed << std::dec << '\n';
    std::cout << "Warmups:   " << kWarmupIterations << '\n';
    std::cout << "Measures:  " << kMeasureIterations << '\n';
#ifdef RNG_BENCH_ENABLE_STEPHANFR_AVX2
    std::cout << "AVX2:      enabled\n\n";
#else
    std::cout << "AVX2:      disabled\n\n";
#endif
}

void print_results(const std::string& suite_name,
                   const std::vector<BenchmarkResult>& results) {
    const double suite_checksum =
        std::accumulate(results.begin(), results.end(), 0.0,
                        [](double sum, const BenchmarkResult& result) {
                            return sum + result.checksum;
                        });

    std::cout << suite_name << '\n';
    std::cout << std::left << std::setw(34) << "benchmark"
              << std::right << std::setw(12) << "best ms"
              << std::setw(12) << "median"
              << std::setw(12) << "mean"
              << std::setw(16) << "M samples/s"
              << std::setw(16) << "ns/sample"
              << '\n';

    std::cout << std::string(102, '-') << '\n';

    for (const auto& result : results) {
        const double ns_per_sample =
            (result.best_ms * 1'000'000.0) / static_cast<double>(kSampleCount);
        std::cout << std::left << std::setw(34) << result.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(3)
                  << result.best_ms
                  << std::setw(12) << result.median_ms
                  << std::setw(12) << result.mean_ms
                  << std::setw(16) << (result.samples_per_second / 1'000'000.0)
                  << std::setw(16) << ns_per_sample
                  << '\n';
    }

    std::cout << "\nsuite checksum: " << std::setprecision(17) << suite_checksum
              << "\n\n";
}

}  // namespace

int main() {
    print_header();

    std::vector<BenchmarkResult> uniform_results;
    uniform_results.reserve(8);
    uniform_results.push_back(
        run_benchmark("xoshiro256++ scalar + bits_to_01",
                      fill_uniform_xoshiro_scalar));
    uniform_results.push_back(run_benchmark("xoshiro256++ x2 + bits_to_01",
                                            fill_uniform_xoshiro_x2));
    uniform_results.push_back(run_benchmark("xoshiro256++ x4 + bits_to_01",
                                            fill_uniform_xoshiro_x4));
#ifdef __AVX2__
    uniform_results.push_back(run_benchmark("xoshiro256++ x4 AVX2 + uniform01_52",
                                            fill_uniform_xoshiro_x4_avx2));
    uniform_results.push_back(run_benchmark("xoshiro256+ x4 AVX2 + uniform01_52",
                                            fill_uniform_xoshiro_plus_x4_avx2));
    uniform_results.push_back(run_benchmark("xoshiro256++ x4 AVX2 unroll2",
                                            fill_uniform_xoshiro_x4_avx2_unroll2));
    uniform_results.push_back(run_benchmark("xoshiro256++ x8 AVX2 (2×4)",
                                            fill_uniform_xoshiro_x8_avx2));
#endif
#ifdef __AVX512F__
    uniform_results.push_back(run_benchmark("xoshiro256++ x8 AVX-512 (1×8)",
                                            fill_uniform_xoshiro_x8_avx512));
    uniform_results.push_back(run_benchmark("xoshiro256++ x16 AVX-512 (2×8)",
                                            fill_uniform_xoshiro_x16_avx512));
#endif
    uniform_results.push_back(run_benchmark("xoshiro256++ x4 fused portable",
                                            fill_uniform_xoshiro_x4_fused));

#ifdef RNG_BENCH_ENABLE_STEPHANFR_AVX2
    uniform_results.push_back(
        run_benchmark("stephanfr xoshiro256+ AVX2 + uniform01_52",
                      zorro_bench::fill_uniform_stephanfr_avx2_52));
#endif

    std::vector<BenchmarkResult> normal_results;
    normal_results.reserve(16);
    zorro::Rng persistent_normal_rng(kSeed);
    normal_results.push_back(run_benchmark(
        "zorro::Rng fill_normal persistent",
        [&](double* out, std::size_t count) {
            persistent_normal_rng.fill_normal(out, count);
        }));
    normal_results.push_back(run_benchmark(
        "zorro::Rng fill_normal fresh",
        [](double* out, std::size_t count) {
            zorro::Rng rng(kSeed);
            rng.fill_normal(out, count);
        }));
    normal_results.push_back(
        run_benchmark("xoshiro256++ scalar + polar",
                      fill_normals_xoshiro_scalar_polar));
    normal_results.push_back(run_benchmark("xoshiro256++ x2 + polar",
                                           fill_normals_xoshiro_x2));
    normal_results.push_back(run_benchmark("xoshiro256++ x4 + polar",
                                           fill_normals_xoshiro_x4));
#ifdef __AVX2__
    normal_results.push_back(run_benchmark("xoshiro256++ x4 AVX2 + polar pm1_52",
                                           fill_normals_xoshiro_x4_avx2));
    normal_results.push_back(run_benchmark("xoshiro256+ x4 AVX2 + polar pm1_52",
                                           fill_normals_xoshiro_plus_x4_avx2));
    normal_results.push_back(run_benchmark("xoshiro256++ x8 AVX2 + polar extract",
                                           fill_normals_xoshiro_x8_avx2));
    normal_results.push_back(run_benchmark("xoshiro256++ x8 AVX2 + polar batched",
                                           fill_normals_xoshiro_x8_avx2_batched));
    normal_results.push_back(run_benchmark("xoshiro256++ x8 AVX2 + veclog batch",
                                           fill_normals_xoshiro_x8_avx2_veclog));
    normal_results.push_back(run_benchmark("xoshiro256++ x8 AVX2 + box-muller",
                                           fill_normals_xoshiro_x8_avx2_box_muller));
    normal_results.push_back(run_benchmark("xoshiro256++ x8 AVX2 + box-muller fastlog",
                                           fill_normals_xoshiro_x8_avx2_box_muller_fastlog));
    normal_results.push_back(
        run_benchmark("xoshiro256++ x8 AVX2 + box-muller approxsincos",
                      fill_normals_xoshiro_x8_avx2_box_muller_approxsincos));
    normal_results.push_back(
        run_benchmark("xoshiro256++ x8 AVX2 + box-muller fullapprox",
                      fill_normals_xoshiro_x8_avx2_box_muller_fullapprox));
    normal_results.push_back(run_benchmark("xoshiro256++ x8 AVX2 + vecpolar",
                                           fill_normals_xoshiro_x8_avx2_vecpolar));
#endif
#ifdef __AVX512F__
    normal_results.push_back(run_benchmark("xoshiro256++ x16 AVX-512 + box-muller fullapprox",
                                           fill_normals_xoshiro_x16_avx512_fullapprox));
    normal_results.push_back(run_benchmark("xoshiro256++ x16 AVX-512 + vecpolar",
                                           fill_normals_xoshiro_x16_avx512_vecpolar));
#endif

#ifdef RNG_BENCH_ENABLE_STEPHANFR_AVX2
    normal_results.push_back(
        run_benchmark("stephanfr xoshiro256+ AVX2 + polar pm1_52",
                      zorro_bench::fill_normals_stephanfr_avx2_52));
    normal_results.push_back(
        run_benchmark("stephanfr xoshiro256+ AVX2 + polar batched",
                      zorro_bench::fill_normals_stephanfr_avx2_52_batched));
    normal_results.push_back(
        run_benchmark("stephanfr xoshiro256+ AVX2 + veclog batch",
                      zorro_bench::fill_normals_stephanfr_avx2_52_veclog));
    normal_results.push_back(
        run_benchmark("stephanfr xoshiro256+ AVX2 + vecpolar",
                      zorro_bench::fill_normals_stephanfr_avx2_52_vecpolar));
#endif

    std::vector<BenchmarkResult> exponential_results;
    exponential_results.reserve(6);
    zorro::Rng persistent_exponential_rng(kSeed);
    exponential_results.push_back(run_benchmark(
        "zorro::Rng fill_exponential persistent",
        [&](double* out, std::size_t count) {
            persistent_exponential_rng.fill_exponential(out, count);
        }));
    exponential_results.push_back(run_benchmark(
        "zorro::Rng fill_exponential fresh",
        [](double* out, std::size_t count) {
            zorro::Rng rng(kSeed);
            rng.fill_exponential(out, count);
        }));
#ifdef __AVX2__
    exponential_results.push_back(run_benchmark("x8 AVX2 + scalar -log(u)",
                                                fill_exponential_x8_naive));
    exponential_results.push_back(run_benchmark("x8 AVX2 + libmvec -log(u)",
                                                fill_exponential_x8_veclog));
    exponential_results.push_back(run_benchmark("x8 AVX2 + fastlog -log(1-u)",
                                                fill_exponential_x8_fastlog));
#endif
#ifdef __AVX512F__
    exponential_results.push_back(run_benchmark("x16 AVX-512 + libmvec -log(u)",
                                                fill_exponential_x16_avx512));
#endif

    std::vector<BenchmarkResult> bernoulli_results;
    bernoulli_results.reserve(4);
#ifdef __AVX2__
    bernoulli_results.push_back(run_benchmark("x8 AVX2 naive (uniform + cmp)",
                                              fill_bernoulli_x8_naive));
    bernoulli_results.push_back(run_benchmark("x8 AVX2 fast (int threshold)",
                                              fill_bernoulli_x8_fast));
#endif
#ifdef __AVX512F__
    bernoulli_results.push_back(run_benchmark("x16 AVX-512 (native ucmp)",
                                              fill_bernoulli_x16_avx512));
#endif

    std::vector<BenchmarkResult> bernoulli_half_results;
    bernoulli_half_results.reserve(4);
#ifdef __AVX2__
    bernoulli_half_results.push_back(run_benchmark("x8 AVX2 naive (uniform + cmp)",
                                                    fill_bernoulli_x8_half_naive));
    bernoulli_half_results.push_back(run_benchmark("x8 AVX2 fast (int threshold)",
                                                    fill_bernoulli_x8_half_fast));
    bernoulli_half_results.push_back(run_benchmark("x8 AVX2 bit-unpack (1 bit = 1 sample)",
                                                    fill_bernoulli_x8_half_bits));
#endif

    std::vector<BenchmarkResult> gamma_results;
    gamma_results.reserve(5);
    gamma_results.push_back(run_benchmark("scalar fused",          fill_gamma_scalar));
#ifdef __AVX2__
    gamma_results.push_back(run_benchmark("x8 AVX2 fused",         fill_gamma_avx2_fused));
    gamma_results.push_back(run_benchmark("x8 AVX2 decoupled",     fill_gamma_avx2_decoupled));
    gamma_results.push_back(run_benchmark("x8+x4 AVX2 full",       fill_gamma_avx2_full));
#endif

    std::vector<BenchmarkResult> student_t_results;
    student_t_results.reserve(5);
    student_t_results.push_back(run_benchmark("scalar fused",      fill_student_t_scalar));
#ifdef __AVX2__
    student_t_results.push_back(run_benchmark("x8 AVX2 fused",     fill_student_t_avx2_fused));
    student_t_results.push_back(run_benchmark("x8 AVX2 decoupled", fill_student_t_avx2_decoupled));
    student_t_results.push_back(run_benchmark("x8+x4 AVX2 fast",   fill_student_t_avx2_fast));
#endif

    print_results("Uniform(0, 1)", uniform_results);
    print_results("Normal(0, 1)", normal_results);
    print_results("Exponential(1)", exponential_results);
    print_results("Bernoulli(0.3)", bernoulli_results);
    print_results("Bernoulli(0.5)", bernoulli_half_results);

    std::vector<BenchmarkResult> bernoulli_u8_results;
    bernoulli_u8_results.reserve(4);
#ifdef __AVX2__
    bernoulli_u8_results.push_back(run_benchmark_u8("x8 u8 naive (uniform + cmp)",
                                                     fill_bernoulli_u8_naive));
    bernoulli_u8_results.push_back(run_benchmark_u8("x8 u8 fast (int threshold)",
                                                     fill_bernoulli_u8_fast));
#endif

    std::vector<BenchmarkResult> bernoulli_u8_half_results;
    bernoulli_u8_half_results.reserve(4);
#ifdef __AVX2__
    bernoulli_u8_half_results.push_back(run_benchmark_u8("x8 u8 naive (uniform + cmp)",
                                                          fill_bernoulli_u8_half_naive));
    bernoulli_u8_half_results.push_back(run_benchmark_u8("x8 u8 fast (int threshold)",
                                                          fill_bernoulli_u8_half_fast));
    bernoulli_u8_half_results.push_back(run_benchmark_u8("x8 u8 bit-unpack",
                                                          fill_bernoulli_u8_half_bits));
    bernoulli_u8_half_results.push_back(run_benchmark_u8("x8 u8 bit-unpack skip16",
                                                          fill_bernoulli_u8_half_bits_skip16));
    bernoulli_u8_half_results.push_back(run_benchmark_bits("x8 bitmask (1 bit/sample)",
                                                            fill_bernoulli_bits));
#endif

    print_results("Bernoulli(0.3) → uint8_t", bernoulli_u8_results);
    print_results("Bernoulli(0.5) → uint8_t", bernoulli_u8_half_results);
    print_results("Gamma(2, 1)", gamma_results);
    print_results("Student's t(5)", student_t_results);
    std::cout << "checksum sink: " << std::setprecision(17) << g_checksum_sink
              << '\n';
    return EXIT_SUCCESS;
}
