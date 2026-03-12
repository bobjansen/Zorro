#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
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

constexpr std::size_t kSampleCount = 1u << 24;
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
        if (i < count)
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
            if (i < count)
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
            if (i < count)
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
    uniform_results.push_back(run_benchmark("xoshiro256++ x4 fused portable",
                                            fill_uniform_xoshiro_x4_fused));

#ifdef RNG_BENCH_ENABLE_STEPHANFR_AVX2
    uniform_results.push_back(
        run_benchmark("stephanfr xoshiro256+ AVX2 + uniform01_52",
                      zorro_bench::fill_uniform_stephanfr_avx2_52));
#endif

    std::vector<BenchmarkResult> normal_results;
    normal_results.reserve(8);
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
    normal_results.push_back(run_benchmark("xoshiro256++ x8 AVX2 + vecpolar",
                                           fill_normals_xoshiro_x8_avx2_vecpolar));
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

    print_results("Uniform(0, 1)", uniform_results);
    print_results("Normal(0, 1)", normal_results);
    std::cout << "checksum sink: " << std::setprecision(17) << g_checksum_sink
              << '\n';
    return EXIT_SUCCESS;
}
