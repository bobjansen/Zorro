#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

#include "benchmarks/zorro_benchmark_kernels.hpp"
#include "zorro/zorro.hpp"

namespace {

constexpr std::uint64_t kSeed = 0x123456789abcdef0ULL;
constexpr std::size_t kLargeSampleCount = 1u << 28;
constexpr std::size_t kKsSampleCount = 1u << 22;

struct SampleSummary {
    double mean = 0.0;
    double variance = 0.0;
    double skewness = 0.0;
    double excess_kurtosis = 0.0;
    double lag1_corr = 0.0;
    double ks_d = 0.0;
    double tail3 = 0.0;
    double tail4 = 0.0;
    double tail5 = 0.0;
};

struct MethodResult {
    std::string name;
    SampleSummary large;
    SampleSummary ks;
};

auto normal_cdf(double x) -> double {
    return 0.5 * (1.0 + std::erf(x / std::numbers::sqrt2_v<double>));
}

auto lag1_correlation(const std::vector<double>& xs) -> double {
    if (xs.size() < 2)
        return 0.0;
    long double sum_x = 0.0L;
    long double sum_y = 0.0L;
    long double sum_xx = 0.0L;
    long double sum_yy = 0.0L;
    long double sum_xy = 0.0L;
    for (std::size_t i = 1; i < xs.size(); ++i) {
        const long double x = xs[i - 1];
        const long double y = xs[i];
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_yy += y * y;
        sum_xy += x * y;
    }
    const long double n = static_cast<long double>(xs.size() - 1);
    const long double cov = n * sum_xy - sum_x * sum_y;
    const long double var_x = n * sum_xx - sum_x * sum_x;
    const long double var_y = n * sum_yy - sum_y * sum_y;
    return static_cast<double>(cov / std::sqrt(var_x * var_y));
}

auto ks_statistic(std::vector<double> samples) -> double {
    std::sort(samples.begin(), samples.end());
    const double n = static_cast<double>(samples.size());
    double d = 0.0;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const double fi = normal_cdf(samples[i]);
        const double lo = static_cast<double>(i) / n;
        const double hi = static_cast<double>(i + 1) / n;
        d = std::max(d, std::fabs(fi - lo));
        d = std::max(d, std::fabs(hi - fi));
    }
    return d;
}

auto summarize_samples(const std::vector<double>& xs, bool with_ks) -> SampleSummary {
    const long double n = static_cast<long double>(xs.size());
    long double mean = 0.0L;
    for (double x : xs)
        mean += x;
    mean /= n;

    long double m2 = 0.0L;
    long double m3 = 0.0L;
    long double m4 = 0.0L;
    std::uint64_t tail3 = 0;
    std::uint64_t tail4 = 0;
    std::uint64_t tail5 = 0;
    for (double x : xs) {
        const long double d = x - mean;
        const long double d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
        const double ax = std::abs(x);
        tail3 += ax > 3.0;
        tail4 += ax > 4.0;
        tail5 += ax > 5.0;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;

    return SampleSummary{
        .mean = static_cast<double>(mean),
        .variance = static_cast<double>(m2),
        .skewness = static_cast<double>(m3 / std::pow(m2, 1.5L)),
        .excess_kurtosis = static_cast<double>(m4 / (m2 * m2) - 3.0L),
        .lag1_corr = lag1_correlation(xs),
        .ks_d = with_ks ? ks_statistic(xs) : 0.0,
        .tail3 = static_cast<double>(tail3) / static_cast<double>(xs.size()),
        .tail4 = static_cast<double>(tail4) / static_cast<double>(xs.size()),
        .tail5 = static_cast<double>(tail5) / static_cast<double>(xs.size()),
    };
}

template <typename FillFn>
auto run_method(std::string name, FillFn&& fill) -> MethodResult {
    std::vector<double> large_samples(kLargeSampleCount);
    fill(large_samples.data(), large_samples.size());

    std::vector<double> ks_samples(kKsSampleCount);
    fill(ks_samples.data(), ks_samples.size());

    return MethodResult{
        .name = std::move(name),
        .large = summarize_samples(large_samples, false),
        .ks = summarize_samples(ks_samples, true),
    };
}

auto report_check(bool ok, const std::string& label, double value,
                  double target, double tolerance) -> int {
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "  " << label
              << " value=" << value
              << " target=" << target
              << " tol=" << tolerance << '\n';
    return ok ? 0 : 1;
}

auto check_near(const std::string& label, double value,
                double target, double tolerance) -> int {
    return report_check(std::abs(value - target) <= tolerance,
                        label, value, target, tolerance);
}

auto check_abs_le(const std::string& label, double value,
                  double tolerance) -> int {
    return report_check(std::abs(value) <= tolerance,
                        label, value, 0.0, tolerance);
}

void print_result(const MethodResult& result) {
    std::cout << result.name << '\n';
    std::cout << "  large mean=" << result.large.mean
              << " variance=" << result.large.variance
              << " skew=" << result.large.skewness
              << " kurt=" << result.large.excess_kurtosis << '\n';
    std::cout << "  large tails: P(|x|>3)=" << result.large.tail3
              << " P(|x|>4)=" << result.large.tail4
              << " P(|x|>5)=" << result.large.tail5 << '\n';
    std::cout << "  ks sample: D=" << result.ks.ks_d
              << " lag1=" << result.ks.lag1_corr << '\n';
}

auto check_distribution(const MethodResult& result) -> int {
    int failures = 0;
    constexpr double ref_tail3 = 2.699796063260188e-03;
    constexpr double ref_tail4 = 6.334248366623993e-05;
    constexpr double ref_tail5 = 5.733031437583892e-07;

    std::cout << "\nChecking " << result.name << '\n';
    failures += check_abs_le("large mean", result.large.mean, 5e-4);
    failures += check_near("large variance", result.large.variance, 1.0, 3e-3);
    failures += check_abs_le("large skewness", result.large.skewness, 5e-3);
    failures += check_abs_le("large excess kurtosis", result.large.excess_kurtosis, 1.5e-2);
    failures += check_near("tail |x|>3", result.large.tail3, ref_tail3, 5e-5);
    failures += check_near("tail |x|>4", result.large.tail4, ref_tail4, 1e-5);
    failures += check_near("tail |x|>5", result.large.tail5, ref_tail5, 1.5e-6);
    failures += check_abs_le("ks lag1 corr", result.ks.lag1_corr, 5e-3);
    failures += check_abs_le("ks D", result.ks.ks_d, 2e-3);
    return failures;
}

auto check_against_exact(const MethodResult& exact,
                         const MethodResult& approx) -> int {
    int failures = 0;
    std::cout << "\nComparing " << approx.name << " against " << exact.name << '\n';
    failures += check_near("mean vs exact", approx.large.mean, exact.large.mean, 3e-4);
    failures += check_near("variance vs exact", approx.large.variance, exact.large.variance, 2e-3);
    failures += check_near("tail3 vs exact", approx.large.tail3, exact.large.tail3, 3e-5);
    failures += check_near("tail4 vs exact", approx.large.tail4, exact.large.tail4, 8e-6);
    failures += check_near("tail5 vs exact", approx.large.tail5, exact.large.tail5, 1.2e-6);
    failures += check_near("ks D vs exact", approx.ks.ks_d, exact.ks.ks_d, 1.5e-3);
    return failures;
}

}  // namespace

int main() {
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "Normal transform quality tests\n";
    std::cout << "  large samples: " << kLargeSampleCount << '\n';
    std::cout << "  ks samples:    " << kKsSampleCount << '\n';
    std::cout << "  seed:          0x" << std::hex << kSeed << std::dec << '\n';

    auto public_result = run_method(
        "public zorro::Rng::fill_normal",
        [](double* out, std::size_t count) {
            zorro::Rng rng(kSeed);
            rng.fill_normal(out, count);
        });

    auto exact_result = run_method(
        "x8 AVX2 box-muller exact",
        [](double* out, std::size_t count) {
            zorro_bench::fill_xoshiro256pp_x8_normal_box_muller_avx2(kSeed, out, count);
        });

    auto approxsincos_result = run_method(
        "x8 AVX2 box-muller approxsincos",
        [](double* out, std::size_t count) {
            zorro_bench::fill_xoshiro256pp_x8_normal_box_muller_avx2_approxsincos(
                kSeed, out, count);
        });

    auto fullapprox_result = run_method(
        "x8 AVX2 box-muller fullapprox",
        [](double* out, std::size_t count) {
            zorro_bench::fill_xoshiro256pp_x8_normal_box_muller_avx2_fullapprox(
                kSeed, out, count);
        });

    print_result(public_result);
    print_result(exact_result);
    print_result(approxsincos_result);
    print_result(fullapprox_result);

    int failures = 0;
    failures += check_distribution(public_result);
    failures += check_distribution(exact_result);
    failures += check_distribution(approxsincos_result);
    failures += check_distribution(fullapprox_result);
    failures += check_against_exact(exact_result, approxsincos_result);
    failures += check_against_exact(exact_result, fullapprox_result);

    if (failures != 0) {
        std::cout << "\n" << failures << " normal-transform check(s) FAILED.\n";
        return 1;
    }

    std::cout << "\nAll normal-transform checks passed.\n";
    return 0;
}
