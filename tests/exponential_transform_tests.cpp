#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "zorro/zorro.hpp"

#ifdef __AVX2__
#include <immintrin.h>
extern "C" __m256d _ZGVdN4v_log(__m256d) noexcept;
#endif

namespace {

constexpr std::uint64_t kSeed = 0x123456789abcdef0ULL;
constexpr std::size_t kLargeSampleCount = 1u << 28;
constexpr std::size_t kChunkSize = 1u << 20;
constexpr std::size_t kKsSampleCount = 1u << 22;
constexpr std::size_t kPairwiseSampleCount = 1u << 24;

struct SampleSummary {
    double mean = 0.0;
    double variance = 0.0;
    double skewness = 0.0;
    double excess_kurtosis = 0.0;
    double lag1_corr = 0.0;
    double ks_d = 0.0;
    double tail1 = 0.0;
    double tail4 = 0.0;
    double tail8 = 0.0;
    double tail12 = 0.0;
};

struct MethodResult {
    std::string name;
    SampleSummary large;
    SampleSummary ks;
};

struct PairwiseErrorSummary {
    double mean_abs = 0.0;
    double rmse = 0.0;
    double max_abs = 0.0;
};

class StreamingAccumulator {
public:
    void push(double x) noexcept {
        ++count_;
        const long double lx = x;
        sum1_ += lx;
        sum2_ += lx * lx;
        sum3_ += lx * lx * lx;
        sum4_ += lx * lx * lx * lx;
        tail1_ += x > 1.0;
        tail4_ += x > 4.0;
        tail8_ += x > 8.0;
        tail12_ += x > 12.0;

        if (has_prev_) {
            const long double prev = prev_;
            pair_sum_x_ += prev;
            pair_sum_y_ += lx;
            pair_sum_xx_ += prev * prev;
            pair_sum_yy_ += lx * lx;
            pair_sum_xy_ += prev * lx;
        }
        prev_ = lx;
        has_prev_ = true;
    }

    auto finalize(bool with_ks, const std::vector<double>& ks_samples) const -> SampleSummary {
        const long double n = static_cast<long double>(count_);
        const long double m1 = sum1_ / n;
        const long double ex2 = sum2_ / n;
        const long double ex3 = sum3_ / n;
        const long double ex4 = sum4_ / n;
        const long double mu2 = ex2 - m1 * m1;
        const long double mu3 = ex3 - 3.0L * m1 * ex2 + 2.0L * m1 * m1 * m1;
        const long double mu4 =
            ex4 - 4.0L * m1 * ex3 + 6.0L * m1 * m1 * ex2 - 3.0L * m1 * m1 * m1 * m1;

        long double lag1 = 0.0L;
        if (count_ > 1) {
            const long double pairs = static_cast<long double>(count_ - 1);
            const long double cov = pairs * pair_sum_xy_ - pair_sum_x_ * pair_sum_y_;
            const long double var_x = pairs * pair_sum_xx_ - pair_sum_x_ * pair_sum_x_;
            const long double var_y = pairs * pair_sum_yy_ - pair_sum_y_ * pair_sum_y_;
            lag1 = cov / std::sqrt(var_x * var_y);
        }

        return SampleSummary{
            .mean = static_cast<double>(m1),
            .variance = static_cast<double>(mu2),
            .skewness = static_cast<double>(mu3 / std::pow(mu2, 1.5L)),
            .excess_kurtosis = static_cast<double>(mu4 / (mu2 * mu2) - 3.0L),
            .lag1_corr = static_cast<double>(lag1),
            .ks_d = with_ks ? ks_statistic(ks_samples) : 0.0,
            .tail1 = static_cast<double>(tail1_) / static_cast<double>(count_),
            .tail4 = static_cast<double>(tail4_) / static_cast<double>(count_),
            .tail8 = static_cast<double>(tail8_) / static_cast<double>(count_),
            .tail12 = static_cast<double>(tail12_) / static_cast<double>(count_),
        };
    }

private:
    static auto exponential_cdf(double x) -> double {
        return x <= 0.0 ? 0.0 : 1.0 - std::exp(-x);
    }

    static auto ks_statistic(std::vector<double> samples) -> double {
        std::sort(samples.begin(), samples.end());
        const double n = static_cast<double>(samples.size());
        double d = 0.0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double fi = exponential_cdf(samples[i]);
            const double lo = static_cast<double>(i) / n;
            const double hi = static_cast<double>(i + 1) / n;
            d = std::max(d, std::fabs(fi - lo));
            d = std::max(d, std::fabs(hi - fi));
        }
        return d;
    }

    std::uint64_t count_ = 0;
    long double sum1_ = 0.0L;
    long double sum2_ = 0.0L;
    long double sum3_ = 0.0L;
    long double sum4_ = 0.0L;
    std::uint64_t tail1_ = 0;
    std::uint64_t tail4_ = 0;
    std::uint64_t tail8_ = 0;
    std::uint64_t tail12_ = 0;
    bool has_prev_ = false;
    long double prev_ = 0.0L;
    long double pair_sum_x_ = 0.0L;
    long double pair_sum_y_ = 0.0L;
    long double pair_sum_xx_ = 0.0L;
    long double pair_sum_yy_ = 0.0L;
    long double pair_sum_xy_ = 0.0L;
};

template <typename Factory>
auto run_method(std::string name, Factory make_fill) -> MethodResult {
    StreamingAccumulator large_acc;
    std::vector<double> chunk(kChunkSize);
    auto large_fill = make_fill();
    std::size_t remaining = kLargeSampleCount;
    while (remaining != 0) {
        const std::size_t batch = std::min(remaining, chunk.size());
        large_fill(chunk.data(), batch);
        for (std::size_t i = 0; i < batch; ++i)
            large_acc.push(chunk[i]);
        remaining -= batch;
    }

    std::vector<double> ks_samples(kKsSampleCount);
    auto ks_fill = make_fill();
    ks_fill(ks_samples.data(), ks_samples.size());
    StreamingAccumulator ks_acc;
    for (double x : ks_samples)
        ks_acc.push(x);

    return MethodResult{
        .name = std::move(name),
        .large = large_acc.finalize(false, ks_samples),
        .ks = ks_acc.finalize(true, ks_samples),
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
    std::cout << "  large tails: P(x>1)=" << result.large.tail1
              << " P(x>4)=" << result.large.tail4
              << " P(x>8)=" << result.large.tail8
              << " P(x>12)=" << result.large.tail12 << '\n';
    std::cout << "  ks sample: D=" << result.ks.ks_d
              << " lag1=" << result.ks.lag1_corr << '\n';
}

auto check_distribution(const MethodResult& result) -> int {
    int failures = 0;
    constexpr double ref_tail1 = 3.6787944117144233e-01;
    constexpr double ref_tail4 = 1.8315638888734179e-02;
    constexpr double ref_tail8 = 3.3546262790251185e-04;
    constexpr double ref_tail12 = 6.14421235332821e-06;

    std::cout << "\nChecking " << result.name << '\n';
    failures += check_near("large mean", result.large.mean, 1.0, 2e-3);
    failures += check_near("large variance", result.large.variance, 1.0, 6e-3);
    failures += check_near("large skewness", result.large.skewness, 2.0, 4e-2);
    failures += check_near("large excess kurtosis", result.large.excess_kurtosis, 6.0, 2e-1);
    failures += check_near("tail x>1", result.large.tail1, ref_tail1, 3e-4);
    failures += check_near("tail x>4", result.large.tail4, ref_tail4, 8e-5);
    failures += check_near("tail x>8", result.large.tail8, ref_tail8, 2e-5);
    failures += check_near("tail x>12", result.large.tail12, ref_tail12, 2e-6);
    failures += check_abs_le("ks lag1 corr", result.ks.lag1_corr, 5e-3);
    failures += check_abs_le("ks D", result.ks.ks_d, 2e-3);
    return failures;
}

auto check_against_exact(const MethodResult& exact,
                         const MethodResult& approx) -> int {
    int failures = 0;
    std::cout << "\nComparing " << approx.name << " against " << exact.name << '\n';
    failures += check_near("mean vs exact", approx.large.mean, exact.large.mean, 5e-5);
    failures += check_near("variance vs exact", approx.large.variance, exact.large.variance, 3e-4);
    failures += check_near("tail1 vs exact", approx.large.tail1, exact.large.tail1, 3e-5);
    failures += check_near("tail4 vs exact", approx.large.tail4, exact.large.tail4, 2e-5);
    failures += check_near("tail8 vs exact", approx.large.tail8, exact.large.tail8, 3e-6);
    failures += check_near("tail12 vs exact", approx.large.tail12, exact.large.tail12, 1e-6);
    failures += check_near("ks D vs exact", approx.ks.ks_d, exact.ks.ks_d, 5e-4);
    return failures;
}

#ifdef __AVX2__
struct ExponentialX8Stream {
    enum class Mode {
        exact,
        fastlog,
    };

    __m256i a0;
    __m256i a1;
    __m256i a2;
    __m256i a3;
    __m256i b0;
    __m256i b1;
    __m256i b2;
    __m256i b3;
    Mode mode;

    explicit ExponentialX8Stream(std::uint64_t seed, Mode mode_in) noexcept : mode(mode_in) {
        alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
        alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
        zorro::detail::seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);
        a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
        a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
        a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
        a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
        b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
        b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
        b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
        b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));
    }

    void fill(double* out, std::size_t count) noexcept {
        const __m256d one = _mm256_set1_pd(1.0);
        const __m256d neg_one = _mm256_set1_pd(-1.0);
        std::size_t i = 0;
        while (i + 8 <= count) {
            const __m256d ua = _mm256_sub_pd(
                one, zorro::detail::u64_to_uniform01_avx2(zorro::detail::next_x4_avx2(a0, a1, a2, a3)));
            const __m256d ub = _mm256_sub_pd(
                one, zorro::detail::u64_to_uniform01_avx2(zorro::detail::next_x4_avx2(b0, b1, b2, b3)));
            if (mode == Mode::exact) {
                _mm256_storeu_pd(out + i, _mm256_mul_pd(neg_one, _ZGVdN4v_log(ua)));
                _mm256_storeu_pd(out + i + 4, _mm256_mul_pd(neg_one, _ZGVdN4v_log(ub)));
            } else {
                _mm256_storeu_pd(out + i, zorro::detail::fast_neglog01_avx2(ua));
                _mm256_storeu_pd(out + i + 4, zorro::detail::fast_neglog01_avx2(ub));
            }
            i += 8;
        }
        if (i < count) {
            alignas(32) double tail[8];
            fill(tail, 8);
            for (std::size_t lane = 0; i < count; ++lane, ++i)
                out[i] = tail[lane];
        }
    }
};

auto compare_exact_vs_fastlog() -> PairwiseErrorSummary {
    ExponentialX8Stream exact(kSeed, ExponentialX8Stream::Mode::exact);
    ExponentialX8Stream fastlog(kSeed, ExponentialX8Stream::Mode::fastlog);
    std::vector<double> exact_chunk(kChunkSize);
    std::vector<double> fastlog_chunk(kChunkSize);

    long double sum_abs = 0.0L;
    long double sum_sq = 0.0L;
    double max_abs = 0.0;
    std::size_t remaining = kPairwiseSampleCount;
    while (remaining != 0) {
        const std::size_t batch = std::min(remaining, exact_chunk.size());
        exact.fill(exact_chunk.data(), batch);
        fastlog.fill(fastlog_chunk.data(), batch);
        for (std::size_t i = 0; i < batch; ++i) {
            const double diff = std::abs(exact_chunk[i] - fastlog_chunk[i]);
            sum_abs += diff;
            sum_sq += static_cast<long double>(diff) * diff;
            max_abs = std::max(max_abs, diff);
        }
        remaining -= batch;
    }

    return PairwiseErrorSummary{
        .mean_abs = static_cast<double>(sum_abs / static_cast<long double>(kPairwiseSampleCount)),
        .rmse = static_cast<double>(std::sqrt(sum_sq / static_cast<long double>(kPairwiseSampleCount))),
        .max_abs = max_abs,
    };
}

void print_pairwise(const PairwiseErrorSummary& summary) {
    std::cout << "exact vs fastlog pairwise error\n";
    std::cout << "  mean abs=" << summary.mean_abs
              << " rmse=" << summary.rmse
              << " max abs=" << summary.max_abs << '\n';
}

auto check_pairwise(const PairwiseErrorSummary& summary) -> int {
    int failures = 0;
    std::cout << "\nChecking exact vs fastlog pairwise error\n";
    failures += check_abs_le("mean abs error", summary.mean_abs, 5e-14);
    failures += check_abs_le("rmse", summary.rmse, 5e-14);
    failures += check_abs_le("max abs error", summary.max_abs, 5e-12);
    return failures;
}
#endif

}  // namespace

int main() {
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "Exponential transform quality tests\n";
    std::cout << "  large samples: " << kLargeSampleCount << '\n';
    std::cout << "  chunk size:    " << kChunkSize << '\n';
    std::cout << "  ks samples:    " << kKsSampleCount << '\n';
    std::cout << "  pairwise:      " << kPairwiseSampleCount << '\n';
    std::cout << "  seed:          0x" << std::hex << kSeed << std::dec << '\n';

#ifndef __AVX2__
    std::cout << "AVX2 not enabled; skipping exponential fastlog checks.\n";
    return 0;
#else
    auto public_result = run_method(
        "public zorro::Rng::fill_exponential",
        [] {
            return [rng = zorro::Rng(kSeed)](double* out, std::size_t count) mutable {
                rng.fill_exponential(out, count);
            };
        });

    auto exact_result = run_method(
        "x8 AVX2 exact -log(1-u)",
        [] {
            return [stream = ExponentialX8Stream(kSeed, ExponentialX8Stream::Mode::exact)](
                       double* out, std::size_t count) mutable {
                stream.fill(out, count);
            };
        });

    auto fastlog_result = run_method(
        "x8 AVX2 fastlog -log(1-u)",
        [] {
            return [stream = ExponentialX8Stream(kSeed, ExponentialX8Stream::Mode::fastlog)](
                       double* out, std::size_t count) mutable {
                stream.fill(out, count);
            };
        });

    const PairwiseErrorSummary pairwise = compare_exact_vs_fastlog();

    print_result(public_result);
    print_result(exact_result);
    print_result(fastlog_result);
    print_pairwise(pairwise);

    int failures = 0;
    failures += check_distribution(public_result);
    failures += check_distribution(exact_result);
    failures += check_distribution(fastlog_result);
    failures += check_against_exact(exact_result, fastlog_result);
    failures += check_pairwise(pairwise);

    if (failures != 0) {
        std::cout << "\n" << failures << " exponential-transform check(s) FAILED.\n";
        return 1;
    }

    std::cout << "\nAll exponential-transform checks passed.\n";
    return 0;
#endif
}
