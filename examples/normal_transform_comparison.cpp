#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "benchmarks/zorro_benchmark_kernels.hpp"
#include "zorro/zorro.hpp"

namespace {

#ifdef __AVX2__
extern "C" __m256d _ZGVdN4v_log(__m256d) noexcept;
extern "C" __m256d _ZGVdN4v_sin(__m256d) noexcept;
extern "C" __m256d _ZGVdN4v_cos(__m256d) noexcept;

inline auto floatbitmask64_avx2(__m256i bits) noexcept -> __m256d {
    const __m256i exponent = _mm256_set1_epi64x(0x3ff0000000000000ULL);
    const __m256i mantissa = _mm256_and_si256(
        bits, _mm256_set1_epi64x(static_cast<std::int64_t>(0x000fffffffffffffULL)));
    return _mm256_castsi256_pd(_mm256_or_si256(mantissa, exponent));
}

inline auto approx_sin8_avx2(__m256d x) noexcept -> __m256d {
    const __m256d c0 = _mm256_set1_pd(2.2214414690791831);
    const __m256d c1 = _mm256_set1_pd(-0.9135311874994298);
    const __m256d c2 = _mm256_set1_pd(0.11270239285845876);
    const __m256d c3 = _mm256_set1_pd(-0.006621000193853499);
    const __m256d c4 = _mm256_set1_pd(0.00022689809942335572);
    const __m256d c5 = _mm256_set1_pd(-5.089532691384022e-06);
    const __m256d c6 = _mm256_set1_pd(8.049906344315649e-08);
    const __m256d c7 = _mm256_set1_pd(-9.453796623737637e-10);
    const __m256d c8 = _mm256_set1_pd(8.320735422342538e-12);
    const __m256d x2 = _mm256_mul_pd(x, x);

    __m256d p = c8;
    p = _mm256_add_pd(_mm256_mul_pd(p, x2), c7);
    p = _mm256_add_pd(_mm256_mul_pd(p, x2), c6);
    p = _mm256_add_pd(_mm256_mul_pd(p, x2), c5);
    p = _mm256_add_pd(_mm256_mul_pd(p, x2), c4);
    p = _mm256_add_pd(_mm256_mul_pd(p, x2), c3);
    p = _mm256_add_pd(_mm256_mul_pd(p, x2), c2);
    p = _mm256_add_pd(_mm256_mul_pd(p, x2), c1);
    p = _mm256_add_pd(_mm256_mul_pd(p, x2), c0);
    return _mm256_mul_pd(p, x);
}

inline auto apply_sign_from_bits_avx2(__m256d magnitude, __m256i sign_source_bits) noexcept
    -> __m256d {
    const __m256i sign_mask =
        _mm256_set1_epi64x(static_cast<std::int64_t>(0x8000000000000000ULL));
    return _mm256_castsi256_pd(_mm256_xor_si256(
        _mm256_castpd_si256(magnitude), _mm256_and_si256(sign_source_bits, sign_mask)));
}

inline void randsincos_approx_avx2(__m256i u, __m256d& s, __m256d& c) noexcept {
    const __m256d r = floatbitmask64_avx2(u);
    const __m256d one_open = _mm256_set1_pd(0.9999999999999999);
    const __m256d sub_one_open = _mm256_set1_pd(1.9999999999999998);
    const __m256d sininput = _mm256_sub_pd(r, one_open);
    const __m256d cosinput =
        _mm256_sub_pd(sub_one_open, _mm256_mul_pd(one_open, r));

    s = apply_sign_from_bits_avx2(approx_sin8_avx2(sininput), u);
    c = apply_sign_from_bits_avx2(approx_sin8_avx2(cosinput), _mm256_slli_epi64(u, 1));
}

inline auto log2_3q_avx2(__m256d v, __m256d e) noexcept -> __m256d {
    const __m256d c0 = _mm256_set1_pd(0.22119417504560815);
    const __m256d c1 = _mm256_set1_pd(0.22007686931522777);
    const __m256d c2 = _mm256_set1_pd(0.26237080574885147);
    const __m256d c3 = _mm256_set1_pd(0.32059774779444955);
    const __m256d c4 = _mm256_set1_pd(0.41219859454853247);
    const __m256d c5 = _mm256_set1_pd(0.5770780162997059);
    const __m256d c6 = _mm256_set1_pd(0.9617966939260809);
    const __m256d scale = _mm256_set1_pd(2.8853900817779268);

    const __m256d m1 = _mm256_mul_pd(v, v);
    const __m256d fma1 = _mm256_add_pd(_mm256_mul_pd(m1, c0), c1);
    const __m256d fma2 = _mm256_add_pd(_mm256_mul_pd(fma1, m1), c2);
    const __m256d fma3 = _mm256_add_pd(_mm256_mul_pd(fma2, m1), c3);
    const __m256d fma4 = _mm256_add_pd(_mm256_mul_pd(fma3, m1), c4);
    const __m256d fma5 = _mm256_add_pd(_mm256_mul_pd(fma4, m1), c5);
    const __m256d fma6 = _mm256_add_pd(_mm256_mul_pd(fma5, m1), c6);

    const __m256d m2 = _mm256_mul_pd(v, scale);
    const __m256d a1 = _mm256_add_pd(e, m2);
    const __m256d s1 = _mm256_sub_pd(e, a1);
    const __m256d a2 = _mm256_add_pd(m2, s1);
    const __m256d a3 = a2;
    const __m256d m3 = _mm256_mul_pd(v, m1);
    const __m256d a4 = _mm256_add_pd(a1, a3);
    return _mm256_add_pd(_mm256_mul_pd(fma6, m3), a4);
}

inline auto fast_neglog01_avx2(__m256d u) noexcept -> __m256d {
    const __m256i bits = _mm256_castpd_si256(u);
    const __m256i exponent_mask = _mm256_set1_epi64x(0x7ff0000000000000ULL);
    const __m256i mantissa_mask = _mm256_set1_epi64x(0x000fffffffffffffULL);
    const __m256i exponent_bits = _mm256_set1_epi64x(0x3ff0000000000000ULL);
    const __m256d four_thirds = _mm256_set1_pd(1.3333333333333333);
    const __m256d neg_ln2 = _mm256_set1_pd(-0.6931471805599453);

    alignas(32) std::uint64_t exp_words[4];
    alignas(32) double exp_doubles[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(exp_words),
                       _mm256_srli_epi64(_mm256_and_si256(bits, exponent_mask), 52));
    for (int lane = 0; lane < 4; ++lane) {
        exp_doubles[lane] =
            static_cast<double>(static_cast<std::int64_t>(exp_words[lane]) - 1023) +
            0.4150374992788438;
    }

    const __m256d mantissa = _mm256_castsi256_pd(
        _mm256_or_si256(_mm256_and_si256(bits, mantissa_mask), exponent_bits));
    const __m256d v = _mm256_div_pd(_mm256_sub_pd(mantissa, four_thirds),
                                    _mm256_add_pd(mantissa, four_thirds));
    const __m256d log2_u = log2_3q_avx2(v, _mm256_load_pd(exp_doubles));
    return _mm256_max_pd(_mm256_mul_pd(neg_ln2, log2_u), _mm256_setzero_pd());
}

struct BoxMullerX8Stream {
    enum class Mode {
        exact,
        fastlog,
        approxsincos,
        fullapprox,
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

    explicit BoxMullerX8Stream(std::uint64_t seed, Mode mode_in) noexcept : mode(mode_in) {
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
        const __m256d neg2 = _mm256_set1_pd(-2.0);
        std::size_t i = 0;
        while (i + 16 <= count) {
            emit16(out + i, one, neg2);
            i += 16;
        }
        if (i < count) {
            alignas(32) double tmp[16];
            emit16(tmp, one, neg2);
            for (std::size_t j = 0; i < count; ++i, ++j)
                out[i] = tmp[j];
        }
    }

    void emit16(double* out, __m256d one, __m256d neg2) noexcept {
        const __m256i u1a = zorro::detail::next_x4_avx2(a0, a1, a2, a3);
        const __m256i u2a = zorro::detail::next_x4_avx2(a0, a1, a2, a3);
        const __m256i u1b = zorro::detail::next_x4_avx2(b0, b1, b2, b3);
        const __m256i u2b = zorro::detail::next_x4_avx2(b0, b1, b2, b3);

        __m256d s_a, c_a, s_b, c_b;
        __m256d radius_a, radius_b;

        if (mode == Mode::approxsincos || mode == Mode::fullapprox) {
            randsincos_approx_avx2(u1a, s_a, c_a);
            randsincos_approx_avx2(u1b, s_b, c_b);
        } else {
            const __m256d theta_a =
                _mm256_mul_pd(_mm256_set1_pd(6.2831853071795864769252867665590058),
                              zorro::detail::u64_to_uniform01_avx2(u1a));
            const __m256d theta_b =
                _mm256_mul_pd(_mm256_set1_pd(6.2831853071795864769252867665590058),
                              zorro::detail::u64_to_uniform01_avx2(u1b));
            s_a = _ZGVdN4v_sin(theta_a);
            c_a = _ZGVdN4v_cos(theta_a);
            s_b = _ZGVdN4v_sin(theta_b);
            c_b = _ZGVdN4v_cos(theta_b);
        }

        const __m256d ur_a = _mm256_sub_pd(one, zorro::detail::u64_to_uniform01_avx2(u2a));
        const __m256d ur_b = _mm256_sub_pd(one, zorro::detail::u64_to_uniform01_avx2(u2b));
        if (mode == Mode::fastlog) {
            radius_a = _mm256_sqrt_pd(_mm256_mul_pd(_mm256_set1_pd(2.0), fast_neglog01_avx2(ur_a)));
            radius_b = _mm256_sqrt_pd(_mm256_mul_pd(_mm256_set1_pd(2.0), fast_neglog01_avx2(ur_b)));
        } else if (mode == Mode::fullapprox) {
            radius_a = _mm256_sqrt_pd(fast_neglog01_avx2(ur_a));
            radius_b = _mm256_sqrt_pd(fast_neglog01_avx2(ur_b));
        } else if (mode == Mode::approxsincos) {
            radius_a = _mm256_sqrt_pd(_mm256_mul_pd(_mm256_set1_pd(-1.0), _ZGVdN4v_log(ur_a)));
            radius_b = _mm256_sqrt_pd(_mm256_mul_pd(_mm256_set1_pd(-1.0), _ZGVdN4v_log(ur_b)));
        } else {
            radius_a = _mm256_sqrt_pd(_mm256_mul_pd(neg2, _ZGVdN4v_log(ur_a)));
            radius_b = _mm256_sqrt_pd(_mm256_mul_pd(neg2, _ZGVdN4v_log(ur_b)));
        }

        _mm256_storeu_pd(out + 0, _mm256_mul_pd(radius_a, s_a));
        _mm256_storeu_pd(out + 4, _mm256_mul_pd(radius_a, c_a));
        _mm256_storeu_pd(out + 8, _mm256_mul_pd(radius_b, s_b));
        _mm256_storeu_pd(out + 12, _mm256_mul_pd(radius_b, c_b));
    }
};
#endif

using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;

constexpr std::uint64_t kDefaultSeed = 0x123456789abcdef0ULL;
constexpr std::size_t kDefaultChunkSize = 1u << 20;
constexpr std::size_t kMaxStoredErrorSamples = 1u << 20;

struct Options {
    std::size_t samples = 1u << 20;
    std::size_t chunk_size = kDefaultChunkSize;
    int repetitions = 9;
    std::uint64_t seed = kDefaultSeed;
};

struct BenchmarkSummary {
    double best_ms = 0.0;
    double median_ms = 0.0;
    double mean_ms = 0.0;
    double samples_per_second = 0.0;
};

struct DistributionSummary {
    double mean = 0.0;
    double stddev = 0.0;
    double abs_tail_3 = 0.0;
    double abs_tail_4 = 0.0;
    double abs_tail_5 = 0.0;
};

struct PairwiseErrorSummary {
    double mean_abs = 0.0;
    double rmse = 0.0;
    double sampled_p99_abs = 0.0;
    double max_abs = 0.0;
    std::size_t sampled_errors = 0;
};

struct DistributionAccumulator {
    std::uint64_t count = 0;
    double sum = 0.0;
    double sum_sq = 0.0;
    std::uint64_t tail3 = 0;
    std::uint64_t tail4 = 0;
    std::uint64_t tail5 = 0;

    void add(std::span<const double> samples) {
        for (double x : samples) {
            sum += x;
            sum_sq += x * x;
            const double ax = std::abs(x);
            tail3 += ax > 3.0;
            tail4 += ax > 4.0;
            tail5 += ax > 5.0;
        }
        count += samples.size();
    }

    auto finish() const -> DistributionSummary {
        const double n = static_cast<double>(count);
        const double mean = sum / n;
        const double variance = std::max(0.0, sum_sq / n - mean * mean);
        return DistributionSummary{
            .mean = mean,
            .stddev = std::sqrt(variance),
            .abs_tail_3 = static_cast<double>(tail3) / n,
            .abs_tail_4 = static_cast<double>(tail4) / n,
            .abs_tail_5 = static_cast<double>(tail5) / n,
        };
    }
};

struct PairwiseErrorAccumulator {
    std::uint64_t count = 0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double max_abs = 0.0;
    std::size_t sample_stride = 1;
    std::vector<double> sampled_abs_errors;

    explicit PairwiseErrorAccumulator(std::size_t total_samples) {
        sample_stride = std::max<std::size_t>(1, total_samples / kMaxStoredErrorSamples);
        sampled_abs_errors.reserve(std::min(total_samples, kMaxStoredErrorSamples));
    }

    void add(std::span<const double> exact, std::span<const double> approx) {
        for (std::size_t i = 0; i < exact.size(); ++i) {
            const double err = approx[i] - exact[i];
            const double abs_err = std::abs(err);
            sum_abs += abs_err;
            sum_sq += err * err;
            max_abs = std::max(max_abs, abs_err);
            if ((count % sample_stride) == 0 && sampled_abs_errors.size() < kMaxStoredErrorSamples)
                sampled_abs_errors.push_back(abs_err);
            ++count;
        }
    }

    auto finish() -> PairwiseErrorSummary {
        double sampled_p99_abs = 0.0;
        if (!sampled_abs_errors.empty()) {
            const std::size_t p99_index =
                (sampled_abs_errors.size() * 99) / 100;
            std::nth_element(sampled_abs_errors.begin(),
                             sampled_abs_errors.begin() + p99_index,
                             sampled_abs_errors.end());
            sampled_p99_abs = sampled_abs_errors[p99_index];
        }

        const double n = static_cast<double>(count);
        return PairwiseErrorSummary{
            .mean_abs = sum_abs / n,
            .rmse = std::sqrt(sum_sq / n),
            .sampled_p99_abs = sampled_p99_abs,
            .max_abs = max_abs,
            .sampled_errors = sampled_abs_errors.size(),
        };
    }
};

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0
        << " [--samples N] [--chunk-size N] [--repetitions N] [--seed HEX_OR_DEC]\n"
        << '\n'
        << "Compare exact Box-Muller against Julia-inspired approximations for the\n"
        << "radius (-log(u)), the angle (sin/cos), and both combined.\n"
        << "Quality analysis streams through chunks so large sample counts do not\n"
        << "require storing all outputs in memory.\n";
}

auto parse_u64(std::string_view text) -> std::uint64_t {
    return std::stoull(std::string(text), nullptr, 0);
}

auto parse_options(int argc, char** argv) -> Options {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto require_value = [&](const char* name) -> std::string_view {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << '\n';
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--samples") {
            options.samples = parse_u64(require_value("--samples"));
            continue;
        }
        if (arg == "--chunk-size") {
            options.chunk_size = parse_u64(require_value("--chunk-size"));
            continue;
        }
        if (arg == "--repetitions") {
            options.repetitions = static_cast<int>(parse_u64(require_value("--repetitions")));
            continue;
        }
        if (arg == "--seed") {
            options.seed = parse_u64(require_value("--seed"));
            continue;
        }

        std::cerr << "Unknown argument: " << arg << '\n';
        print_usage(argv[0]);
        std::exit(1);
    }

    if (options.chunk_size == 0) {
        std::cerr << "--chunk-size must be positive\n";
        std::exit(1);
    }
    return options;
}

template <typename FillFn>
auto benchmark_fill(FillFn&& fill, std::span<double> scratch,
                    int repetitions) -> BenchmarkSummary {
    constexpr int kWarmups = 2;
    for (int i = 0; i < kWarmups; ++i)
        fill(scratch.data(), scratch.size());

    std::vector<double> times_ms;
    times_ms.reserve(static_cast<std::size_t>(repetitions));
    for (int i = 0; i < repetitions; ++i) {
        const auto start = Clock::now();
        fill(scratch.data(), scratch.size());
        const auto stop = Clock::now();
        times_ms.push_back(Milliseconds(stop - start).count());
    }

    std::vector<double> sorted = times_ms;
    std::sort(sorted.begin(), sorted.end());
    const double best_ms = sorted.front();
    const double median_ms = sorted[sorted.size() / 2];
    const double mean_ms =
        std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
    return BenchmarkSummary{
        .best_ms = best_ms,
        .median_ms = median_ms,
        .mean_ms = mean_ms,
        .samples_per_second = static_cast<double>(scratch.size()) / (best_ms / 1000.0),
    };
}

template <typename FillFn>
auto stream_distribution(FillFn&& fill, std::span<double> chunk_buffer,
                         std::size_t total_samples) -> DistributionSummary {
    DistributionAccumulator acc;
    std::size_t remaining = total_samples;
    while (remaining > 0) {
        const std::size_t chunk = std::min(remaining, chunk_buffer.size());
        fill(chunk_buffer.data(), chunk);
        acc.add(chunk_buffer.first(chunk));
        remaining -= chunk;
    }
    return acc.finish();
}

template <typename ExactFillFn, typename ApproxFillFn>
auto stream_pairwise_error(ExactFillFn&& fill_exact, ApproxFillFn&& fill_approx,
                           std::span<double> exact_buffer,
                           std::span<double> approx_buffer,
                           std::size_t total_samples) -> PairwiseErrorSummary {
    PairwiseErrorAccumulator acc(total_samples);
    std::size_t remaining = total_samples;
    while (remaining > 0) {
        const std::size_t chunk = std::min({remaining, exact_buffer.size(), approx_buffer.size()});
        fill_exact(exact_buffer.data(), chunk);
        fill_approx(approx_buffer.data(), chunk);
        acc.add(exact_buffer.first(chunk), approx_buffer.first(chunk));
        remaining -= chunk;
    }
    return acc.finish();
}

void print_benchmark_row(const std::string& name, const BenchmarkSummary& summary,
                         std::size_t samples) {
    const double ns_per_sample =
        (summary.best_ms * 1'000'000.0) / static_cast<double>(samples);
    std::cout << std::left << std::setw(34) << name
              << std::right << std::setw(12) << std::fixed << std::setprecision(3)
              << summary.best_ms
              << std::setw(12) << summary.median_ms
              << std::setw(12) << summary.mean_ms
              << std::setw(16) << (summary.samples_per_second / 1'000'000.0)
              << std::setw(16) << ns_per_sample
              << '\n';
}

void print_distribution_row(const std::string& name,
                            const DistributionSummary& summary) {
    std::cout << std::left << std::setw(34) << name
              << std::right << std::setw(14) << std::scientific << std::setprecision(6)
              << summary.mean
              << std::setw(14) << summary.stddev
              << std::setw(14) << summary.abs_tail_3
              << std::setw(14) << summary.abs_tail_4
              << std::setw(14) << summary.abs_tail_5
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    const std::size_t benchmark_samples = std::min(options.samples, options.chunk_size);

    std::vector<double> benchmark_buffer(benchmark_samples);
    std::vector<double> analysis_a(options.chunk_size);
    std::vector<double> analysis_b(options.chunk_size);

    auto fill_exact_box = [&](double* out, std::size_t count) {
        zorro_bench::fill_xoshiro256pp_x8_normal_box_muller_avx2(options.seed, out, count);
    };
    auto fill_fastlog_box = [&](double* out, std::size_t count) {
        zorro_bench::fill_xoshiro256pp_x8_normal_box_muller_avx2_fastlog(
            options.seed, out, count);
    };
    auto fill_approxsincos_box = [&](double* out, std::size_t count) {
        zorro_bench::fill_xoshiro256pp_x8_normal_box_muller_avx2_approxsincos(
            options.seed, out, count);
    };
    auto fill_fullapprox_box = [&](double* out, std::size_t count) {
        zorro_bench::fill_xoshiro256pp_x8_normal_box_muller_avx2_fullapprox(
            options.seed, out, count);
    };
    auto fill_vecpolar = [&](double* out, std::size_t count) {
        zorro_bench::fill_xoshiro256pp_x8_normal_vecpolar_avx2(options.seed, out, count);
    };
    auto fill_public = [&](double* out, std::size_t count) {
        zorro::Rng rng(options.seed);
        rng.fill_normal(out, count);
    };

    const BenchmarkSummary exact_bench =
        benchmark_fill(fill_exact_box, benchmark_buffer, options.repetitions);
    const BenchmarkSummary fastlog_bench =
        benchmark_fill(fill_fastlog_box, benchmark_buffer, options.repetitions);
    const BenchmarkSummary approxsincos_bench =
        benchmark_fill(fill_approxsincos_box, benchmark_buffer, options.repetitions);
    const BenchmarkSummary fullapprox_bench =
        benchmark_fill(fill_fullapprox_box, benchmark_buffer, options.repetitions);
    const BenchmarkSummary vecpolar_bench =
        benchmark_fill(fill_vecpolar, benchmark_buffer, options.repetitions);
    const BenchmarkSummary public_bench =
        benchmark_fill(fill_public, benchmark_buffer, options.repetitions);

#ifdef __AVX2__
    BoxMullerX8Stream exact_stream(options.seed, BoxMullerX8Stream::Mode::exact);
    BoxMullerX8Stream fastlog_stream(options.seed, BoxMullerX8Stream::Mode::fastlog);
    BoxMullerX8Stream approxsincos_stream(options.seed, BoxMullerX8Stream::Mode::approxsincos);
    BoxMullerX8Stream fullapprox_stream(options.seed, BoxMullerX8Stream::Mode::fullapprox);

    BoxMullerX8Stream exact_for_fastlog(options.seed, BoxMullerX8Stream::Mode::exact);
    BoxMullerX8Stream fastlog_for_error(options.seed, BoxMullerX8Stream::Mode::fastlog);
    BoxMullerX8Stream exact_for_approxsincos(options.seed, BoxMullerX8Stream::Mode::exact);
    BoxMullerX8Stream approxsincos_for_error(options.seed, BoxMullerX8Stream::Mode::approxsincos);
    BoxMullerX8Stream exact_for_fullapprox(options.seed, BoxMullerX8Stream::Mode::exact);
    BoxMullerX8Stream fullapprox_for_error(options.seed, BoxMullerX8Stream::Mode::fullapprox);
#endif
    zorro::Rng public_rng(options.seed);

    const DistributionSummary exact_summary =
        stream_distribution([&](double* out, std::size_t count) {
#ifdef __AVX2__
            exact_stream.fill(out, count);
#else
            fill_exact_box(out, count);
#endif
        }, analysis_a, options.samples);
    const DistributionSummary fastlog_summary =
        stream_distribution([&](double* out, std::size_t count) {
#ifdef __AVX2__
            fastlog_stream.fill(out, count);
#else
            fill_fastlog_box(out, count);
#endif
        }, analysis_a, options.samples);
    const DistributionSummary approxsincos_summary =
        stream_distribution([&](double* out, std::size_t count) {
#ifdef __AVX2__
            approxsincos_stream.fill(out, count);
#else
            fill_approxsincos_box(out, count);
#endif
        }, analysis_a, options.samples);
    const DistributionSummary fullapprox_summary =
        stream_distribution([&](double* out, std::size_t count) {
#ifdef __AVX2__
            fullapprox_stream.fill(out, count);
#else
            fill_fullapprox_box(out, count);
#endif
        }, analysis_a, options.samples);
    const DistributionSummary public_summary =
        stream_distribution([&](double* out, std::size_t count) {
            public_rng.fill_normal(out, count);
        }, analysis_a, options.samples);

    const PairwiseErrorSummary fastlog_error =
        stream_pairwise_error(
            [&](double* out, std::size_t count) {
#ifdef __AVX2__
                exact_for_fastlog.fill(out, count);
#else
                fill_exact_box(out, count);
#endif
            },
            [&](double* out, std::size_t count) {
#ifdef __AVX2__
                fastlog_for_error.fill(out, count);
#else
                fill_fastlog_box(out, count);
#endif
            },
            analysis_a, analysis_b, options.samples);
    const PairwiseErrorSummary approxsincos_error =
        stream_pairwise_error(
            [&](double* out, std::size_t count) {
#ifdef __AVX2__
                exact_for_approxsincos.fill(out, count);
#else
                fill_exact_box(out, count);
#endif
            },
            [&](double* out, std::size_t count) {
#ifdef __AVX2__
                approxsincos_for_error.fill(out, count);
#else
                fill_approxsincos_box(out, count);
#endif
            },
            analysis_a, analysis_b, options.samples);
    const PairwiseErrorSummary fullapprox_error =
        stream_pairwise_error(
            [&](double* out, std::size_t count) {
#ifdef __AVX2__
                exact_for_fullapprox.fill(out, count);
#else
                fill_exact_box(out, count);
#endif
            },
            [&](double* out, std::size_t count) {
#ifdef __AVX2__
                fullapprox_for_error.fill(out, count);
#else
                fill_fullapprox_box(out, count);
#endif
            },
            analysis_a, analysis_b, options.samples);

    std::cout << "Normal transform comparison\n";
    std::cout << "  samples:             " << options.samples << '\n';
    std::cout << "  chunk size:          " << options.chunk_size << '\n';
    std::cout << "  benchmark size:      " << benchmark_samples << '\n';
    std::cout << "  repetitions:         " << options.repetitions << '\n';
    std::cout << "  seed:                0x" << std::hex << options.seed << std::dec << '\n';
    std::cout << '\n';

    std::cout << "Throughput\n";
    std::cout << std::left << std::setw(34) << "method"
              << std::right << std::setw(12) << "best ms"
              << std::setw(12) << "median"
              << std::setw(12) << "mean"
              << std::setw(16) << "M samples/s"
              << std::setw(16) << "ns/sample"
              << '\n';
    std::cout << std::string(102, '-') << '\n';
    print_benchmark_row("public zorro::Rng::fill_normal", public_bench, benchmark_samples);
    print_benchmark_row("x8 AVX2 box-muller exact", exact_bench, benchmark_samples);
    print_benchmark_row("x8 AVX2 box-muller fastlog", fastlog_bench, benchmark_samples);
    print_benchmark_row("x8 AVX2 box-muller approxsincos",
                        approxsincos_bench, benchmark_samples);
    print_benchmark_row("x8 AVX2 box-muller fullapprox",
                        fullapprox_bench, benchmark_samples);
    print_benchmark_row("x8 AVX2 vecpolar", vecpolar_bench, benchmark_samples);
    std::cout << '\n';

    std::cout << "Distribution summary\n";
    std::cout << "  reference P(|X| > 3): " << std::scientific << std::setprecision(6)
              << std::erfc(3.0 / std::sqrt(2.0)) << '\n';
    std::cout << "  reference P(|X| > 4): " << std::erfc(4.0 / std::sqrt(2.0)) << '\n';
    std::cout << "  reference P(|X| > 5): " << std::erfc(5.0 / std::sqrt(2.0)) << '\n';
    std::cout << std::left << std::setw(34) << "method"
              << std::right << std::setw(14) << "mean"
              << std::setw(14) << "stddev"
              << std::setw(14) << "P(|x|>3)"
              << std::setw(14) << "P(|x|>4)"
              << std::setw(14) << "P(|x|>5)"
              << '\n';
    std::cout << std::string(90, '-') << '\n';
    print_distribution_row("public zorro::Rng::fill_normal", public_summary);
    print_distribution_row("x8 AVX2 box-muller exact", exact_summary);
    print_distribution_row("x8 AVX2 box-muller fastlog", fastlog_summary);
    print_distribution_row("x8 AVX2 box-muller approxsincos", approxsincos_summary);
    print_distribution_row("x8 AVX2 box-muller fullapprox", fullapprox_summary);
    std::cout << '\n';

    std::cout << "Pairwise approximation error vs exact\n";
    std::cout << "  sampled abs-error count: "
              << fastlog_error.sampled_errors << '\n';
    std::cout << std::left << std::setw(34) << "method"
              << std::right << std::setw(14) << "mean abs"
              << std::setw(14) << "rmse"
              << std::setw(14) << "p99 abs*"
              << std::setw(14) << "max abs"
              << '\n';
    std::cout << std::string(90, '-') << '\n';
    auto print_error_row = [](const std::string& name, const PairwiseErrorSummary& summary) {
        std::cout << std::left << std::setw(34) << name
                  << std::right << std::setw(14) << std::scientific << std::setprecision(6)
                  << summary.mean_abs
                  << std::setw(14) << summary.rmse
                  << std::setw(14) << summary.sampled_p99_abs
                  << std::setw(14) << summary.max_abs
                  << '\n';
    };
    print_error_row("x8 AVX2 box-muller fastlog", fastlog_error);
    print_error_row("x8 AVX2 box-muller approxsincos", approxsincos_error);
    print_error_row("x8 AVX2 box-muller fullapprox", fullapprox_error);
    std::cout << "\n* p99 is computed from a capped sampled subset of absolute errors.\n";

    return 0;
}
