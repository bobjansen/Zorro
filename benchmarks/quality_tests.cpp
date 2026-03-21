#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#include "zorro/zorro.hpp"

namespace {

constexpr std::uint64_t kSeed = 0x123456789abcdef0ULL;
constexpr std::size_t kRawWordCount = 1u << 22;
constexpr std::size_t kPermutationGroups = 1u << 19;
constexpr std::size_t kMatrixCount = 16384;
constexpr std::size_t kDistributionSamples = 1u << 20;

struct RawSuiteResult {
    std::string name;
    double monobit_p;
    double byte_chi2;
    double byte_p;
    double lag1_corr;
    double lag1_corr_p;
    double permutation_chi2;
    double permutation_p;
    double rank_chi2;
    double rank_p;
};

struct ContinuousDistributionResult {
    std::string name;
    double mean;
    double variance;
    double skewness;
    double excess_kurtosis;
    double lag1_corr;
    double lag1_corr_p;
    double ks_d;
    double ks_p;
};

struct BernoulliResult {
    std::string name;
    double mean;
    double mean_p;
    double lag1_corr;
    double lag1_corr_p;
    double runs_p;
};

struct Moments {
    double mean;
    double variance;
    double skewness;
    double excess_kurtosis;
};

auto clamp_probability(double p) -> double {
    return std::clamp(p, 0.0, 1.0);
}

auto normal_two_sided_p(double z) -> double {
    return clamp_probability(std::erfc(std::fabs(z) / std::numbers::sqrt2_v<double>));
}

auto chi_square_upper_tail_p(double chi2, double dof) -> double {
    if (!(chi2 >= 0.0) || !(dof > 0.0))
        return std::numeric_limits<double>::quiet_NaN();
    const double x = std::pow(chi2 / dof, 1.0 / 3.0);
    const double mean = 1.0 - 2.0 / (9.0 * dof);
    const double stddev = std::sqrt(2.0 / (9.0 * dof));
    return 0.5 * std::erfc((x - mean) / (stddev * std::numbers::sqrt2_v<double>));
}

auto ks_pvalue(std::size_t n, double d) -> double {
    if (n == 0)
        return std::numeric_limits<double>::quiet_NaN();
    const double root_n = std::sqrt(static_cast<double>(n));
    const double lambda = (root_n + 0.12 + 0.11 / root_n) * d;
    double sum = 0.0;
    for (int k = 1; k < 100; ++k) {
        const double term =
            2.0 * ((k % 2 == 1) ? 1.0 : -1.0) * std::exp(-2.0 * k * k * lambda * lambda);
        sum += term;
        if (std::fabs(term) < 1e-12)
            break;
    }
    return clamp_probability(sum);
}

auto uniform01(std::uint64_t x) -> double {
    return zorro::bits_to_01(x);
}

auto lag1_correlation(const std::vector<double>& xs) -> double {
    if (xs.size() < 2)
        return std::numeric_limits<double>::quiet_NaN();
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

auto lag1_correlation_pvalue(double corr, std::size_t sample_count) -> double {
    if (sample_count < 2)
        return std::numeric_limits<double>::quiet_NaN();
    const double z = corr * std::sqrt(static_cast<double>(sample_count - 1));
    return normal_two_sided_p(z);
}

auto permutation_bucket(const std::array<std::uint64_t, 4>& values) -> int {
    std::array<int, 4> order = {0, 1, 2, 3};
    std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return values[lhs] < values[rhs];
    });

    constexpr std::array<int, 4> kFactorials = {6, 2, 1, 1};
    int code = 0;
    for (int i = 0; i < 4; ++i) {
        int smaller = 0;
        for (int j = i + 1; j < 4; ++j) {
            if (order[j] < order[i])
                ++smaller;
        }
        code += smaller * kFactorials[i];
    }
    return code;
}

auto matrix_rank_32(std::array<std::uint32_t, 32> rows) -> int {
    int rank = 0;
    for (int bit = 31; bit >= 0; --bit) {
        int pivot = rank;
        while (pivot < 32 && ((rows[pivot] >> bit) & 1u) == 0u)
            ++pivot;
        if (pivot == 32)
            continue;
        std::swap(rows[rank], rows[pivot]);
        for (int row = 0; row < 32; ++row) {
            if (row != rank && ((rows[row] >> bit) & 1u) != 0u)
                rows[row] ^= rows[rank];
        }
        ++rank;
        if (rank == 32)
            break;
    }
    return rank;
}

auto binary_matrix_rank_probability(int rank, int rows, int cols) -> long double {
    if (rank < 0 || rank > rows || rank > cols)
        return 0.0L;
    long double prob = std::exp2l(static_cast<long double>(rank) *
                                      static_cast<long double>(rows + cols - rank) -
                                  static_cast<long double>(rows) * cols);
    for (int i = 0; i < rank; ++i) {
        const long double num_rows = 1.0L - std::exp2l(i - rows);
        const long double num_cols = 1.0L - std::exp2l(i - cols);
        const long double den = 1.0L - std::exp2l(i - rank);
        prob *= (num_rows * num_cols) / den;
    }
    return prob;
}

template <typename Factory>
auto run_raw_suite(std::string name, Factory&& make_source) -> RawSuiteResult {
    std::uint64_t ones = 0;
    std::array<std::uint64_t, 256> byte_counts = {};
    long double sum_x = 0.0L;
    long double sum_y = 0.0L;
    long double sum_xx = 0.0L;
    long double sum_yy = 0.0L;
    long double sum_xy = 0.0L;
    auto source = make_source();
    std::uint64_t previous = source.next();

    for (std::size_t i = 0; i < kRawWordCount; ++i) {
        const std::uint64_t word = (i == 0) ? previous : source.next();
        ones += std::popcount(word);
        for (int byte = 0; byte < 8; ++byte)
            ++byte_counts[(word >> (byte * 8)) & 0xffu];
        if (i > 0) {
            const long double x = uniform01(previous);
            const long double y = uniform01(word);
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_yy += y * y;
            sum_xy += x * y;
        }
        previous = word;
    }

    const long double bit_count =
        static_cast<long double>(kRawWordCount) * 64.0L;
    const long double z = (static_cast<long double>(ones) - bit_count / 2.0L) /
                          std::sqrt(bit_count / 4.0L);
    const double monobit_p = normal_two_sided_p(static_cast<double>(z));

    const double expected_bytes =
        static_cast<double>(kRawWordCount) * 8.0 / 256.0;
    double byte_chi2 = 0.0;
    for (std::uint64_t observed : byte_counts) {
        const double diff = static_cast<double>(observed) - expected_bytes;
        byte_chi2 += diff * diff / expected_bytes;
    }

    const long double n_pairs = static_cast<long double>(kRawWordCount - 1);
    const long double cov = n_pairs * sum_xy - sum_x * sum_y;
    const long double var_x = n_pairs * sum_xx - sum_x * sum_x;
    const long double var_y = n_pairs * sum_yy - sum_y * sum_y;
    const double lag1_corr = static_cast<double>(cov / std::sqrt(var_x * var_y));

    std::array<std::uint64_t, 24> permutation_counts = {};
    {
        auto permutation_source = make_source();
        for (std::size_t i = 0; i < kPermutationGroups; ++i) {
            const std::array<std::uint64_t, 4> values = {
                permutation_source.next(),
                permutation_source.next(),
                permutation_source.next(),
                permutation_source.next(),
            };
            ++permutation_counts[permutation_bucket(values)];
        }
    }

    const double expected_permutation =
        static_cast<double>(kPermutationGroups) / 24.0;
    double permutation_chi2 = 0.0;
    for (std::uint64_t observed : permutation_counts) {
        const double diff = static_cast<double>(observed) - expected_permutation;
        permutation_chi2 += diff * diff / expected_permutation;
    }

    std::array<std::uint64_t, 3> rank_counts = {};
    {
        auto rank_source = make_source();
        for (std::size_t matrix = 0; matrix < kMatrixCount; ++matrix) {
            std::array<std::uint32_t, 32> rows = {};
            for (std::uint32_t& row : rows)
                row = static_cast<std::uint32_t>(rank_source.next() >> 32);
            const int rank = matrix_rank_32(rows);
            if (rank == 32) {
                ++rank_counts[0];
            } else if (rank == 31) {
                ++rank_counts[1];
            } else {
                ++rank_counts[2];
            }
        }
    }

    const long double p32 = binary_matrix_rank_probability(32, 32, 32);
    const long double p31 = binary_matrix_rank_probability(31, 32, 32);
    const long double p30_or_less = 1.0L - p32 - p31;
    const std::array<long double, 3> expected_rank = {
        static_cast<long double>(kMatrixCount) * p32,
        static_cast<long double>(kMatrixCount) * p31,
        static_cast<long double>(kMatrixCount) * p30_or_less,
    };

    double rank_chi2 = 0.0;
    for (int i = 0; i < 3; ++i) {
        const long double diff =
            static_cast<long double>(rank_counts[i]) - expected_rank[i];
        rank_chi2 += static_cast<double>(diff * diff / expected_rank[i]);
    }

    return RawSuiteResult{
        .name = std::move(name),
        .monobit_p = monobit_p,
        .byte_chi2 = byte_chi2,
        .byte_p = chi_square_upper_tail_p(byte_chi2, 255.0),
        .lag1_corr = lag1_corr,
        .lag1_corr_p = lag1_correlation_pvalue(lag1_corr, kRawWordCount),
        .permutation_chi2 = permutation_chi2,
        .permutation_p = chi_square_upper_tail_p(permutation_chi2, 23.0),
        .rank_chi2 = rank_chi2,
        .rank_p = chi_square_upper_tail_p(rank_chi2, 2.0),
    };
}

auto summarize_moments(const std::vector<double>& xs) -> Moments {
    const long double n = static_cast<long double>(xs.size());
    long double mean = 0.0L;
    for (double x : xs)
        mean += x;
    mean /= n;

    long double m2 = 0.0L;
    long double m3 = 0.0L;
    long double m4 = 0.0L;
    for (double x : xs) {
        const long double d = x - mean;
        const long double d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;

    const long double skewness = m3 / std::pow(m2, 1.5L);
    const long double kurtosis = m4 / (m2 * m2) - 3.0L;
    return Moments{
        .mean = static_cast<double>(mean),
        .variance = static_cast<double>(m2),
        .skewness = static_cast<double>(skewness),
        .excess_kurtosis = static_cast<double>(kurtosis),
    };
}

template <typename CdfFn>
auto ks_statistic(std::vector<double> samples, CdfFn&& cdf) -> double {
    std::sort(samples.begin(), samples.end());
    const double n = static_cast<double>(samples.size());
    double d = 0.0;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const double fi = cdf(samples[i]);
        const double lo = static_cast<double>(i) / n;
        const double hi = static_cast<double>(i + 1) / n;
        d = std::max(d, std::fabs(fi - lo));
        d = std::max(d, std::fabs(hi - fi));
    }
    return d;
}

template <typename FillFn, typename CdfFn>
auto run_continuous_distribution_suite(std::string name, FillFn&& fill,
                                       CdfFn&& cdf) -> ContinuousDistributionResult {
    std::vector<double> samples(kDistributionSamples);
    fill(samples.data(), samples.size());
    const Moments moments = summarize_moments(samples);
    const double corr = lag1_correlation(samples);
    const double d = ks_statistic(samples, std::forward<CdfFn>(cdf));
    return ContinuousDistributionResult{
        .name = std::move(name),
        .mean = moments.mean,
        .variance = moments.variance,
        .skewness = moments.skewness,
        .excess_kurtosis = moments.excess_kurtosis,
        .lag1_corr = corr,
        .lag1_corr_p = lag1_correlation_pvalue(corr, samples.size()),
        .ks_d = d,
        .ks_p = ks_pvalue(samples.size(), d),
    };
}

template <typename FillFn>
auto run_bernoulli_suite(std::string name, FillFn&& fill) -> BernoulliResult {
    std::vector<double> samples(kDistributionSamples);
    fill(samples.data(), samples.size());

    std::size_t ones = 0;
    std::size_t runs = samples.empty() ? 0 : 1;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (samples[i] > 0.5)
            ++ones;
        if (i > 0 && (samples[i] > 0.5) != (samples[i - 1] > 0.5))
            ++runs;
    }

    const double n = static_cast<double>(samples.size());
    const double p = 0.3;
    const double q = 1.0 - p;
    const double mean = static_cast<double>(ones) / n;
    const double mean_z = (static_cast<double>(ones) - n * p) / std::sqrt(n * p * q);

    const double n1 = static_cast<double>(ones);
    const double n0 = n - n1;
    const double expected_runs = 1.0 + 2.0 * n1 * n0 / n;
    const double runs_var =
        2.0 * n1 * n0 * (2.0 * n1 * n0 - n) / (n * n * (n - 1.0));
    const double runs_z = (static_cast<double>(runs) - expected_runs) / std::sqrt(runs_var);

    const double corr = lag1_correlation(samples);
    return BernoulliResult{
        .name = std::move(name),
        .mean = mean,
        .mean_p = normal_two_sided_p(mean_z),
        .lag1_corr = corr,
        .lag1_corr_p = lag1_correlation_pvalue(corr, samples.size()),
        .runs_p = normal_two_sided_p(runs_z),
    };
}

struct ScalarWordSource {
    explicit ScalarWordSource(std::uint64_t seed) : rng(seed) {}
    auto next() -> std::uint64_t { return rng(); }
    zorro::Xoshiro256pp rng;
};

struct X2WordSource {
    explicit X2WordSource(std::uint64_t seed) : rng(seed) {}

    auto next() -> std::uint64_t {
        if (index == buffer.size()) {
            buffer = rng();
            index = 0;
        }
        return buffer[index++];
    }

    zorro::Xoshiro256pp_x2 rng;
    std::array<std::uint64_t, 2> buffer = {};
    std::size_t index = 2;
};

struct X4WordSource {
    explicit X4WordSource(std::uint64_t seed) : rng(seed) {}

    auto next() -> std::uint64_t {
        if (index == buffer.size()) {
            buffer = rng();
            index = 0;
        }
        return buffer[index++];
    }

    zorro::Xoshiro256pp_x4_portable rng;
    std::array<std::uint64_t, 4> buffer = {};
    std::size_t index = 4;
};

void print_raw_suite(const RawSuiteResult& suite) {
    std::cout << suite.name << '\n';
    std::cout << "  monobit p=" << suite.monobit_p << '\n';
    std::cout << "  byte chi2=" << suite.byte_chi2 << " p=" << suite.byte_p << '\n';
    std::cout << "  lag1 corr=" << suite.lag1_corr << " p=" << suite.lag1_corr_p << '\n';
    std::cout << "  permutation chi2=" << suite.permutation_chi2
              << " p=" << suite.permutation_p << '\n';
    std::cout << "  rank chi2=" << suite.rank_chi2 << " p=" << suite.rank_p << '\n';
}

void print_distribution_suite(const ContinuousDistributionResult& suite) {
    std::cout << suite.name << '\n';
    std::cout << "  mean=" << suite.mean << " variance=" << suite.variance
              << " skew=" << suite.skewness
              << " excess_kurtosis=" << suite.excess_kurtosis << '\n';
    std::cout << "  lag1 corr=" << suite.lag1_corr << " p=" << suite.lag1_corr_p
              << '\n';
    std::cout << "  KS D=" << suite.ks_d << " p=" << suite.ks_p << '\n';
}

void print_bernoulli_suite(const BernoulliResult& suite) {
    std::cout << suite.name << '\n';
    std::cout << "  mean=" << suite.mean << " p=" << suite.mean_p << '\n';
    std::cout << "  lag1 corr=" << suite.lag1_corr << " p=" << suite.lag1_corr_p
              << '\n';
    std::cout << "  runs p=" << suite.runs_p << '\n';
}

}  // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Raw bitstream suites\n";
    print_raw_suite(run_raw_suite("Xoshiro256pp scalar", [] {
        return ScalarWordSource(kSeed);
    }));
    print_raw_suite(run_raw_suite("Xoshiro256pp x2 interleaved", [] {
        return X2WordSource(kSeed);
    }));
    print_raw_suite(run_raw_suite("Xoshiro256pp x4 portable interleaved", [] {
        return X4WordSource(kSeed);
    }));

    zorro::Rng rng_uniform(kSeed);
    zorro::Rng rng_normal(kSeed);
    zorro::Rng rng_exponential(kSeed);
    zorro::Rng rng_bernoulli(kSeed);

    std::cout << "\nDistribution suites\n";
    print_distribution_suite(run_continuous_distribution_suite(
        "Rng::fill_uniform [0,1)", [&](double* out, std::size_t count) {
            rng_uniform.fill_uniform(out, count);
        },
        [](double x) {
            return std::clamp(x, 0.0, 1.0);
        }));

    print_distribution_suite(run_continuous_distribution_suite(
        "Rng::fill_normal N(0,1)", [&](double* out, std::size_t count) {
            rng_normal.fill_normal(out, count);
        },
        [](double x) {
            return 0.5 * (1.0 + std::erf(x / std::numbers::sqrt2_v<double>));
        }));

    print_distribution_suite(run_continuous_distribution_suite(
        "Rng::fill_exponential Exp(1)", [&](double* out, std::size_t count) {
            rng_exponential.fill_exponential(out, count);
        },
        [](double x) {
            return x <= 0.0 ? 0.0 : 1.0 - std::exp(-x);
        }));

    print_bernoulli_suite(
        run_bernoulli_suite("Rng::fill_bernoulli p=0.3",
                            [&](double* out, std::size_t count) {
                                rng_bernoulli.fill_bernoulli(out, count, 0.3);
                            }));

    return 0;
}
