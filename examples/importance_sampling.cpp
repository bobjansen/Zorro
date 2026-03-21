#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "zorro/zorro.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

// This example estimates a rare portfolio-loss probability under a correlated
// Gaussian factor model.
//
// One path draws:
// - k shared factors f_j ~ N(0, 1)
// - d idiosyncratic shocks e_i ~ N(0, 1)
//
// Then asset returns are
//
//   x_i = mean_i + sigma_i * e_i + sum_j loading_{j,i} * f_j
//
// and the portfolio loss is
//
//   L = -sum_i weight_i * x_i
//
// We estimate P(L >= threshold). For a high threshold, plain Monte Carlo wastes
// many paths on non-loss scenarios. Importance sampling shifts the Gaussian
// proposal toward the loss region and reweights each path with the exact
// likelihood ratio.

struct Options {
  std::string backend = "both";
  std::size_t dimension = 1024;
  std::size_t factors = 8;
  std::size_t paths = 250000;
  std::size_t repetitions = 4;
  double rare_z = 5.0;
  std::uint64_t seed = 0x123456789abcdef0ULL;
};

struct Scenario {
  std::vector<double> means;
  std::vector<double> idiosyncratic_scale;
  std::vector<double> loadings;
  std::vector<double> weights;

  // Loss can be written directly as:
  //
  //   L = baseline_loss
  //       + sum_j factor_loss_coeff[j] * f_j
  //       + sum_i idio_loss_coeff[i]   * e_i
  //
  // so the simulation can work with the compressed loss coefficients without
  // explicitly materializing x_i.
  std::vector<double> factor_loss_coeff;
  std::vector<double> idio_loss_coeff;

  // Mean-shift proposal for importance sampling. The shift points in the
  // steepest loss direction and is scaled so the proposal mean lands on the
  // threshold. This is simple, easy to explain, and often a strong variance
  // reduction for Gaussian rare-event tails.
  std::vector<double> factor_shift;
  std::vector<double> idio_shift;

  double baseline_loss = 0.0;
  double loss_stddev = 0.0;
  double loss_threshold = 0.0;
  double exact_tail_probability = 0.0;
  double shift_norm_sq = 0.0;
};

struct EstimateResult {
  std::string backend;
  std::string method;
  double elapsed_seconds = 0.0;
  double normals_per_second = 0.0;
  double estimate = 0.0;
  double standard_error = 0.0;
  double relative_error = 0.0;
  double hit_rate = 0.0;
  double upper_95_bound = 0.0;
  double checksum = 0.0;
  std::size_t hit_count = 0;
  std::size_t total_paths = 0;
};

// The example is meant to compare RNG throughput inside a Monte Carlo kernel,
// not the cost of millions of tiny API calls.
constexpr std::size_t kTargetBatchNormals = 1u << 18;

class RandomSource {
public:
  virtual ~RandomSource() = default;
  [[nodiscard]] virtual auto name() const -> std::string_view = 0;
  virtual void fill_standard_normal(double *out, std::size_t count) = 0;
};

class StdRandomSource final : public RandomSource {
public:
  explicit StdRandomSource(std::uint64_t seed) : engine_(seed) {}

  [[nodiscard]] auto name() const -> std::string_view override { return "std"; }

  void fill_standard_normal(double *out, std::size_t count) override {
    for (std::size_t i = 0; i < count; ++i)
      out[i] = normal_(engine_);
  }

private:
  std::mt19937_64 engine_;
  std::normal_distribution<double> normal_;
};

class ZorroRandomSource final : public RandomSource {
public:
  explicit ZorroRandomSource(std::uint64_t seed) : rng_(seed) {}

  [[nodiscard]] auto name() const -> std::string_view override {
    return "zorro";
  }

  void fill_standard_normal(double *out, std::size_t count) override {
    rng_.fill_normal(out, count, 0.0, 1.0);
  }

private:
  zorro::Rng rng_;
};

[[noreturn]] void print_usage_and_exit(const char *argv0, int exit_code) {
  std::ostream &stream = exit_code == EXIT_SUCCESS ? std::cout : std::cerr;
  stream
      << "Usage: " << argv0 << " [--backend std|zorro|both]"
      << " [--dimension N] [--factors N] [--paths N]"
      << " [--repetitions N] [--rare-z X] [--seed N]\n";
  std::exit(exit_code);
}

auto parse_size(std::string_view text, std::string_view flag) -> std::size_t {
  try {
    std::size_t offset = 0;
    const auto value = std::stoull(std::string(text), &offset, 10);
    if (offset != text.size())
      throw std::invalid_argument("trailing characters");
    return static_cast<std::size_t>(value);
  } catch (const std::exception &) {
    throw std::runtime_error("invalid value for " + std::string(flag) + ": " +
                             std::string(text));
  }
}

auto parse_u64(std::string_view text, std::string_view flag) -> std::uint64_t {
  try {
    std::size_t offset = 0;
    const auto value = std::stoull(std::string(text), &offset, 0);
    if (offset != text.size())
      throw std::invalid_argument("trailing characters");
    return static_cast<std::uint64_t>(value);
  } catch (const std::exception &) {
    throw std::runtime_error("invalid value for " + std::string(flag) + ": " +
                             std::string(text));
  }
}

auto parse_double(std::string_view text, std::string_view flag) -> double {
  try {
    std::size_t offset = 0;
    const double value = std::stod(std::string(text), &offset);
    if (offset != text.size())
      throw std::invalid_argument("trailing characters");
    return value;
  } catch (const std::exception &) {
    throw std::runtime_error("invalid value for " + std::string(flag) + ": " +
                             std::string(text));
  }
}

auto parse_options(int argc, char **argv) -> Options {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    auto require_value = [&](std::string_view flag) -> std::string_view {
      if (i + 1 >= argc)
        throw std::runtime_error("missing value for " + std::string(flag));
      return argv[++i];
    };

    if (arg == "--help")
      print_usage_and_exit(argv[0], EXIT_SUCCESS);
    if (arg == "--backend") {
      options.backend = std::string(require_value(arg));
      continue;
    }
    if (arg == "--dimension") {
      options.dimension = parse_size(require_value(arg), arg);
      continue;
    }
    if (arg == "--factors") {
      options.factors = parse_size(require_value(arg), arg);
      continue;
    }
    if (arg == "--paths") {
      options.paths = parse_size(require_value(arg), arg);
      continue;
    }
    if (arg == "--repetitions") {
      options.repetitions = parse_size(require_value(arg), arg);
      continue;
    }
    if (arg == "--rare-z") {
      options.rare_z = parse_double(require_value(arg), arg);
      continue;
    }
    if (arg == "--seed") {
      options.seed = parse_u64(require_value(arg), arg);
      continue;
    }
    throw std::runtime_error("unknown option: " + std::string(arg));
  }

  if (options.backend != "std" && options.backend != "zorro" &&
      options.backend != "both") {
    throw std::runtime_error("backend must be one of: std, zorro, both");
  }
  if (options.dimension == 0 || options.factors == 0 || options.paths == 0 ||
      options.repetitions == 0) {
    throw std::runtime_error(
        "dimension, factors, paths, and repetitions must be > 0");
  }
  if (options.rare_z <= 0.0)
    throw std::runtime_error("rare-z must be > 0");
  return options;
}

auto normal_tail_probability(double z) -> double {
  return 0.5 * std::erfc(z / std::sqrt(2.0));
}

auto make_scenario(const Options &options) -> Scenario {
  Scenario scenario;
  scenario.means.resize(options.dimension);
  scenario.idiosyncratic_scale.resize(options.dimension);
  scenario.loadings.resize(options.dimension * options.factors);
  scenario.weights.resize(options.dimension);
  scenario.factor_loss_coeff.assign(options.factors, 0.0);
  scenario.idio_loss_coeff.resize(options.dimension);
  scenario.factor_shift.resize(options.factors);
  scenario.idio_shift.resize(options.dimension);

  const double inv_sqrt_dimension =
      1.0 / std::sqrt(static_cast<double>(options.dimension));

  for (std::size_t i = 0; i < options.dimension; ++i) {
    const double idx = static_cast<double>(i + 1);

    // Deterministic but structured portfolio specification. The signs and
    // amplitudes are chosen so the portfolio is diversified rather than a
    // degenerate all-positive loading on a single mode.
    scenario.means[i] = 0.0005 + 0.002 * std::sin(idx * 0.019);
    scenario.idiosyncratic_scale[i] =
        0.18 + 0.12 * (1.0 + std::cos(idx * 0.013));
    scenario.weights[i] =
        (0.75 + 0.2 * std::sin(idx * 0.005) + 0.05 * std::cos(idx * 0.021)) *
        inv_sqrt_dimension;

    for (std::size_t f = 0; f < options.factors; ++f) {
      const double factor = static_cast<double>(f + 1);
      scenario.loadings[f * options.dimension + i] =
          0.035 * std::cos(idx * 0.0031 * factor) +
          (0.05 + 0.01 * factor) * std::sin(idx * 0.0017 * (factor + 1.0));
    }
  }

  // Compress the factor model into direct coefficients of the scalar loss.
  for (std::size_t i = 0; i < options.dimension; ++i) {
    scenario.baseline_loss -= scenario.weights[i] * scenario.means[i];
    scenario.idio_loss_coeff[i] =
        -scenario.weights[i] * scenario.idiosyncratic_scale[i];
    for (std::size_t f = 0; f < options.factors; ++f) {
      scenario.factor_loss_coeff[f] -=
          scenario.weights[i] * scenario.loadings[f * options.dimension + i];
    }
  }

  double variance = 0.0;
  for (double coeff : scenario.factor_loss_coeff)
    variance += coeff * coeff;
  for (double coeff : scenario.idio_loss_coeff)
    variance += coeff * coeff;
  scenario.loss_stddev = std::sqrt(variance);
  scenario.loss_threshold =
      scenario.baseline_loss + options.rare_z * scenario.loss_stddev;

  // Under the original measure, L is affine in independent standard normals,
  // so L itself is Gaussian. That gives an exact answer for the target tail
  // probability, which is useful for validating the Monte Carlo estimators.
  scenario.exact_tail_probability = normal_tail_probability(options.rare_z);

  // Mean-shift proposal: delta = alpha * c, where c is the gradient of the
  // linear loss in standard-normal coordinates. alpha is chosen so that
  // E_q[L] = threshold, i.e. the shifted proposal sits on the rare-event
  // boundary instead of wasting most paths far from it.
  const double alpha =
      (scenario.loss_threshold - scenario.baseline_loss) / variance;
  for (std::size_t f = 0; f < options.factors; ++f) {
    scenario.factor_shift[f] = alpha * scenario.factor_loss_coeff[f];
    scenario.shift_norm_sq += scenario.factor_shift[f] * scenario.factor_shift[f];
  }
  for (std::size_t i = 0; i < options.dimension; ++i) {
    scenario.idio_shift[i] = alpha * scenario.idio_loss_coeff[i];
    scenario.shift_norm_sq += scenario.idio_shift[i] * scenario.idio_shift[i];
  }

  return scenario;
}

void print_configuration(const Options &options, const Scenario &scenario) {
  const std::size_t normals_per_path = options.dimension + options.factors;
  const std::size_t batch_paths = std::max<std::size_t>(
      1, std::min(options.paths,
                  std::max<std::size_t>(
                      std::size_t{64},
                      kTargetBatchNormals / normals_per_path)));

  std::cout << "Importance sampling for rare portfolio loss\n"
            << "  backends:            " << options.backend << '\n'
            << "  dimension:           " << options.dimension << '\n'
            << "  common factors:      " << options.factors << '\n'
            << "  paths per repeat:    " << options.paths << '\n'
            << "  repetitions:         " << options.repetitions << '\n'
            << "  batched paths:       " << batch_paths << '\n'
            << "  rare z-score:        " << std::fixed << std::setprecision(2)
            << options.rare_z << '\n'
            << "  loss mean:           " << std::setprecision(6)
            << scenario.baseline_loss << '\n'
            << "  loss stddev:         " << scenario.loss_stddev << '\n'
            << "  loss threshold:      " << scenario.loss_threshold << '\n'
            << "  exact tail prob:     " << std::scientific << std::setprecision(6)
            << scenario.exact_tail_probability << '\n'
            << "  seed:                0x" << std::hex << options.seed << std::dec
            << "\n\n";
}

enum class Method {
  Plain,
  ImportanceSampling,
};

auto method_name(Method method) -> std::string {
  if (method == Method::Plain)
    return "plain_mc";
  return "importance_sampling";
}

auto run_estimate(RandomSource &rng, const Scenario &scenario,
                  const Options &options, Method method) -> EstimateResult {
  const std::size_t normals_per_path = options.dimension + options.factors;
  const std::size_t batch_paths = std::max<std::size_t>(
      1, std::min(options.paths,
                  std::max<std::size_t>(
                      std::size_t{64},
                      kTargetBatchNormals / normals_per_path)));

  std::vector<double> factor_normals(batch_paths * options.factors);
  std::vector<double> idio_normals(batch_paths * options.dimension);

  double sum = 0.0;
  double sum_sq = 0.0;
  double checksum = 0.0;
  std::size_t hit_count = 0;

  const auto started = Clock::now();
  for (std::size_t repeat = 0; repeat < options.repetitions; ++repeat) {
    for (std::size_t base = 0; base < options.paths; base += batch_paths) {
      const std::size_t chunk_paths =
          std::min(batch_paths, options.paths - base);
      rng.fill_standard_normal(factor_normals.data(),
                               chunk_paths * options.factors);
      rng.fill_standard_normal(idio_normals.data(),
                               chunk_paths * options.dimension);

      for (std::size_t path = 0; path < chunk_paths; ++path) {
        const double *factor_row =
            factor_normals.data() + path * options.factors;
        const double *idio_row =
            idio_normals.data() + path * options.dimension;

        // We always draw standard normals first. Importance sampling then adds
        // a deterministic shift delta and applies the exact Gaussian likelihood
        // ratio p(x) / q(x) = exp(-delta·z - ||delta||^2 / 2), where z is the
        // original standard-normal draw before shifting.
        double loss = scenario.baseline_loss;
        double shift_dot_standard_draw = 0.0;

        for (std::size_t f = 0; f < options.factors; ++f) {
          double sample = factor_row[f];
          if (method == Method::ImportanceSampling) {
            sample += scenario.factor_shift[f];
            shift_dot_standard_draw += scenario.factor_shift[f] * factor_row[f];
          }
          loss += scenario.factor_loss_coeff[f] * sample;
        }
        for (std::size_t i = 0; i < options.dimension; ++i) {
          double sample = idio_row[i];
          if (method == Method::ImportanceSampling) {
            sample += scenario.idio_shift[i];
            shift_dot_standard_draw += scenario.idio_shift[i] * idio_row[i];
          }
          loss += scenario.idio_loss_coeff[i] * sample;
        }

        const bool hit = loss >= scenario.loss_threshold;
        const double weight =
            method == Method::ImportanceSampling
                ? std::exp(-shift_dot_standard_draw - 0.5 * scenario.shift_norm_sq)
                : 1.0;
        const double contribution = hit ? weight : 0.0;

        sum += contribution;
        sum_sq += contribution * contribution;
        hit_count += hit ? 1u : 0u;
        checksum += 1e-3 * loss + contribution;
      }
    }
  }
  const double elapsed_seconds = Seconds(Clock::now() - started).count();

  const double total_paths =
      static_cast<double>(options.paths) * static_cast<double>(options.repetitions);
  const double total_normals =
      total_paths * static_cast<double>(normals_per_path);
  const double estimate = sum / total_paths;
  const double second_moment = sum_sq / total_paths;
  const double variance = std::max(0.0, second_moment - estimate * estimate);
  const double standard_error = std::sqrt(variance / total_paths);
  const double relative_error =
      estimate > 0.0 ? standard_error / estimate : std::numeric_limits<double>::infinity();
  // For very rare events, zero hits are common. The plug-in standard error is
  // then misleadingly zero, so also report a simple one-sided 95% upper bound.
  // With zero hits this is the familiar "rule of three": about 3 / N.
  const double upper_95_bound =
      hit_count == 0
          ? -std::log(0.05) / total_paths
          : estimate + 1.645 * standard_error;

  return EstimateResult{
      .backend = std::string(rng.name()),
      .method = method_name(method),
      .elapsed_seconds = elapsed_seconds,
      .normals_per_second = total_normals / elapsed_seconds,
      .estimate = estimate,
      .standard_error = standard_error,
      .relative_error = relative_error,
      .hit_rate = static_cast<double>(hit_count) / total_paths,
      .upper_95_bound = upper_95_bound,
      .checksum = checksum,
      .hit_count = hit_count,
      .total_paths = static_cast<std::size_t>(total_paths),
  };
}

void print_result(const EstimateResult &result, const Scenario &scenario) {
  std::cout << result.backend << " / " << result.method << '\n'
            << "  elapsed:             " << std::fixed << std::setprecision(3)
            << result.elapsed_seconds << " s\n"
            << "  normal draws/s:      " << std::setprecision(2)
            << (result.normals_per_second / 1'000'000.0) << " M\n"
            << "  estimate:            " << std::scientific << std::setprecision(6)
            << result.estimate << '\n'
            << "  standard error:      " << result.standard_error << '\n'
            << "  relative error:      " << std::fixed << std::setprecision(4)
            << result.relative_error << '\n'
            << "  est / exact ratio:   " << std::setprecision(6)
            << (scenario.exact_tail_probability > 0.0
                    ? result.estimate / scenario.exact_tail_probability
                    : std::numeric_limits<double>::infinity())
            << '\n'
            << "  sampled hit rate:    " << std::scientific << std::setprecision(6)
            << result.hit_rate << '\n'
            << "  hits / paths:        " << std::fixed << result.hit_count << " / "
            << result.total_paths << '\n'
            << "  upper 95% bound:     " << std::scientific << std::setprecision(6)
            << result.upper_95_bound << '\n'
            << "  checksum:            " << std::scientific << std::setprecision(6)
            << result.checksum << "\n\n";
}

auto make_backends(const Options &options)
    -> std::vector<std::unique_ptr<RandomSource>> {
  std::vector<std::unique_ptr<RandomSource>> backends;
  if (options.backend == "std" || options.backend == "both")
    backends.push_back(std::make_unique<StdRandomSource>(options.seed));
  if (options.backend == "zorro" || options.backend == "both")
    backends.push_back(std::make_unique<ZorroRandomSource>(options.seed));
  return backends;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    const Scenario scenario = make_scenario(options);
    const auto backends = make_backends(options);

    print_configuration(options, scenario);
    for (const auto &backend : backends) {
      print_result(run_estimate(*backend, scenario, options, Method::Plain),
                   scenario);
      print_result(
          run_estimate(*backend, scenario, options, Method::ImportanceSampling),
          scenario);
    }
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    print_usage_and_exit(argv[0], EXIT_FAILURE);
  }
}
