#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
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

// This example simulates repeated draws from a high-dimensional latent-factor
// model. One simulated path is a vector x in R^dimension with coordinates
//
//   x_i = mean_i + sigma_i * epsilon_i + sum_j loading_{j,i} * factor_j
//
// where:
// - epsilon_i are independent N(0, 1) shocks
// - factor_j are shared Student-t shocks with heavier tails
//
// The shared factors create cross-coordinate dependence, while the Gaussian
// idiosyncratic term keeps each coordinate noisy on its own. This is a useful
// benchmark shape because it needs both a large block of normals and a smaller
// block of heavy-tailed draws for every batch.

struct Options {
  std::string backend = "both";
  std::size_t dimension = 2048;
  std::size_t paths = 4096;
  std::size_t repetitions = 8;
  std::size_t factors = 6;
  double degrees_of_freedom = 7.0;
  std::uint64_t seed = 0x123456789abcdef0ULL;
};

struct Scenario {
  // Deterministic parameters for the synthetic factor model.
  std::vector<double> means;
  std::vector<double> idiosyncratic_scale;
  std::vector<double> loadings;
  std::vector<double> weights;
  double exceedance_threshold = 0.0;
};

// The example compares RNG backends, so it batches enough coordinates to avoid
// tiny per-call overhead dominating the timings at small dimensions.
constexpr std::size_t kTargetBatchCoordinates = 1u << 18;

struct SimulationResult {
  std::string backend;
  double elapsed_seconds = 0.0;
  double coordinates_per_second = 0.0;
  double mean_projection = 0.0;
  double projection_stddev = 0.0;
  double mean_squared_radius = 0.0;
  double exceedance_probability = 0.0;
  double checksum = 0.0;
};

class RandomSource {
public:
  virtual ~RandomSource() = default;

  [[nodiscard]] virtual auto name() const -> std::string_view = 0;
  virtual void fill_normal(double *out, std::size_t count, double mean,
                           double stddev) = 0;
  virtual void fill_student_t(double *out, std::size_t count, double nu) = 0;
};

class StdRandomSource final : public RandomSource {
public:
  explicit StdRandomSource(std::uint64_t seed) : engine_(seed) {}

  [[nodiscard]] auto name() const -> std::string_view override { return "std"; }

  void fill_normal(double *out, std::size_t count, double mean,
                   double stddev) override {
    const std::normal_distribution<double>::param_type params(mean, stddev);
    for (std::size_t i = 0; i < count; ++i)
      out[i] = normal_(engine_, params);
  }

  void fill_student_t(double *out, std::size_t count, double nu) override {
    const std::student_t_distribution<double>::param_type params(nu);
    for (std::size_t i = 0; i < count; ++i)
      out[i] = student_t_(engine_, params);
  }

private:
  std::mt19937_64 engine_;
  std::normal_distribution<double> normal_;
  std::student_t_distribution<double> student_t_;
};

class ZorroRandomSource final : public RandomSource {
public:
  explicit ZorroRandomSource(std::uint64_t seed) : rng_(seed) {}

  [[nodiscard]] auto name() const -> std::string_view override {
    return "zorro";
  }

  void fill_normal(double *out, std::size_t count, double mean,
                   double stddev) override {
    rng_.fill_normal(out, count, mean, stddev);
  }

  void fill_student_t(double *out, std::size_t count, double nu) override {
    rng_.fill_student_t(out, count, nu);
  }

private:
  zorro::Rng rng_;
};

[[noreturn]] void print_usage_and_exit(const char *argv0, int exit_code) {
  std::ostream &stream = exit_code == EXIT_SUCCESS ? std::cout : std::cerr;
  stream
      << "Usage: " << argv0 << " [--backend std|zorro|both]"
      << " [--dimension N] [--paths N] [--repetitions N]"
      << " [--factors N] [--degrees-of-freedom X] [--seed N]\n";
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
    if (arg == "--paths") {
      options.paths = parse_size(require_value(arg), arg);
      continue;
    }
    if (arg == "--repetitions") {
      options.repetitions = parse_size(require_value(arg), arg);
      continue;
    }
    if (arg == "--factors") {
      options.factors = parse_size(require_value(arg), arg);
      continue;
    }
    if (arg == "--degrees-of-freedom") {
      options.degrees_of_freedom = parse_double(require_value(arg), arg);
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
    throw std::runtime_error(
        "backend must be one of: std, zorro, both");
  }
  if (options.dimension == 0 || options.paths == 0 || options.repetitions == 0)
    throw std::runtime_error("dimension, paths, and repetitions must be > 0");
  if (options.factors == 0)
    throw std::runtime_error("factors must be > 0");
  if (options.degrees_of_freedom <= 2.0)
    throw std::runtime_error("degrees of freedom must be > 2");
  return options;
}

auto make_scenario(std::size_t dimension, std::size_t factors) -> Scenario {
  Scenario scenario;
  scenario.means.resize(dimension);
  scenario.idiosyncratic_scale.resize(dimension);
  scenario.loadings.resize(dimension * factors);
  scenario.weights.resize(dimension);

  const double inv_sqrt_dimension = 1.0 / std::sqrt(static_cast<double>(dimension));
  for (std::size_t i = 0; i < dimension; ++i) {
    const double idx = static_cast<double>(i + 1);
    // Keep the setup deterministic but nontrivial: a small structured mean,
    // heterogeneous idiosyncratic scales, and smooth factor loadings so the
    // workload is reproducible without looking like iid white noise. The
    // 1/sqrt(dimension) normalization keeps the projection statistic roughly
    // comparable as the ambient dimension changes.
    scenario.means[i] = 0.01 * std::sin(idx * 0.017);
    scenario.idiosyncratic_scale[i] =
        0.45 + 0.25 * (1.0 + std::sin(idx * 0.011));
    scenario.weights[i] =
        (0.6 * std::sin(idx * 0.007) + 0.4 * std::cos(idx * 0.013)) *
        inv_sqrt_dimension;
    for (std::size_t f = 0; f < factors; ++f) {
      const double factor = static_cast<double>(f + 1);
      scenario.loadings[f * dimension + i] =
          (0.07 + 0.015 * factor) * std::cos(idx * 0.003 * factor) +
          0.03 * std::sin(idx * 0.0017 * (factor + 1.0));
    }
  }

  // The tail metric tracks whether any coordinate in a path is "large".  A
  // fixed threshold would be too sparse in low dimensions or too common in
  // high dimensions because the expected max coordinate grows with dimension.
  // This mild log(dim) adjustment keeps the diagnostic informative across a
  // broad range of dimensions; it is a reporting heuristic, not model theory.
  scenario.exceedance_threshold =
      2.8 + 0.05 * std::log(static_cast<double>(dimension));
  return scenario;
}

void print_configuration(const Options &options, const Scenario &scenario) {
  const std::size_t batch_paths = std::max<std::size_t>(
      1, std::min(options.paths,
                  std::max<std::size_t>(
                      std::size_t{64},
                      kTargetBatchCoordinates / options.dimension)));
  std::cout << "High-dimensional factor model\n"
            << "  backends:            " << options.backend << '\n'
            << "  dimension:           " << options.dimension << '\n'
            << "  paths per repeat:    " << options.paths << '\n'
            << "  repetitions:         " << options.repetitions << '\n'
            << "  batched paths:       " << batch_paths << '\n'
            << "  common factors:      " << options.factors << '\n'
            << "  Student-t dof:       " << options.degrees_of_freedom << '\n'
            << "  exceedance thr:      " << std::fixed << std::setprecision(3)
            << scenario.exceedance_threshold << '\n'
            << "  seed:                0x" << std::hex << options.seed << std::dec
            << "\n\n";
}

auto run_simulation(RandomSource &rng, const Scenario &scenario,
                    const Options &options) -> SimulationResult {
  // Batch multiple paths per RNG call. This keeps the comparison focused on
  // backend throughput instead of per-call overhead from very small draws.
  const std::size_t batch_paths = std::max<std::size_t>(
      1, std::min(options.paths,
                  std::max<std::size_t>(
                      std::size_t{64},
                      kTargetBatchCoordinates / options.dimension)));
  std::vector<double> common_factors(batch_paths * options.factors);
  std::vector<double> idiosyncratic_noise(batch_paths * options.dimension);

  double projection_sum = 0.0;
  double projection_sum_sq = 0.0;
  double squared_radius_sum = 0.0;
  double checksum = 0.0;
  std::size_t exceedances = 0;

  const auto started = Clock::now();
  for (std::size_t repeat = 0; repeat < options.repetitions; ++repeat) {
    for (std::size_t base = 0; base < options.paths; base += batch_paths) {
      const std::size_t chunk_paths =
          std::min(batch_paths, options.paths - base);
      rng.fill_student_t(common_factors.data(), chunk_paths * options.factors,
                         options.degrees_of_freedom);
      rng.fill_normal(idiosyncratic_noise.data(), chunk_paths * options.dimension,
                      0.0, 1.0);

      for (std::size_t path = 0; path < chunk_paths; ++path) {
        // Each path gets:
        // - one shared vector of heavy-tailed factor realizations
        // - one independent Gaussian shock per coordinate
        const double *factor_row =
            common_factors.data() + path * options.factors;
        const double *noise_row =
            idiosyncratic_noise.data() + path * options.dimension;

        // Build one high-dimensional draw from common heavy-tailed factors plus
        // independent Gaussian noise, then summarize it with a few scalar
        // diagnostics:
        // - projection: a weighted linear functional of x
        // - squared_radius: ||x||^2, a size/energy proxy
        // - max_abs: the largest coordinate magnitude in the path
        //
        // These summaries make the workload look more like a downstream Monte
        // Carlo kernel instead of "generate numbers and throw them away".
        double projection = 0.0;
        double squared_radius = 0.0;
        double max_abs = 0.0;
        for (std::size_t i = 0; i < options.dimension; ++i) {
          double value =
              scenario.means[i] + scenario.idiosyncratic_scale[i] * noise_row[i];
          for (std::size_t f = 0; f < options.factors; ++f)
            value += scenario.loadings[f * options.dimension + i] *
                     factor_row[f];
          projection += scenario.weights[i] * value;
          squared_radius += value * value;
          max_abs = std::max(max_abs, std::abs(value));
        }

        projection_sum += projection;
        projection_sum_sq += projection * projection;
        squared_radius_sum += squared_radius;
        checksum += projection + 1e-6 * squared_radius + 0.01 * max_abs;
        exceedances += max_abs > scenario.exceedance_threshold ? 1u : 0u;
      }
    }
  }
  const double elapsed_seconds = Seconds(Clock::now() - started).count();

  const double total_paths =
      static_cast<double>(options.paths) * static_cast<double>(options.repetitions);
  const double total_coordinates =
      total_paths * static_cast<double>(options.dimension);
  const double mean_projection = projection_sum / total_paths;
  const double projection_variance =
      std::max(0.0, projection_sum_sq / total_paths -
                        mean_projection * mean_projection);

  return SimulationResult{
      .backend = std::string(rng.name()),
      .elapsed_seconds = elapsed_seconds,
      .coordinates_per_second = total_coordinates / elapsed_seconds,
      .mean_projection = mean_projection,
      .projection_stddev = std::sqrt(projection_variance),
      .mean_squared_radius = squared_radius_sum / total_paths,
      .exceedance_probability =
          static_cast<double>(exceedances) / total_paths,
      .checksum = checksum,
  };
}

void print_result(const SimulationResult &result) {
  // Report both throughput and a few stable summary statistics so it is easy
  // to tell whether two backends are sampling from plausibly similar laws.
  std::cout << result.backend << '\n'
            << "  elapsed:             " << std::fixed << std::setprecision(3)
            << result.elapsed_seconds << " s\n"
            << "  coordinates/s:       " << std::setprecision(2)
            << (result.coordinates_per_second / 1'000'000.0) << " M\n"
            << "  mean projection:     " << std::setprecision(6)
            << result.mean_projection << '\n'
            << "  projection stddev:   " << result.projection_stddev << '\n'
            << "  mean squared radius: " << result.mean_squared_radius << '\n'
            << "  P(max |x_i| > thr):  " << result.exceedance_probability << '\n'
            << "  checksum:            " << std::setprecision(12)
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
    const Scenario scenario = make_scenario(options.dimension, options.factors);
    const auto backends = make_backends(options);

    print_configuration(options, scenario);
    for (const auto &backend : backends)
      print_result(run_simulation(*backend, scenario, options));
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    print_usage_and_exit(argv[0], EXIT_FAILURE);
  }
}
