# zorro

Zorro is a standalone `xoshiro256++` experiment: scalar, 2-lane, 4-lane, and wider SIMD-oriented layouts plus a benchmark harness for uniform and normal generation.

The benchmark target is configured for C++23.

The core public API lives in `include/zorro/zorro.hpp` under the `zorro`
namespace. The root-level `zorro.hpp` and `rng.hpp` headers are compatibility
shims during the cleanup.

The benchmark target compares `2^24` samples for both `uniform(0, 1)` and standard-normal (`N(0, 1)`) generation across:

- `std::mt19937`
- scalar `xoshiro256++`
- 2-lane portable `xoshiro256++`
- 4-lane portable `xoshiro256++`
- Stephan Friedl's handwritten AVX2 `Xoshiro256PlusSIMD` implementation

Recent AWS runs under `aws/results/20260312-085709` show that the most interesting normal-path variants are the vectorized transforms (`veclog` and especially `vecpolar`), not the earlier extract/batched-only variants.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

For local machine-specific benchmark runs:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRNG_BENCH_ENABLE_NATIVE=ON
cmake --build build -j
```

If you need to disable the handwritten AVX2 comparison:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRNG_BENCH_ENABLE_STEPHANFR_AVX2=OFF
```

## Run

```bash
./build/benchmark_distributions
```

The benchmark uses a fixed seed and reports the best, median, and mean wall-clock time plus derived throughput in samples/second.

For the Zorro benchmarks, the driver uses local engine instances rather than the thread-local helper API so the measurements focus on generator and distribution throughput.

## Third-party code

The handwritten SIMD comparison uses a vendored snapshot of Stephan Friedl's `Xoshiro256PlusSIMD` headers under `third_party/stephanfr_xoshiro256plussimd`, taken from upstream commit `29b30821f8f59b6106462c03d1d0225c93c4545d`. The upstream license is preserved in `third_party/stephanfr_xoshiro256plussimd/LICENSE`.
