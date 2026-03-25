# zorro

Zorro is a header-only `xoshiro256++` library
([paper](https://vigna.di.unimi.it/papers.php#BlVSLPNG) by Blackman and
Vigna) with automatic SIMD dispatch. Add one header and get high-throughput
random generation for uniform, normal, exponential, Bernoulli, gamma, and
Student's t distributions.

## Quick start

Zorro is header-only. The main header is:

```text
include/zorro/zorro.hpp
```

That is the canonical public include. Compile with `-O3` (`C++20` or later)
plus the SIMD flags you want:

- portable: no extra SIMD flags
- AVX2: `-mavx2`
- AVX2 + FMA: `-mavx2 -mfma`
- AVX-512: `-mavx512f -mavx512vl -mavx512dq`
- host-tuned: `-march=native`

For example:

```bash
g++ -O3 -std=c++20 -mavx2 -I./include main.cpp
```

For repository builds with CMake, `-DRNG_BENCH_ENABLE_AVX512=ON` enables the
AVX-512 targets explicitly.

### `zorro::Rng`

`zorro::Rng` is the main bulk-fill API. You provide a `double*` output buffer
and a sample count. See `examples/print.cpp` for a complete example covering
uniform, normal, exponential, Bernoulli, gamma, and Student's t generation.

Main methods:

- `fill_uniform(out, count)` for uniform samples in `[0, 1)`.
- `fill_uniform(out, count, low, high)` for uniform samples in `[low, high)`.
- `fill_normal(out, count, mean, stddev)` for normal samples.
- `fill_exponential(out, count, lambda)` for exponential samples.
- `fill_bernoulli(out, count, p)` for `0.0` / `1.0` Bernoulli output.
- `fill_gamma(out, count, alpha)` for `Gamma(alpha, 1)`.
- `fill_student_t(out, count, nu)` for Student's t samples.

From this repo, you can build it directly with:

```bash
g++ -O3 -std=c++20 -mavx2 -I./include examples/print.cpp
```

Notes:

- `fill_gamma(..., alpha)` currently assumes `alpha >= 1`.
- `fill_bernoulli` writes `double` outputs (`0.0` or `1.0`), not `bool`.
- For small scalar-style draws, `zorro::Xoshiro256pp` also works as a standard
  `UniformRandomBitGenerator`.

```cpp
#include "zorro/zorro.hpp"
#include <random>

int main() {
    zorro::Xoshiro256pp rng(42);
    std::uniform_int_distribution<int> die(1, 6);
    return die(rng);
}
```

### Thread-local convenience

If you do not want to thread an RNG object through your code, use the built-in
thread-local singleton:

```cpp
#include "zorro/zorro.hpp"
#include <vector>

int main() {
    std::vector<double> buf(4096);

    zorro::reseed(1234);
    zorro::get_rng().fill_normal(buf.data(), buf.size());
}
```

## Repository examples

### High-dimensional example

The repo also includes a standalone example that simulates a high-dimensional
factor model with heavy-tailed common shocks. It exposes a small swappable
backend interface so the same workload can run on either `std` or `zorro`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target high_dimensional_random_variables
./build/high_dimensional_random_variables --backend both
```

Useful options:

- `--backend std|zorro|both` to switch the RNG implementation at runtime.
- `--dimension N` to change the random-vector dimension.
- `--paths N` and `--repetitions N` to make the run shorter or longer.
- `--factors N` and `--degrees-of-freedom X` to change the factor model.

### Importance sampling example

There is also a rare-event example for importance sampling. It estimates a
portfolio tail probability under a correlated Gaussian factor model using both
plain Monte Carlo and a mean-shift importance proposal, again with a swappable
`std`/`zorro` backend.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target importance_sampling
./build/importance_sampling --backend both
```

Useful options:

- `--rare-z X` sets the loss threshold in standard deviations above the mean.
- `--dimension N`, `--factors N`, `--paths N`, and `--repetitions N` scale the workload.
- `--backend std|zorro|both` switches the RNG implementation at runtime.

## SIMD dispatch

The SIMD tier is selected at compile time based on which instruction sets are
enabled. In the public header, AVX-512 currently accelerates the integer-heavy
kernels (`uniform`, `bernoulli`), while the log-heavy transforms use the AVX2
or portable implementations.

| Compile flags | Engine | Uniform throughput |
|---|---|---|
| `-mavx512f -mavx512vl -mavx512dq` | 16-wide AVX-512 (2x8 interleaved) | ~3300 M/s |
| `-mavx2` | 8-wide AVX2 (2x4 interleaved) | ~2900 M/s |
| (neither) | 4-wide portable (compiler auto-vec) | ~1200 M/s |

For Zen 4-style CPUs, this keeps the public library on the simpler and usually
more defensible side of the tradeoff: AVX-512 where it helps clearly, AVX2 or
portable code for the heavier transcendental paths.

### Log-heavy transforms

The public header no longer depends on `libmvec`.

- `fill_exponential()` uses the validated AVX2 fast `-log(1-u)` approximation
  when AVX2 is available.
- `fill_normal()`, `fill_gamma()`, and `fill_student_t()` use SIMD for the RNG
  core and scalar `std::log`/`std::sqrt` in their acceptance or transform
  steps.
- The benchmark harness still includes `libmvec`-backed `veclog` and
  `vecpolar` kernels for comparison.

### Lower-level API

The header also provides building-block types for direct use:

- **`zorro::Xoshiro256pp`** -- scalar engine, satisfies `UniformRandomBitGenerator`
  (drop-in for `std::uniform_int_distribution(rng)` etc.)
- **`zorro::Xoshiro256pp_x2`** / **`zorro::Xoshiro256pp_x4_portable`** -- portable
  multi-lane engines (SoA layout, compiler auto-vectorizes)
- **`zorro::splitmix64`**, **`zorro::bits_to_01`**, **`zorro::bits_to_pm1`** --
  seed expansion and bit-to-float conversion utilities
- **`zorro::get_rng()`** / **`zorro::reseed(seed)`** -- thread-local `Rng` singleton

## Benchmarks

The benchmark target compares `2^20` samples across eight distribution suites
and is configured for C++23.

- **Uniform(0, 1)** -- scalar, x2, x4 portable, x4/x8 AVX2, x8/x16 AVX-512
- **Normal(0, 1)** -- polar, batched, veclog, vecpolar, and Box-Muller approximation variants (AVX2 and AVX-512)
- **Exponential(1)** -- scalar log, AVX2 libmvec, AVX2 fastlog approximation, AVX-512 libmvec
- **Bernoulli(0.3)** -- naive, integer-threshold AVX2, native ucmp AVX-512
- **Bernoulli(0.3/0.5) -> uint8_t** -- compact 1-byte output with bit-unpack special case for p=0.5
- **Gamma(2, 1)** -- scalar fused, x8 AVX2 fused/decoupled/full
- **Student's t(5)** -- scalar fused, x8 AVX2 fused/decoupled/fast

Each suite also includes
[Stephan Friedl's](https://stephanfr.com/2021/08/17/serial-and-simd-implementation-of-the-xoroshiro256-random-number-generator/)
handwritten AVX2 `Xoshiro256PlusSIMD` variants where applicable.

A literate programming walkthrough of the core AVX2 4x2 loop lives in
[docs/xoshiro256pp_avx_4x2.md](docs/xoshiro256pp_avx_4x2.md).

### Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

For local machine-specific benchmark runs:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRNG_BENCH_ENABLE_NATIVE=ON
cmake --build build -j
```

CMake options:

| Option | Default | Description |
|---|---|---|
| `RNG_BENCH_ENABLE_NATIVE` | `OFF` | Compile with `-march=native -mtune=native` |
| `RNG_BENCH_ENABLE_AVX512` | auto-detected | AVX-512F/VL/DQ kernels. Auto-enabled when the host CPU supports AVX-512 (compile-and-run test), disabled otherwise. |
| `RNG_BENCH_ENABLE_STEPHANFR_AVX2` | auto-detected | Stephan Friedl's AVX2 comparison. Auto-enabled when the compiler supports `-mavx2`. |
| `RNG_BENCH_MARCH_FLAGS` | `""` | Extra SIMD override flags appended after `-march=native` (e.g. `-mno-avx512f`). |

### Run

```bash
./build/benchmark_distributions
```

The benchmark uses a fixed seed and reports the best, median, and mean
wall-clock time plus derived throughput in samples/second.

### Dieharder

If `dieharder` is installed, you can run a targeted battery against the raw
`xoshiro256++` bitstreams exported by the library:

```bash
./benchmarks/run_dieharder.sh
```

Useful options:

- `--stream scalar|x2|x4|all` to select which generator layout to test.
- `--out-dir DIR` to keep the raw `dieharder` reports in a specific folder.

The script builds `dieharder_stream` if needed and automatically reruns any
`WEAK` or `FAILED` result in ambiguity-resolution mode (`-k 2 -Y 1`).

### TestU01

If TestU01 is installed, the project also exposes a direct runner linked
against `unif01`/`bbattery`:

```bash
./benchmarks/run_testu01.sh
```

Useful options:

- `--stream scalar|x2|x4|all` to select which generator layout to test.
- `--battery smallcrush|crush|bigcrush` to pick the TestU01 battery.
- `--out-dir DIR` to keep the raw TestU01 reports in a specific folder.

The default is `SmallCrush`, which is the practical quick check. `Crush` and
especially `BigCrush` are much more expensive.

### AWS benchmarking

`aws/bench.sh` launches spot instances, uploads the source, builds with
both GCC and Clang across multiple SIMD levels (native, no-avx512, no-avx2,
no-avx), runs `benchmark_distributions`, and runs the transform-quality tests
(`normal_transform_tests` and `exponential_transform_tests`) where AVX2 is
available. Results are saved under `aws/results/`.

```bash
./aws/bench.sh                                    # defaults: c6i, c7i, c7a
./aws/bench.sh --instances c7i.xlarge,c7a.xlarge  # custom instance types
```

## Findings

Results from `aws/bench.sh` on March 24, 2026 (GCC 15, native SIMD, best ms).
The latest successful runs in `aws/results/` are:

- `c8a.large` -- AMD EPYC 9R45
- `c8i.large` -- Intel Xeon 6975P-C

### Uniform generation

| Kernel | c8a.large (AMD) | c8i.large (Intel) |
|---|---|---|
| scalar | 1.024 / 1024 M/s | 1.152 / 910 M/s |
| x8 AVX2 (2x4) | 0.258 / 4072 M/s | 0.320 / 3275 M/s |
| x16 AVX-512 (2x8) | 0.130 / 8083 M/s | 0.284 / 3690 M/s |

On the current AMD `c8a.large` host, uniform generation scales extremely well
with AVX-512: `x16` is almost 2x the `x8` AVX2 path. On the Intel `c8i.large`
host, AVX-512 still wins, but the gain is more modest.

### Normal generation

| Kernel | c8a.large (AMD) | c8i.large (Intel) |
|---|---|---|
| x8 AVX2 + box-muller fullapprox | 1.117 / 939 M/s | 1.287 / 815 M/s |
| x16 AVX-512 + box-muller fullapprox | 1.123 / 934 M/s | 1.007 / 1041 M/s |
| x8 AVX2 + vecpolar | 2.581 / 406 M/s | 2.757 / 380 M/s |
| x16 AVX-512 + vecpolar | 2.673 / 392 M/s | 1.899 / 552 M/s |

The new full-approximation Box-Muller path is now the best normal kernel in the
tree. On Intel, widening it to `x16` AVX-512 gives another ~1.28x over the AVX2
fullapprox path. On the AMD `c8a.large` run, the `x16` AVX-512 fullapprox row
is effectively a tie with AVX2 rather than an improvement.

AVX-512 vecpolar still helps on Intel thanks to native 512-bit sqrt/div and
hardware compress-store, but it is no longer the strongest normal path. On the
current AMD host, vecpolar does not improve with AVX-512 and is slightly slower
than the AVX2 version in these runs.

### Exponential and Bernoulli

| Kernel | c8a.large (AMD) | c8i.large (Intel) |
|---|---|---|
| Exp x8 AVX2 libmvec | 1.335 / 786 M/s | 1.624 / 646 M/s |
| Exp x8 AVX2 fastlog | 1.194 / 878 M/s | 1.217 / 862 M/s |
| Exp x16 AVX-512 libmvec | 0.850 / 1233 M/s | 1.180 / 889 M/s |
| Bernoulli x8 AVX2 fast | 0.236 / 4435 M/s | 0.310 / 3383 M/s |
| Bernoulli x16 AVX-512 ucmp | 0.122 / 8585 M/s | 0.281 / 3735 M/s |

For exponential, the validated AVX2 fastlog approximation still beats the
exact AVX2 `libmvec` path on both current AWS machines. Intel still benefits
slightly from the wider AVX-512 `libmvec` path, but the biggest surprise in the
latest run is AMD `c8a.large`, where AVX-512 `libmvec` jumps to 1233 M/s and
overtakes both AVX2 variants by a wide margin. The public
`zorro::Rng::fill_exponential()` path still sticks to the AVX2 fastlog
implementation; the exact `libmvec` variants remain benchmark-only.

AVX-512 Bernoulli uses native `_mm512_cmp_epu64_mask` to eliminate the
sign-flip workaround required by AVX2's signed-only `_mm256_cmpgt_epi64`.

### Bernoulli output formats and the p=0.5 special case

The Bernoulli kernels support two output formats:

- **double** (8 bytes/sample) -- 0.0 or 1.0, compatible with the other distribution suites.
- **uint8_t** (1 byte/sample) -- 0 or 1, 8x less memory traffic.

For **p = 0.5**, every bit of the raw xoshiro256++ output is an independent
Bernoulli trial. Instead of generating a uniform double and comparing, we
unpack individual bits directly into the output -- one RNG call produces
208 samples (52 quality bits x 4 SIMD lanes) instead of 4.

The low 12 bits of each uint64 are discarded, matching the uniform path's
`>> 12` and avoiding the weakest bits of the xoshiro256++ output function.

Local AVX2 results (2^20 samples, best of 21 runs):

| Kernel | Output | Best ms | Throughput |
|---|---|---|---|
| naive (uniform + cmp) | double | 0.349 | 3.0 G/s |
| fast (int threshold) | double | 0.312 | 3.4 G/s |
| **bit-unpack (p=0.5)** | **double** | **0.176** | **6.0 G/s** |
| naive (uniform + cmp) | uint8_t | 0.550 | 1.9 G/s |
| fast (int threshold) | uint8_t | 0.336 | 3.1 G/s |
| **bit-unpack (p=0.5)** | **uint8_t** | **0.050** | **21.2 G/s** |

The uint8_t bit-unpack path achieves 21 billion samples/second -- roughly
7 samples per clock cycle on a 3 GHz machine. At this speed, the RNG core
itself is no longer the bottleneck; the limit is the byte-scatter from bits
to the output buffer.

### Fused vs decoupled generation

**Gamma(2, 1) -- full/fused still win decisively**

| Kernel | c8a.large (AMD) | c8i.large (Intel) |
|---|---|---|
| scalar fused | 61 M/s | 58 M/s |
| x8 AVX2 fused | 148 M/s | 144 M/s |
| x8+x4 AVX2 full (vectorized MT) | 160 M/s | 144 M/s |
| x8 AVX2 decoupled | 90 M/s | 77 M/s |

Fused keeps the PRNG state registers hot and feeds the Marsaglia-Tsang
acceptance loop directly from a 64-sample L1-resident buffer. Decoupled
materialises intermediate normals and uniforms to the heap, paying full memory
bandwidth for no gain.

**Student's t(5) -- decoupled wins**

| Kernel | c8a.large (AMD) | c8i.large (Intel) |
|---|---|---|
| scalar fused | 34 M/s | 34 M/s |
| x8 AVX2 fused | 48 M/s | 45 M/s |
| x8 AVX2 decoupled | 77 M/s | 72 M/s |
| x8+x4 AVX2 fast | 77 M/s | 73 M/s |

For a compound distribution the fused variant forces a scalar Gamma loop into
the hot path. Decoupled lets each sub-distribution run its own best kernel
independently; the memory-traffic cost is more than recovered.

**Summary**

- Fused integration is the right default when the distribution is
  self-contained and fits in L1 (Gamma: about 1.6-1.9x over decoupled in the
  latest AWS run).
- Decoupled is better for compound distributions where one sub-computation
  would otherwise serialize an otherwise-vectorised pipeline (Student-t: about
  1.5-1.6x over fused).
- AVX-512 is not extended to Gamma or Student-t: their bottleneck is the
  scalar Marsaglia-Tsang acceptance loop, not the PRNG or vectorized math.

### AMD vs Intel: when AVX-512 helps and when it hurts

On the latest AWS pair, AVX-512 helps most clearly in the integer-heavy kernels
(uniform, Bernoulli) and in Intel's fullapprox normal path. The benchmark-only
vecpolar kernels remain a useful counterexample: Intel still benefits from
AVX-512 there, while the current AMD run does not.

That is the main reason the public library now keeps a narrower AVX-512 scope:
use it where it wins clearly, and avoid carrying extra complexity into the
public header for the transform paths that no longer need it.

## Third-party code

The handwritten SIMD comparison uses a vendored snapshot of Stephan Friedl's `Xoshiro256PlusSIMD` headers under `third_party/stephanfr_xoshiro256plussimd`, taken from upstream commit `29b30821f8f59b6106462c03d1d0225c93c4545d`. The upstream license is preserved in `third_party/stephanfr_xoshiro256plussimd/LICENSE`.
