# zorro

Zorro is a header-only `xoshiro256++` library with automatic SIMD dispatch.
Add one header and get high-throughput random generation for uniform, normal,
exponential, Bernoulli, gamma, and Student's t distributions.

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

Results from `aws/bench.sh` (GCC 15, native SIMD, best ms).

### Uniform generation

| Kernel | c7a (Zen 4) | c7i (SPR) | c6i (Ice Lake) |
|---|---|---|---|
| scalar | 1.072 / 978 M/s | 1.133 / 925 M/s | 1.498 / 700 M/s |
| x8 AVX2 (2x4) | 0.268 / 3917 M/s | 0.314 / 3344 M/s | 0.342 / 3070 M/s |
| x16 AVX-512 (2x8) | 0.274 / 3832 M/s | 0.289 / 3626 M/s | 0.281 / 3734 M/s |

AVX-512 uniform is a modest-to-clear win on Intel (~8-22% over AVX2) but roughly breaks
even on AMD, where integer/bitwise 512-bit ops split into 256-bit micro-op
pairs at full throughput.

### Normal generation

| Kernel | c7a (Zen 4) | c7i (SPR) | c6i (Ice Lake) |
|---|---|---|---|
| x8 AVX2 + box-muller fullapprox | 1.545 / 683 M/s | 1.281 / 835 M/s | 2.049 / 524 M/s |
| x16 AVX-512 + box-muller fullapprox | 1.557 / 678 M/s | 1.025 / 1023 M/s | 1.536 / 683 M/s |
| x8 AVX2 + vecpolar | 3.026 / 347 M/s | 2.792 / 376 M/s | 3.726 / 281 M/s |
| x16 AVX-512 + vecpolar | 3.027 / 346 M/s | 1.978 / 530 M/s | 2.727 / 385 M/s |

The new full-approximation Box-Muller path is now the best normal kernel in the
tree. On Intel, widening it to `x16` AVX-512 gives another ~1.3x over the AVX2
fullapprox path. On AMD Zen 4, the AVX-512 fullapprox row is effectively a tie
with AVX2 because the benchmark kernel follows the same runtime policy as the
AVX-512 vecpolar experiment and falls back to the AVX2 implementation on AMD.

AVX-512 vecpolar still helps on Intel thanks to native 512-bit sqrt/div and
hardware compress-store, but it is no longer the strongest normal path. On AMD
Zen 4, vecpolar also falls back to AVX2 at runtime (CPUID vendor check) because
512-bit sqrt, div, and `_mm512_mask_compressstoreu_pd` serialize on the 256-bit
datapath.

### Exponential and Bernoulli

| Kernel | c7a (Zen 4) | c7i (SPR) | c6i (Ice Lake) |
|---|---|---|---|
| Exp x8 AVX2 libmvec | 1.714 / 612 M/s | 1.557 / 674 M/s | 2.372 / 442 M/s |
| Exp x8 AVX2 fastlog | 1.412 / 743 M/s | 1.246 / 841 M/s | 1.754 / 598 M/s |
| Exp x16 AVX-512 libmvec | 1.335 / 785 M/s | 1.003 / 1045 M/s | 1.403 / 748 M/s |
| Bernoulli x8 AVX2 fast | 0.269 / 3893 M/s | 0.357 / 2934 M/s | 0.360 / 2912 M/s |
| Bernoulli x16 AVX-512 ucmp | 0.247 / 4247 M/s | 0.325 / 3229 M/s | 0.363 / 2890 M/s |

For exponential, the validated AVX2 fastlog approximation beats the exact AVX2
libmvec path on all three AWS machines. Intel still benefits from the wider
`_ZGVeN8v_log` AVX-512 path, while Zen 4 sees only a small additional gain.
The public `zorro::Rng::fill_exponential()` path now sticks to the AVX2
fastlog implementation; the exact `libmvec` variants remain benchmark-only.

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

**Gamma(2, 1) -- fused wins decisively**

| Kernel | c7a (Zen 4) | c7i (SPR) |
|---|---|---|
| scalar fused | 48 M/s | 47 M/s |
| x8 AVX2 fused | 125 M/s | 127 M/s |
| x8+x4 AVX2 full (vectorized MT) | 133 M/s | 126 M/s |
| x8 AVX2 decoupled | 60 M/s | 66 M/s |

Fused keeps the PRNG state registers hot and feeds the Marsaglia-Tsang
acceptance loop directly from a 64-sample L1-resident buffer. Decoupled
materialises intermediate normals and uniforms to the heap, paying full memory
bandwidth for no gain.

**Student's t(5) -- decoupled wins**

| Kernel | c7a (Zen 4) | c7i (SPR) |
|---|---|---|
| scalar fused | 26 M/s | 29 M/s |
| x8 AVX2 fused | 35 M/s | 38 M/s |
| x8 AVX2 decoupled | 62 M/s | 61 M/s |
| x8+x4 AVX2 fast | 64 M/s | 62 M/s |

For a compound distribution the fused variant forces a scalar Gamma loop into
the hot path. Decoupled lets each sub-distribution run its own best kernel
independently; the memory-traffic cost is more than recovered.

**Summary**

- Fused integration is the right default when the distribution is
  self-contained and fits in L1 (Gamma: 2x over decoupled).
- Decoupled is better for compound distributions where one sub-computation
  would otherwise serialize an otherwise-vectorised pipeline (Student-t: 1.7x
  over fused).
- AVX-512 is not extended to Gamma or Student-t: their bottleneck is the
  scalar Marsaglia-Tsang acceptance loop, not the PRNG or vectorized math.

### AMD vs Intel: when AVX-512 helps and when it hurts

AMD Zen 4 implements AVX-512 by splitting every 512-bit op into two 256-bit
micro-ops. This is transparent for integer-heavy kernels (uniform, Bernoulli)
where both halves pipeline through the ALUs at full throughput. But for
FP-math-heavy kernels with sqrt, div, and compress-store (vecpolar), the two
halves serialize on the single divider unit, making 512-bit strictly worse
than 256-bit.

The `cpu_is_amd()` check in the vecpolar kernel detects this at runtime and
routes AMD hardware to the AVX2 path automatically.

## Third-party code

The handwritten SIMD comparison uses a vendored snapshot of Stephan Friedl's `Xoshiro256PlusSIMD` headers under `third_party/stephanfr_xoshiro256plussimd`, taken from upstream commit `29b30821f8f59b6106462c03d1d0225c93c4545d`. The upstream license is preserved in `third_party/stephanfr_xoshiro256plussimd/LICENSE`.
