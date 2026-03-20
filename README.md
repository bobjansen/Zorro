# zorro

Zorro is a standalone `xoshiro256++` experiment: scalar, 2-lane, 4-lane, and
wider SIMD-oriented layouts (AVX2 and AVX-512) plus a benchmark harness for
multiple distributions.

The benchmark target is configured for C++23.

The core public API lives in `include/zorro/zorro.hpp` under the `zorro`
namespace. The root-level `zorro.hpp` and `rng.hpp` headers are compatibility
shims during the cleanup.

The benchmark target compares `2^20` samples across eight distribution suites:

- **Uniform(0, 1)** — scalar, x2, x4 portable, x4/x8 AVX2, x8/x16 AVX-512
- **Normal(0, 1)** — polar, batched, veclog, vecpolar (AVX2 and AVX-512)
- **Exponential(1)** — scalar log, veclog AVX2, veclog AVX-512
- **Bernoulli(0.3)** — naive, integer-threshold AVX2, native ucmp AVX-512
- **Bernoulli(0.3/0.5) → uint8_t** — compact 1-byte output with bit-unpack special case for p=0.5
- **Gamma(2, 1)** — scalar fused, x8 AVX2 fused/decoupled/full
- **Student's t(5)** — scalar fused, x8 AVX2 fused/decoupled/fast

Each suite also includes Stephan Friedl's handwritten AVX2 `Xoshiro256PlusSIMD`
variants where applicable.

A literate programming walkthrough of the core AVX2 4×2 loop lives in
[docs/xoshiro256pp_avx_4x2.md](docs/xoshiro256pp_avx_4x2.md).

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

CMake options:

| Option | Default | Description |
|---|---|---|
| `RNG_BENCH_ENABLE_NATIVE` | `OFF` | Compile with `-march=native -mtune=native` |
| `RNG_BENCH_ENABLE_AVX512` | auto-detected | AVX-512F/VL/DQ kernels. Auto-enabled when the host CPU supports AVX-512 (compile-and-run test), disabled otherwise. |
| `RNG_BENCH_ENABLE_STEPHANFR_AVX2` | auto-detected | Stephan Friedl's AVX2 comparison. Auto-enabled when the compiler supports `-mavx2`. |
| `RNG_BENCH_MARCH_FLAGS` | `""` | Extra SIMD override flags appended after `-march=native` (e.g. `-mno-avx512f`). |

## Run

```bash
./build/benchmark_distributions
```

The benchmark uses a fixed seed and reports the best, median, and mean
wall-clock time plus derived throughput in samples/second.

### AWS benchmarking

`aws/bench.sh` launches spot instances, uploads the source, builds with
both GCC and Clang across multiple SIMD levels (native, no-avx512, no-avx2,
no-avx), and collects results. Results are saved under `aws/results/`.

```bash
./aws/bench.sh                                    # defaults: c6i, c7i, c7a
./aws/bench.sh --instances c7i.xlarge,c7a.xlarge  # custom instance types
```

## Findings

Results from `aws/bench.sh` (GCC 15, native SIMD, best ms).

### Uniform generation

| Kernel | c7a (Zen 4) | c7i (SPR) | c6i (Ice Lake) |
|---|---|---|---|
| scalar | 1.073 / 977 M/s | 1.262 / 831 M/s | 1.492 / 703 M/s |
| x8 AVX2 (2×4) | 0.268 / 3916 M/s | 0.362 / 2895 M/s | 0.348 / 3015 M/s |
| x16 AVX-512 (2×8) | 0.273 / 3840 M/s | 0.318 / 3299 M/s | 0.321 / 3263 M/s |

AVX-512 uniform is a modest win on Intel (~8-14% over AVX2) but roughly breaks
even on AMD, where integer/bitwise 512-bit ops split into 256-bit micro-op
pairs at full throughput.

### Normal generation

| Kernel | c7a (Zen 4) | c7i (SPR) | c6i (Ice Lake) |
|---|---|---|---|
| x8 AVX2 + vecpolar | 3.046 / 344 M/s | 3.022 / 347 M/s | 3.627 / 289 M/s |
| x16 AVX-512 + vecpolar | 3.052 / 344 M/s | 2.158 / 486 M/s | 2.643 / 397 M/s |

AVX-512 vecpolar gives a 1.3-1.4× speedup on Intel thanks to native 512-bit
sqrt/div and hardware compress-store. On AMD Zen 4, the kernel automatically
falls back to AVX2 vecpolar at runtime (CPUID vendor check) because 512-bit
sqrt, div, and `_mm512_mask_compressstoreu_pd` serialize on the 256-bit
datapath, making the AVX-512 version ~1.6× *slower* than AVX2 on AMD hardware.

### Exponential and Bernoulli

| Kernel | c7a (Zen 4) | c7i (SPR) | c6i (Ice Lake) |
|---|---|---|---|
| Exp x8 AVX2 veclog | 1.743 / 602 M/s | 2.123 / 494 M/s | 2.355 / 445 M/s |
| Exp x16 AVX-512 veclog | 1.322 / 793 M/s | 1.370 / 766 M/s | 1.374 / 763 M/s |
| Bernoulli x8 AVX2 fast | 0.269 / 3893 M/s | 0.357 / 2934 M/s | 0.360 / 2912 M/s |
| Bernoulli x16 AVX-512 ucmp | 0.247 / 4247 M/s | 0.325 / 3229 M/s | 0.363 / 2890 M/s |

AVX-512 exponential benefits strongly from the 8-wide `_ZGVeN8v_log` (+32-71%
over AVX2). AVX-512 Bernoulli uses native `_mm512_cmp_epu64_mask` to eliminate
the sign-flip workaround required by AVX2's signed-only `_mm256_cmpgt_epi64`.

### Bernoulli output formats and the p=0.5 special case

The Bernoulli kernels support two output formats:

- **double** (8 bytes/sample) — 0.0 or 1.0, compatible with the other distribution suites.
- **uint8_t** (1 byte/sample) — 0 or 1, 8x less memory traffic.

For **p = 0.5**, every bit of the raw xoshiro256++ output is an independent
Bernoulli trial. Instead of generating a uniform double and comparing, we
unpack individual bits directly into the output — one RNG call produces
208 samples (52 quality bits × 4 SIMD lanes) instead of 4.

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

### Fused vs decoupled generation

**Gamma(2, 1) — fused wins decisively**

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

**Student's t(5) — decoupled wins**

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
  self-contained and fits in L1 (Gamma: 2× over decoupled).
- Decoupled is better for compound distributions where one sub-computation
  would otherwise serialize an otherwise-vectorised pipeline (Student-t: 1.7×
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
