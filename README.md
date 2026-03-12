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

## Findings

### Normal generation

vecpolar (x8 AVX2 dual-stream + blended `_ZGVdN4v_log`) is the fastest N(0,1)
path at ~292 M/s (Zorro) and ~303 M/s (stephanfr). The earlier extract and
batched variants top out around 162 M/s. The veclog batch variant sits in
between at ~263 M/s. The vectorized transform — not the wider PRNG — is the
dominant factor.

### Fused vs decoupled generation

The thesis "tight integration of raw bit generation with the follow-up
distribution algorithm is an advantage" holds, but conditionally:

**Gamma(2, 1) — fused wins decisively**

| Kernel | M/s |
|---|---|
| scalar fused | 47 |
| x8 AVX2 fused | 117 |
| x8+x4 AVX2 full (vectorized MT) | 116 |
| x8 AVX2 decoupled | 45 |

Fused keeps the PRNG state registers hot and feeds the Marsaglia-Tsang
acceptance loop directly from a 64-sample L1-resident buffer. Decoupled
materialises ~320 MB of intermediate normals and uniforms to the heap, paying
full memory bandwidth for no gain. A fully vectorised MT acceptance loop
(4-wide AVX2 uniforms + unconditional `veclog` pair) does not improve on fused:
the fast-accept path fires ~80 % of the time, the scalar loop was never the
bottleneck, and the unconditional `veclog` overhead on the slow path absorbs
any benefit from eliminating serial dependency chains.

**Student's t(5) — decoupled wins**

| Kernel | M/s |
|---|---|
| scalar fused | 27 |
| x8 AVX2 fused | 35 |
| x8 AVX2 decoupled (vecpolar Z + gamma_fused V) | 39 |
| x8+x4 AVX2 fast (vecpolar Z + gamma_full V) | 39 |

For a compound distribution the fused variant forces a scalar Gamma loop into
the hot path immediately after the AVX2 normal generation. Decoupled lets each
sub-distribution run its own best kernel independently; the memory-traffic cost
(two heap-write + two heap-read passes) is more than recovered. The choice of
Gamma kernel (fused vs full) makes no difference at this level — both hit the
same throughput ceiling set by the two-pass memory traffic.

**Summary**

- Fused integration is the right default when the distribution is
  self-contained and fits in L1 (Gamma: 2.5× over decoupled).
- Decoupled is better for compound distributions where one sub-computation
  would otherwise serialize an otherwise-vectorised pipeline (Student-t: 1.1×
  over fused, with headroom limited by memory bandwidth rather than compute).

## Third-party code

The handwritten SIMD comparison uses a vendored snapshot of Stephan Friedl's `Xoshiro256PlusSIMD` headers under `third_party/stephanfr_xoshiro256plussimd`, taken from upstream commit `29b30821f8f59b6106462c03d1d0225c93c4545d`. The upstream license is preserved in `third_party/stephanfr_xoshiro256plussimd/LICENSE`.
