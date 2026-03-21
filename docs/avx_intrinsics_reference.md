# AVX Intrinsics Reference

A glossary of every Intel SIMD intrinsic used in the Zorro codebase, grouped
by operation category. Each entry gives the signature, what it computes, and
the latency/throughput class on a typical recent core (Zen 4 / Sapphire
Rapids).

Naming convention: `_mm256_` operates on 256-bit registers (AVX2, 4×i64 or
4×f64), `_mm512_` on 512-bit registers (AVX-512, 8×i64 or 8×f64), and
`_mm_` on 128-bit registers (SSE/AVX, 2×i64 or 2×f64).

---

## Load / Store

| Intrinsic | Operation |
|---|---|
| `_mm256_load_si256(p)` | Aligned 256-bit integer load from `p` (must be 32-byte aligned). |
| `_mm256_load_pd(p)` | Aligned 256-bit load of 4 doubles from `p`. |
| `_mm256_store_si256(p, a)` | Aligned 256-bit integer store to `p`. |
| `_mm256_store_pd(p, a)` | Aligned 256-bit store of 4 doubles to `p`. |
| `_mm256_storeu_si256(p, a)` | Unaligned 256-bit integer store. No alignment requirement; same throughput as aligned on modern µarchs. |
| `_mm256_storeu_pd(p, a)` | Unaligned 256-bit store of 4 doubles. |
| `_mm512_load_si512(p)` | Aligned 512-bit integer load (64-byte aligned). |
| `_mm512_store_si512(p, a)` | Aligned 512-bit integer store. |
| `_mm512_store_pd(p, a)` | Aligned 512-bit store of 8 doubles. |
| `_mm512_storeu_pd(p, a)` | Unaligned 512-bit store of 8 doubles. |
| `_mm512_mask_compressstoreu_pd(p, k, a)` | Store only the lanes selected by mask `k`, packed contiguously into memory starting at `p`. Used by vecpolar to write only accepted pairs. High latency on AMD Zen 4 (serialises on the 256-bit datapath). |
| `_mm_storel_epi64(p, a)` | Store the low 64 bits of a 128-bit register to memory. |

---

## Integer Arithmetic

| Intrinsic | Operation |
|---|---|
| `_mm256_add_epi64(a, b)` | Lane-wise 64-bit integer addition: `a[i] + b[i]`. Core of the xoshiro256++ output scrambler (`s[0] + s[3]`). 1 cycle latency. |
| `_mm256_slli_epi64(a, imm)` | Lane-wise left shift by immediate: `a[i] << imm`. |
| `_mm256_srli_epi64(a, imm)` | Lane-wise logical right shift by immediate: `a[i] >> imm`. Used to extract the 52-bit mantissa for uniform-to-double conversion. |
| `_mm256_mul_epu32(a, b)` | Multiply the low 32-bit unsigned integers of each 64-bit lane, producing 64-bit results. Used in Stephan Friedl's implementation. |
| `_mm256_extract_epi32(a, imm)` | Extract a single 32-bit integer from the specified position. |
| `_mm512_add_epi64(a, b)` | 512-bit lane-wise 64-bit addition (8 lanes). |
| `_mm512_slli_epi64(a, imm)` | 512-bit lane-wise left shift by immediate. |
| `_mm512_srli_epi64(a, imm)` | 512-bit lane-wise logical right shift by immediate. |

---

## Integer Rotate

| Intrinsic | Operation |
|---|---|
| `_mm256_rol_epi64(a, imm)` | Lane-wise 64-bit left rotate by immediate. **Requires AVX-512VL** (available on 256-bit registers when the CPU supports AVX-512VL+DQ). Replaces the 3-instruction shift-or-shift emulation. |
| `_mm512_rol_epi64(a, imm)` | 512-bit lane-wise 64-bit left rotate by immediate. **Requires AVX-512F.** Single instruction replacing the `(x << n) \| (x >> (64-n))` pattern. |

Note: on pure AVX2 (no AVX-512VL), `rotl` is emulated as
`_mm256_or_si256(_mm256_slli_epi64(x, n), _mm256_srli_epi64(x, 64-n))`.

---

## Bitwise Logic

| Intrinsic | Operation |
|---|---|
| `_mm256_xor_si256(a, b)` | 256-bit bitwise XOR. Used heavily in the xoshiro256 state update. |
| `_mm256_or_si256(a, b)` | 256-bit bitwise OR. Used in rotate emulation and uniform-to-double bit manipulation. |
| `_mm256_and_si256(a, b)` | 256-bit bitwise AND. |
| `_mm256_and_pd(a, b)` | 256-bit bitwise AND on double registers. Used to mask sign bits or apply conditions in FP domain. |
| `_mm256_or_pd(a, b)` | 256-bit bitwise OR on double registers. |
| `_mm512_xor_si512(a, b)` | 512-bit bitwise XOR. |
| `_mm512_or_si512(a, b)` | 512-bit bitwise OR. |

---

## Floating-Point Arithmetic

| Intrinsic | Operation |
|---|---|
| `_mm256_add_pd(a, b)` | Lane-wise double addition. |
| `_mm256_sub_pd(a, b)` | Lane-wise double subtraction. Used in `[1,2) - 1.0 → [0,1)` uniform conversion. |
| `_mm256_mul_pd(a, b)` | Lane-wise double multiplication. |
| `_mm256_div_pd(a, b)` | Lane-wise double division. Higher latency (~13–14 cycles); used in the polar method. |
| `_mm256_sqrt_pd(a)` | Lane-wise double square root. ~13–19 cycle latency; used in polar-method and Student-t kernels. |
| `_mm512_add_pd(a, b)` | 512-bit lane-wise double addition (8 lanes). |
| `_mm512_sub_pd(a, b)` | 512-bit lane-wise double subtraction. |
| `_mm512_mul_pd(a, b)` | 512-bit lane-wise double multiplication. |
| `_mm512_div_pd(a, b)` | 512-bit lane-wise double division. On AMD Zen 4, each 512-bit div splits into two 256-bit micro-ops that serialize on the divider — making this ~2× slower than the AVX2 equivalent. |
| `_mm512_sqrt_pd(a)` | 512-bit lane-wise double square root. Same AMD serialisation issue as `_mm512_div_pd`. |

---

## Comparison

| Intrinsic | Operation |
|---|---|
| `_mm256_cmp_pd(a, b, imm)` | Lane-wise double comparison; `imm` selects the predicate (e.g. `_CMP_LT_OQ` for ordered less-than). Returns a 256-bit mask with all-ones or all-zeros per lane. |
| `_mm256_cmpeq_epi64(a, b)` | Lane-wise 64-bit integer equality. Returns all-ones per lane on match. |
| `_mm256_cmpgt_epi64(a, b)` | Lane-wise **signed** 64-bit integer greater-than. The sign interpretation requires a workaround (XOR with sign bit) when comparing unsigned thresholds — see the Bernoulli AVX2 kernel. |
| `_mm512_cmp_pd_mask(a, b, imm)` | 512-bit double comparison returning an 8-bit `__mmask8`. More compact than the 256-bit version which returns a full vector. |
| `_mm512_cmp_epu64_mask(a, b, imm)` | Lane-wise **unsigned** 64-bit integer comparison returning `__mmask8`. Eliminates the sign-flip workaround that AVX2's `_mm256_cmpgt_epi64` requires. |

---

## Mask and Blend

| Intrinsic | Operation |
|---|---|
| `_mm256_movemask_pd(a)` | Extract the sign bit of each of 4 doubles into a 4-bit integer. Used to convert vector comparison results to scalar branch conditions. |
| `_mm256_blendv_pd(a, b, mask)` | Lane-wise conditional select: for each lane, pick `b` if the mask's sign bit is set, else `a`. Used for conditional assignment without branches. |
| `_mm512_mask_blend_pd(k, a, b)` | AVX-512 masked blend: for each bit in `k`, pick from `b` if set, `a` if clear. |
| `_mm512_maskz_mov_pd(k, a)` | Zero-masked move: lanes where `k` is 0 become 0.0, lanes where `k` is 1 keep the value from `a`. |

---

## Shuffle and Permute

| Intrinsic | Operation |
|---|---|
| `_mm256_unpacklo_pd(a, b)` | Interleave low doubles from each 128-bit half: `{a[0], b[0], a[2], b[2]}`. Used to deinterleave paired polar-method outputs. |
| `_mm256_unpackhi_pd(a, b)` | Interleave high doubles from each 128-bit half: `{a[1], b[1], a[3], b[3]}`. |
| `_mm256_permute2f128_pd(a, b, imm)` | Select 128-bit halves from `a` and `b` to form a new 256-bit register. Cross-lane permute for rearranging deinterleaved pairs. |
| `_mm256_shuffle_epi8(a, b)` | Byte-level shuffle within each 128-bit lane, controlled by `b`. |
| `_mm256_extractf128_pd(a, imm)` | Extract the low (`imm=0`) or high (`imm=1`) 128-bit half of a 256-bit double register. |
| `_mm256_castsi256_si128(a)` | Extract the low 128 bits of a 256-bit integer register. Zero-cost (register aliasing). |
| `_mm_unpackhi_pd(a, b)` | 128-bit interleave high doubles: `{a[1], b[1]}`. |

---

## Type Reinterpretation (Zero-Cost Casts)

These are compiler hints that reinterpret a register's type without emitting
any instructions. The bits do not change — only the type visible to
subsequent intrinsics.

| Intrinsic | Operation |
|---|---|
| `_mm256_castsi256_pd(a)` | Reinterpret 256-bit integer register as 4 doubles. Central to the bit-manipulation uniform-to-double conversion. |
| `_mm256_castpd256_pd128(a)` | Extract the low 128-bit half of a 256-bit double register (type cast, not a shuffle). |
| `_mm512_castsi512_pd(a)` | Reinterpret 512-bit integer register as 8 doubles. |

---

## Broadcast / Set

| Intrinsic | Operation |
|---|---|
| `_mm256_set1_epi64x(x)` | Broadcast a 64-bit integer to all 4 lanes. |
| `_mm256_set1_pd(x)` | Broadcast a double to all 4 lanes. |
| `_mm256_set1_epi8(x)` | Broadcast a byte to all 32 byte positions. |
| `_mm256_set_epi64x(d, c, b, a)` | Set 4 lanes to specific 64-bit values (highest lane first). |
| `_mm256_set_epi8(...)` | Set all 32 bytes to specific values. |
| `_mm256_setzero_pd()` | Return a 256-bit register with all bits zero (4 doubles of 0.0). |
| `_mm256_min_epu8(a, b)` | Lane-wise unsigned byte minimum. Used in the Stephan Friedl implementation for byte-level operations. |
| `_mm512_set1_epi64(x)` | Broadcast a 64-bit integer to all 8 lanes. |
| `_mm512_set1_pd(x)` | Broadcast a double to all 8 lanes. |
| `_mm512_setzero_pd()` | Return a 512-bit register of all zeros. |

---

## Scalar Extraction

| Intrinsic | Operation |
|---|---|
| `_mm_cvtsd_f64(a)` | Extract the lowest double from a 128-bit register as a scalar `double`. |

---

## Instruction Set Requirements

| ISA extension | Intrinsics it unlocks |
|---|---|
| **AVX2** | All `_mm256_*` integer operations (`add_epi64`, `slli`, `srli`, `xor_si256`, `cmpgt_epi64`, etc.) |
| **AVX** | `_mm256_*` floating-point operations (`add_pd`, `cmp_pd`, `sqrt_pd`, etc.), permute, blend |
| **AVX-512F** | All `_mm512_*` operations, `__mmask8` types |
| **AVX-512VL** | `_mm256_rol_epi64` — 512-bit rotate instruction on 256-bit registers |
| **AVX-512DQ** | `_mm512_cmp_epu64_mask` — unsigned integer comparison |
| **SSE2** | `_mm_storel_epi64`, `_mm_unpackhi_pd`, `_mm_cvtsd_f64` |
