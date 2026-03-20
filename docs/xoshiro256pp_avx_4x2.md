# xoshiro256++ AVX2 4×2 Loop — A Literate Walk-Through

This document is a literate-programming presentation of the core AVX2 kernel
in `benchmarks/zorro_benchmark_kernels.cpp`. Every code fragment shown here is
extracted verbatim from that file. The goal is to tell the story in the order
a reader needs it — motivation first, details second — rather than in the order
the compiler requires.

---

## 1. The Algorithm: xoshiro256++

xoshiro256++ (pronounced "crosshatch-oh 256 plus-plus") is a pseudo-random
number generator designed by David Blackman and Sebastiano Vigna. Its state is
four 64-bit words `s[0..3]`, giving 256 bits total. Each call produces one
64-bit output and updates the state.

### 1.1 The output function

The ++ scrambler applies an extra rotate-and-add on top of the + scrambler:

```
output = rotl(s[0] + s[3], 23) + s[0]
```

The addition `s[0] + s[3]` and the final `+ s[0]` are non-linear (they carry
across bit boundaries), which breaks the linear structure of the underlying
recurrence at the output stage. Compared to xoshiro256+, which uses the simpler
`output = s[0] + s[3]`, the ++ scrambler has better statistical properties in
the low bits at the cost of two extra additions per sample.

### 1.2 The state recurrence

The state update is a linear recurrence over GF(2) — purely XORs and shifts,
no arithmetic — which is fast and amenable to SIMD:

```
t    = s[1] << 17
s[2] ^= s[0]
s[3] ^= s[1]
s[1] ^= s[2]
s[0] ^= s[3]
s[2] ^= t
s[3] = rotl(s[3], 45)
```

The ordering of the XOR assignments matters: each right-hand side uses the
*old* value of its operand, so `t` must be computed before `s[1]` is
overwritten, and the six XOR steps must execute in the order shown.

The scalar implementation in the source makes both concerns explicit:

```cpp
// One step of the xoshiro256 state recurrence (output-function agnostic).
inline void state_advance(std::uint64_t (&s)[4]) noexcept {
    const std::uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl64(s[3], 45);
}
```

---

## 2. The SIMD Opportunity

Every call to `state_advance` on one stream is completely independent of every
call on any other stream. If we run *k* streams in parallel, each step produces
*k* outputs simultaneously — with no cross-lane communication needed at all.

### 2.1 Structure-of-Arrays (SoA) layout

A natural AoS (Array-of-Structures) layout would store each stream's four
state words together. For SIMD we invert this: one register holds the same
word from *all* streams.

```
  s0  = [ lane0.s[0] | lane1.s[0] | lane2.s[0] | lane3.s[0] ]  (__m256i)
  s1  = [ lane0.s[1] | lane1.s[1] | lane2.s[1] | lane3.s[1] ]  (__m256i)
  s2  = [ lane0.s[2] | lane1.s[2] | lane2.s[2] | lane3.s[2] ]  (__m256i)
  s3  = [ lane0.s[3] | lane1.s[3] | lane2.s[3] | lane3.s[3] ]  (__m256i)
```

An AVX2 `__m256i` register is 256 bits = 4 × 64-bit lanes. Each intrinsic
operates on all four lanes in a single instruction. No shuffle, no permute.

### 2.2 The 4×2 structure

The benchmarks that go beyond x4 use **two independent sets** of four lanes:

```
  a = { a0, a1, a2, a3 }   ← four streams, first group
  b = { b0, b1, b2, b3 }   ← four streams, second group
```

Advancing `a` and `b` in the same loop iteration produces **eight** outputs
per cycle, hence "4×2". The two groups share no state, so their dependency
chains do not overlap — the CPU can execute the two groups' instructions
in parallel, hiding the latency of each individual step.

---

## 3. Rotate-Left Helper

The scrambler and state recurrence both need 64-bit rotate-left. There is no
`rotl` intrinsic in base AVX2, so we synthesise it from shifts:

```cpp
template <int k>
inline auto rotl64_avx2(__m256i x) noexcept -> __m256i {
#ifdef __AVX512VL__
    return _mm256_rol_epi64(x, k);
#else
    return _mm256_or_si256(_mm256_slli_epi64(x, k), _mm256_srli_epi64(x, 64 - k));
#endif
}
```

Several things to note:

* `k` is a **compile-time template parameter**, not a runtime variable. Both
  `_mm256_slli_epi64` and `_mm256_srli_epi64` require an immediate shift count,
  which the compiler can satisfy because `k` and `64 - k` are constant
  expressions. There is no branch at runtime.

* The `#ifdef __AVX512VL__` path uses `_mm256_rol_epi64`, available on
  Ice Lake and later. The compiler selects between the two paths at compile
  time; neither path is slower than a single rotate instruction when AVX-512
  is available.

* For the fallback: `rotl(x, k) = (x << k) | (x >> (64 - k))`. The OR
  combines the bits from both halves. The right-shift discards exactly the
  bits that the left-shift moved out of the high end, and vice versa.

---

## 4. The Core Step: `next_x4_avx2`

This function advances all four lanes of one group in one call and returns
the four output values as a single `__m256i`. It is the tightest loop in
the whole codebase.

```cpp
inline auto next_x4_avx2(__m256i& s0, __m256i& s1, __m256i& s2,
                         __m256i& s3) noexcept -> __m256i {
    const __m256i result =
        _mm256_add_epi64(rotl64_avx2<23>(_mm256_add_epi64(s0, s3)), s0);
    const __m256i t = _mm256_slli_epi64(s1, 17);
    s2 = _mm256_xor_si256(s2, s0);
    s3 = _mm256_xor_si256(s3, s1);
    s1 = _mm256_xor_si256(s1, s2);
    s0 = _mm256_xor_si256(s0, s3);
    s2 = _mm256_xor_si256(s2, t);
    s3 = rotl64_avx2<45>(s3);
    return result;
}
```

### Line-by-line

**Line 1 — Compute the output before mutating state.**

```cpp
const __m256i result =
    _mm256_add_epi64(rotl64_avx2<23>(_mm256_add_epi64(s0, s3)), s0);
```

This is the ++ scrambler applied to all four lanes simultaneously:
`rotl(s0 + s3, 23) + s0`. The result must be captured *now* because the
state update steps below will overwrite `s0` and `s3`. Compare to the scalar:

```cpp
result[lane] = rotl64(s0[lane] + s3[lane], 23) + s0[lane];
```

**Line 2 — Save `s1 << 17` before `s1` is overwritten.**

```cpp
const __m256i t = _mm256_slli_epi64(s1, 17);
```

The recurrence needs the *old* value of `s1` shifted left by 17. Since `s1`
will be clobbered two steps later (`s1 ^= s2`), we stash the shifted value
in `t` now.

**Lines 3–6 — XOR cascade.**

```cpp
s2 = _mm256_xor_si256(s2, s0);   // s2 ^= s0  (old s0)
s3 = _mm256_xor_si256(s3, s1);   // s3 ^= s1  (old s1)
s1 = _mm256_xor_si256(s1, s2);   // s1 ^= s2  (new s2, which used old s0)
s0 = _mm256_xor_si256(s0, s3);   // s0 ^= s3  (new s3, which used old s1)
```

Dependency analysis is important here. After line 3, `s2` has absorbed `s0`.
After line 4, `s3` has absorbed `s1`. Lines 5 and 6 then mix those together
into `s1` and `s0` respectively. The XOR chain diffuses each word's bits
across the whole state. This matches the scalar sequence exactly — the
parenthetical annotations above confirm which *generation* of each value is
used at each step.

**Line 7 — Apply the saved shift.**

```cpp
s2 = _mm256_xor_si256(s2, t);    // s2 ^= (old s1 << 17)
```

`t` was computed from the pre-update `s1`, so this correctly implements the
recurrence rule despite `s1` having been overwritten in line 5.

**Line 8 — Final rotate.**

```cpp
s3 = rotl64_avx2<45>(s3);
```

The recurrence ends with a 45-bit left rotation of `s3`. This is the only
non-XOR operation in the state update, chosen to maximise the avalanche of
the linear recurrence.

**Return.** The function returns the outputs computed in line 1. The four
output values for the four streams sit in the four 64-bit lanes of `result`.

---

## 5. Converting Random Bits to Floating-Point

Raw 64-bit integers need to become doubles in `[0, 1)` (uniform) or `(-1, 1)`
(for Marsaglia polar). Both conversions exploit IEEE 754 double layout directly
to avoid a slow integer-to-float conversion path.

### 5.1 `[0, 1)` — 52-bit mantissa trick

A double in `[1.0, 2.0)` has exponent bits `0x3FF` (biased exponent 1023) and
a 52-bit mantissa that varies freely. The trick is:

1. Right-shift the random bits by 12 to get a 52-bit value (discarding the top
   12 bits that would have gone into the sign and exponent fields).
2. OR in the `[1.0, 2.0)` exponent pattern.
3. Reinterpret the resulting bit pattern as a double. It lies in `[1.0, 2.0)`.
4. Subtract 1.0 to map it to `[0.0, 1.0)`.

```cpp
inline auto u64_to_uniform01_52_avx2(__m256i bits) noexcept -> __m256d {
    const __m256i exponent = _mm256_set1_epi64x(0x3ff0000000000000ULL);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256i mantissa = _mm256_srli_epi64(bits, 12);
    const __m256d one_to_two =
        _mm256_castsi256_pd(_mm256_or_si256(mantissa, exponent));
    return _mm256_sub_pd(one_to_two, one);
}
```

`_mm256_castsi256_pd` is a zero-cost type pun — it reinterprets the register's
bits as doubles without any computation. The subtraction of 1.0 is an exact
operation because `[1.0, 2.0)` and 1.0 share the same exponent, making their
difference exactly representable.

This technique yields 52 bits of mantissa (the full precision of a double),
with the result uniformly distributed over `[0, 1)` on a 2^-52 grid.

### 5.2 `(-1, 1)` — scale to signed range

The polar method needs pairs `(u1, u2)` in `(-1, 1)`. A simple linear mapping
from `[0, 1)` suffices:

```cpp
inline auto u64_to_pm1_52_avx2(__m256i bits) noexcept -> __m256d {
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d two = _mm256_set1_pd(2.0);
    return _mm256_sub_pd(_mm256_mul_pd(u64_to_uniform01_52_avx2(bits), two), one);
}
```

`[0, 1) * 2 - 1 = [-1, 1)`. The distribution is symmetric around zero on a
2^-51 grid. (The interval is half-open at 1.0 but includes values arbitrarily
close to both −1 and +1.)

---

## 6. Seeding: SplitMix64 and the Long Jump

Eight independent streams need eight independent starting states. A single
user-provided seed is expanded into 8 × 4 words using two helpers.

### 6.1 SplitMix64

SplitMix64 is a fast 64-bit generator used purely to expand a single seed into
high-quality initial state for a better generator. Each call returns a
different 64-bit value from an incrementing counter:

```cpp
inline auto splitmix64(std::uint64_t& state) noexcept -> std::uint64_t {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
```

The constant `0x9e3779b97f4a7c15` is a 64-bit Fibonacci hash multiplier
(golden ratio × 2^64). The three multiply-xorshift operations form a finaliser
that diffuses each bit of the counter into the full 64-bit output.

Four consecutive SplitMix64 calls from the seed establish the initial state
for stream 0.

### 6.2 Long jump

Simply stepping the RNG forward to get well-separated streams is impractical
— even 2^64 ordinary steps would take many seconds. The long-jump polynomial
advances the state by exactly 2^192 steps in O(256) ordinary steps using the
theory of linear recurrences over GF(2):

```cpp
inline void long_jump(std::uint64_t (&s)[4]) noexcept {
    static constexpr std::uint64_t kCoeffs[4] = {
        0x76e15d3efefdcbbfULL, 0xc5004e441c522fb3ULL,
        0x77710069854ee241ULL, 0x39109bb02acbe635ULL,
    };
    std::uint64_t t[4] = {};
    for (auto coeff : kCoeffs) {
        for (int b = 0; b < 64; ++b) {
            if (coeff & (std::uint64_t{1} << b)) {
                t[0] ^= s[0]; t[1] ^= s[1]; t[2] ^= s[2]; t[3] ^= s[3];
            }
            state_advance(s);
        }
    }
    s[0] = t[0]; s[1] = t[1]; s[2] = t[2]; s[3] = t[3];
}
```

The four 64-bit coefficients together encode the 256-bit polynomial
`p(x) = x^(2^192)` modulo the characteristic polynomial of the recurrence.
The algorithm walks every bit of the polynomial; when a bit is set, it
accumulates the current state into `t` via XOR. After all 256 bits are
processed, `t` holds the state that is exactly 2^192 steps ahead of where we
started. The same polynomial (and therefore the same `kCoeffs`) applies to
all xoshiro256 variants because the recurrence is identical across +, ++, and **.

### 6.3 Seeding all eight lanes

```cpp
inline void seed_lanes(std::uint64_t seed, int n,
                       std::uint64_t* s0, std::uint64_t* s1,
                       std::uint64_t* s2, std::uint64_t* s3) noexcept {
    std::uint64_t state[4] = {
        splitmix64(seed), splitmix64(seed),
        splitmix64(seed), splitmix64(seed),
    };
    for (int lane = 0; lane < n; ++lane) {
        s0[lane] = state[0]; s1[lane] = state[1];
        s2[lane] = state[2]; s3[lane] = state[3];
        if (lane + 1 < n) long_jump(state);
    }
}
```

Stream 0 starts at the SplitMix64-expanded state. Each subsequent stream
starts 2^192 steps ahead of the previous. With 8 streams the period per
stream is 2^256 / 8 = 2^253 — more than enough for any practical use.

`seed_x8` calls `seed_lanes(seed, 8, ...)` and partitions the result into
two `[4]`-element SoA arrays ready for the two `__m256i` register groups.

---

## 7. The 4×2 Loop: Putting It Together

With all building blocks in place, here is the complete uniform-distribution
kernel for the 4×2 (x8) configuration:

```cpp
void fill_xoshiro256pp_x8_uniform01_avx2(std::uint64_t seed, double* out,
                                         std::size_t count) noexcept {
#ifdef __AVX2__
```

### 7.1 State allocation and initialisation

```cpp
    alignas(32) std::uint64_t sa0[4], sa1[4], sa2[4], sa3[4];
    alignas(32) std::uint64_t sb0[4], sb1[4], sb2[4], sb3[4];
    alignas(32) double tail[4];
    seed_x8(seed, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);

    __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa0));
    __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa1));
    __m256i a2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa2));
    __m256i a3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sa3));
    __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb0));
    __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb1));
    __m256i b2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb2));
    __m256i b3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb3));
```

State is allocated on the stack with 32-byte alignment (required by
`_mm256_load_si256`) and seeded via `seed_x8`. The scalar arrays are then
loaded into eight AVX2 registers. From this point, the state lives entirely
in registers — no memory accesses inside the loop.

### 7.2 Main loop body

```cpp
    std::size_t i = 0;
    while (i + 8 <= count) {
        const __m256i ra = next_x4_avx2(a0, a1, a2, a3);
        const __m256i rb = next_x4_avx2(b0, b1, b2, b3);
        _mm256_storeu_pd(out + i,     u64_to_uniform01_52_avx2(ra));
        _mm256_storeu_pd(out + i + 4, u64_to_uniform01_52_avx2(rb));
        i += 8;
    }
```

This is the heart of the 4×2 pattern. Each iteration:

1. Advances group `a` (4 lanes) → `ra` holds four fresh 64-bit random values.
2. Advances group `b` (4 lanes) → `rb` holds four more.
3. Converts each group to doubles in `[0, 1)` and stores them to adjacent
   locations in `out`.

Because `a` and `b` share no registers, the CPU can execute the two
`next_x4_avx2` calls in an interleaved, pipelined fashion. The 8-instruction
chain of each call (including the output computation and state update) has
an end-to-end latency of roughly 10–15 cycles on modern µarchs, but the
second group's instructions can fill the scheduler while the first group's
results propagate — raising throughput well beyond what one group alone could
achieve.

### 7.3 Scalar remainder

```cpp
    while (i + 4 <= count) {
        const __m256d values = u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3));
        _mm256_storeu_pd(out + i, values);
        i += 4;
    }
    if (i < count) {
        const __m256d values = u64_to_uniform01_52_avx2(next_x4_avx2(a0, a1, a2, a3));
        _mm256_store_pd(tail, values);
        for (std::size_t lane = 0; i < count; ++lane, ++i)
            out[i] = tail[lane];
    }
#else
    fill_xoshiro256pp_x4_uniform01(seed, out, count);
#endif
}
```

When fewer than eight elements remain, the code falls back to single-group
steps (consuming 4 outputs per step) and finally extracts individual doubles
from a 32-byte-aligned scratch buffer `tail`. The `#else` branch provides a
portable scalar fallback for non-AVX2 compilation targets.

---

## 8. Why Two Independent Groups Beat One Unrolled

An earlier variant (`fill_xoshiro256pp_x4_uniform01_avx2_unroll2`) achieves
the same 8 outputs per loop iteration differently: it calls `next_x4_avx2`
twice on the *same* `(s0, s1, s2, s3)` state:

```cpp
const __m256i r0 = next_x4_avx2(s0, s1, s2, s3);
const __m256i r1 = next_x4_avx2(s0, s1, s2, s3);
```

The problem is a serial dependency: the second call cannot begin until the
first call has finished updating `s0–s3`, because the second call reads those
same registers as inputs. The latency of one full step sits on the critical
path before the second step can even start computing its output.

With two independent groups the critical paths are disjoint:

```
group a: a_result ← next(a0,a1,a2,a3)  → store
group b: b_result ← next(b0,b1,b2,b3)  → store
```

Neither path depends on the other. A modern out-of-order CPU can issue
instructions from both paths simultaneously, achieving close to the
throughput of a single `next_x4_avx2` call for the combined 8-output work.
This is the core benefit of the 4×2 decomposition over a 1×8 approach.

---

## 9. Extension to AVX-512: The 8×2 Pattern

The 4×2 decomposition generalises directly to AVX-512 as an 8×2 pattern:
each group holds 8 lanes in a single `__m512i` register rather than 4 in
`__m256i`. The arithmetic is identical — add, rotate, XOR, shift — but two
operations benefit from wider hardware:

- **Native 64-bit rotate.** AVX2 emulates `rotl(x, 23)` with a
  shift-pair-or sequence (3 instructions). AVX-512 provides `_mm512_rol_epi64`
  as a single instruction.
- **Wider stores.** Each `_mm512_storeu_pd` writes 8 doubles at once,
  doubling the per-iteration output to 16 samples.

The seeding and remainder logic stay the same — `seed_lanes` already accepts
an arbitrary lane count, and the tail loop falls back to 8-wide steps then
scalar extraction.

### 9.1 When AVX-512 hurts: AMD Zen 4

AMD Zen 4 implements AVX-512 by splitting every 512-bit operation into two
256-bit micro-ops. For integer/bitwise kernels (uniform, Bernoulli) the two
halves pipeline through the ALUs at full throughput, so the 8×2 pattern
breaks even with the AVX2 4×2 pattern. But for FP-math-heavy kernels that
use `_mm512_sqrt_pd`, `_mm512_div_pd`, or `_mm512_mask_compressstoreu_pd`
(the vecpolar normal kernel), the two halves serialize on the single divider
unit, making 512-bit strictly slower than 256-bit.

The AVX-512 vecpolar kernel detects AMD hardware at runtime via CPUID leaf 0
and falls back to the AVX2 vecpolar path automatically:

```cpp
inline auto cpu_is_amd() noexcept -> bool {
    static const bool is_amd = [] {
        unsigned eax, ebx, ecx, edx;
        __cpuid(0, eax, ebx, ecx, edx);
        // "AuthenticAMD" in EBX-EDX-ECX order
        return ebx == 0x68747541u && edx == 0x69746e65u && ecx == 0x444d4163u;
    }();
    return is_amd;
}
```

This keeps the benchmark harness portable across Intel and AMD without
requiring separate build targets.

---

*All code fragments are extracted verbatim from
`benchmarks/zorro_benchmark_kernels.cpp` in the Zorro repository.*
