#!/usr/bin/env bash
# Runs on an Ubuntu 24.04 EC2 instance: install latest GCC + Clang, build, benchmark.
# Build/install noise goes to stderr; clean results go to stdout.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# ── Install compilers ─────────────────────────────────────────────────────────

# LLVM repo for latest Clang
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
    | sudo tee /etc/apt/trusted.gpg.d/llvm.asc >/dev/null
CODENAME=$(lsb_release -cs)
for suffix in "" "-20" "-19" "-18"; do
    echo "deb http://apt.llvm.org/$CODENAME/ llvm-toolchain-${CODENAME}${suffix} main"
done | sudo tee /etc/apt/sources.list.d/llvm.list >/dev/null

# GCC PPA for latest GCC
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test >/dev/null 2>&1 || true

sudo apt-get update -qq >&2

# Build essentials
sudo apt-get install -y -qq cmake make build-essential lsb-release wget >&2

# Latest GCC (try 15 down to 14)
GCC=""
for v in 15 14 13; do
    if sudo apt-get install -y -qq g++-$v 2>/dev/null >&2; then
        GCC="g++-$v"; break
    fi
done
[[ -z "$GCC" ]] && GCC=g++

# Latest Clang (try 20 down to 18)
CLANG=""
for v in 20 19 18; do
    if sudo apt-get install -y -qq clang-$v 2>/dev/null >&2; then
        CLANG="clang++-$v"; break
    fi
done
[[ -z "$CLANG" ]] && { sudo apt-get install -y -qq clang >&2; CLANG=clang++; }

# ── Instance metadata (IMDSv2) ───────────────────────────────────────────────

TOKEN=$(curl -sf -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null) || TOKEN=""
if [[ -n "$TOKEN" ]]; then
    ITYPE=$(curl -sf -H "X-aws-ec2-metadata-token: $TOKEN" \
        http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null) || ITYPE="unknown"
else
    ITYPE=$(curl -sf http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null) || ITYPE="unknown"
fi

# ── Build with both compilers × SIMD levels ───────────────────────────────────

cd ~/rng

# name → RNG_BENCH_MARCH_FLAGS value  (empty = full native)
declare -A SIMD_LEVELS=(
    [native]=""
    [no-avx512]="-mno-avx512f -mno-avx512vl -mno-avx512bw -mno-avx512dq -mno-avx512cd -mno-avx512vnni"
    [no-avx2]="-mno-avx2 -mno-avx512f -mno-avx512vl -mno-avx512bw -mno-avx512dq -mno-avx512cd -mno-avx512vnni"
    [no-avx]="-mno-avx -mno-avx2 -mno-avx512f -mno-avx512vl -mno-avx512bw -mno-avx512dq -mno-avx512cd -mno-avx512vnni"
)
SIMD_ORDER=(native no-avx512 no-avx2 no-avx)

for COMPILER in "$GCC" "$CLANG"; do
    CTAG="${COMPILER%%-*}"  # "g++" or "clang++"
    for SIMD in "${SIMD_ORDER[@]}"; do
        FLAGS="${SIMD_LEVELS[$SIMD]}"
        DIR="build-${CTAG}-${SIMD}"
        # stephanfr requires AVX2 — skip for no-avx2 and no-avx builds
        STEPHANFR=ON
        [[ "$SIMD" == "no-avx2" || "$SIMD" == "no-avx" ]] && STEPHANFR=OFF
        cmake -S . -B "$DIR" \
            -DCMAKE_CXX_COMPILER="$COMPILER" \
            -DRNG_BENCH_ENABLE_NATIVE=ON \
            -DRNG_BENCH_ENABLE_STEPHANFR_AVX2="$STEPHANFR" \
            -DRNG_BENCH_MARCH_FLAGS="$FLAGS" \
            >&2
        cmake --build "$DIR" -j"$(nproc)" >&2
    done
done

# ── Report ───────────────────────────────────────────────────────────────────

echo "=== INSTANCE TYPE ==="
echo "$ITYPE"
echo ""

echo "=== CPU ==="
lscpu | grep -E 'Model name|CPU\(s\)|Thread|Core|Socket|cache|MHz|Byte Order'
echo ""

echo "=== SIMD FLAGS ==="
grep -oP '(avx|sse|fma)\S*' /proc/cpuinfo | sort -u | tr '\n' ' '
echo ""
echo ""

echo "=== COMPILERS ==="
echo "GCC:   $($GCC --version | head -1)"
echo "Clang: $($CLANG --version | head -1)"
echo ""

# ── AVX-512 rotate check & benchmarks ────────────────────────────────────────

for COMPILER in "$GCC" "$CLANG"; do
    CTAG="${COMPILER%%-*}"
    for SIMD in "${SIMD_ORDER[@]}"; do
        BIN="build-${CTAG}-${SIMD}/benchmark_distributions"
        [[ -f "$BIN" ]] || continue
        COUNT=$(objdump -d -M intel --no-show-raw-insn "$BIN" 2>/dev/null | grep -c 'vprol' || true)
        echo "=== BENCHMARK ($($COMPILER --version | head -1)) SIMD=$SIMD vprolq=$COUNT ==="
        taskset -c 0 "./$BIN"
        echo ""
    done
done
