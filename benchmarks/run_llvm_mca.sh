#!/usr/bin/env bash
set -euo pipefail

# Zorro-specific wrapper around llvm-mca-tool.sh.
#
# When run without --source, compiles the built-in MCA harness
# (llvm_mca_harness.cpp) and analyzes the zorro xoshiro256++ kernels.
# All generic llvm-mca-tool.sh flags are forwarded through.
#
# Typical usage:
#   ./benchmarks/run_llvm_mca.sh
#   ./benchmarks/run_llvm_mca.sh --mcpu znver4
#   ./benchmarks/run_llvm_mca.sh --kernel zorro_mca_avx2_next_x4
#   ./benchmarks/run_llvm_mca.sh --summary-only
#   ./benchmarks/run_llvm_mca.sh --source include/zorro/zorro.hpp --function '*bernoulli*' --loop
#
# For non-zorro usage, call llvm-mca-tool.sh directly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOL="${SCRIPT_DIR}/llvm-mca-tool.sh"

# Scan for --source in the arguments. If present, this is an ad-hoc
# analysis — forward everything to the generic tool as-is.
has_source=0
for arg in "$@"; do
  if [[ "${arg}" == "--source" ]]; then
    has_source=1
    break
  fi
done

if [[ "${has_source}" -eq 1 ]]; then
  # Ad-hoc mode: inject zorro include paths into default cxxflags unless
  # the user provided their own --cxxflags.
  has_cxxflags=0
  for arg in "$@"; do
    if [[ "${arg}" == "--cxxflags" ]]; then
      has_cxxflags=1
      break
    fi
  done

  if [[ "${has_cxxflags}" -eq 0 ]]; then
    export MCA_CXXFLAGS="-O3 -std=c++20 -mavx2 -fno-unroll-loops -I${REPO_ROOT} -I${REPO_ROOT}/include -fno-exceptions -fno-rtti"
  fi

  exec "${TOOL}" "$@"
fi

# --- Built-in harness mode -------------------------------------------------
# Zorro-specific: the harness has hand-crafted kernels with known names,
# descriptions, and per-kernel output counts for the summary table.

HARNESS="${SCRIPT_DIR}/llvm_mca_harness.cpp"
DEFAULT_CXXFLAGS="-O3 -std=c++20 -mavx2 -fno-unroll-loops -I${REPO_ROOT} -I${REPO_ROOT}/include -fno-exceptions -fno-rtti"

BUILTIN_KERNELS=(
  zorro_mca_scalar_step
  zorro_mca_scalar_step_unroll2
  zorro_mca_avx2_next_x4
  zorro_mca_avx2_next_x4_uniform01
  zorro_mca_avx2_interleaved_uniform_x8
  zorro_mca_avx512_next_x8
  zorro_mca_avx512_next_x8_uniform01
  zorro_mca_avx512_interleaved_uniform_x16
)

kernel_outputs_per_iter() {
  case "$1" in
    zorro_mca_scalar_step)                     echo 1  ;;
    zorro_mca_scalar_step_unroll2)             echo 2  ;;
    zorro_mca_avx2_next_x4|zorro_mca_avx2_next_x4_uniform01)
                                               echo 4  ;;
    zorro_mca_avx2_interleaved_uniform_x8|zorro_mca_avx512_next_x8|zorro_mca_avx512_next_x8_uniform01)
                                               echo 8  ;;
    zorro_mca_avx512_interleaved_uniform_x16)  echo 16 ;;
    *)                                         echo 1  ;;
  esac
}

describe_kernel() {
  case "$1" in
    zorro_mca_scalar_step)
      echo "scalar xoshiro256++ operator()" ;;
    zorro_mca_scalar_step_unroll2)
      echo "scalar xoshiro256++ operator(), two draws per loop" ;;
    zorro_mca_avx2_next_x4)
      echo "raw AVX2 4-lane xoshiro256++ core step" ;;
    zorro_mca_avx2_next_x4_uniform01)
      echo "raw AVX2 core step plus integer-to-double uniform conversion" ;;
    zorro_mca_avx2_interleaved_uniform_x8)
      echo "two interleaved AVX2 streams, matching the x8 uniform shape" ;;
    zorro_mca_avx512_next_x8)
      echo "raw AVX-512 8-lane xoshiro256++ core step" ;;
    zorro_mca_avx512_next_x8_uniform01)
      echo "raw AVX-512 x8 core step plus integer-to-double uniform conversion" ;;
    zorro_mca_avx512_interleaved_uniform_x16)
      echo "two interleaved AVX-512 streams, matching the x16 uniform shape" ;;
  esac
}

compile_flags_enable_avx512() {
  local flags="$1"
  local token
  [[ "${flags}" == *-mavx512f* ]] && return 0
  [[ "${flags}" == *-march=native* ]] && return 0
  for token in ${flags}; do
    case "${token}" in
      -march=*avx512*|-mcpu=*avx512*) return 0 ;;
      -march=skylake-avx512|-mcpu=skylake-avx512|\
      -march=cannonlake|-mcpu=cannonlake|\
      -march=icelake-client|-mcpu=icelake-client|\
      -march=icelake-server|-mcpu=icelake-server|\
      -march=cascadelake|-mcpu=cascadelake|\
      -march=cooperlake|-mcpu=cooperlake|\
      -march=tigerlake|-mcpu=tigerlake|\
      -march=sapphirerapids|-mcpu=sapphirerapids|\
      -march=knl|-mcpu=knl|\
      -march=knm|-mcpu=knm)
        return 0 ;;
    esac
  done
  return 1
}

is_builtin_kernel() {
  local needle="$1"
  for kernel in "${BUILTIN_KERNELS[@]}"; do
    [[ "${kernel}" == "${needle}" ]] && return 0
  done
  return 1
}

# --- Parse arguments, extracting what we handle and forwarding the rest ---
COMPILER="clang++"
LLVM_MCA_BIN="${LLVM_MCA_BIN:-}"
CXXFLAGS="${DEFAULT_CXXFLAGS}"
MCPU_LIST="skylake,znver4"
ITERATIONS=100
OUT_DIR="${REPO_ROOT}/build/llvm-mca"
LIST_ONLY=0
SUMMARY_ONLY=0
KERNEL_LIST=""
CE_HELPER=0
CE_OUT_DIR=""
TIMELINE=0
TIMELINE_MAX_CYCLES=80
TIMELINE_MAX_ITERATIONS=""
BOTTLENECK_ANALYSIS=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Zorro harness mode (default):
  $(basename "$0")
  $(basename "$0") --mcpu znver4
  $(basename "$0") --kernel zorro_mca_avx2_next_x4,zorro_mca_avx2_interleaved_uniform_x8
  $(basename "$0") --cxxflags "-O3 -std=c++20 -mavx2 -mavx512vl ..."

Ad-hoc source mode (forwarded to llvm-mca-tool.sh):
  $(basename "$0") --source my_kernel.cpp --function '*hot_loop*' --loop

Harness-mode options:
  --compiler PATH       C++ compiler (default: ${COMPILER})
  --llvm-mca PATH       llvm-mca binary
  --cxxflags FLAGS      Override compile flags
  --mcpu LIST           Comma-separated llvm-mca cpu list (default: ${MCPU_LIST})
  --kernel LIST         Comma-separated kernel list to analyze
  --iterations N        llvm-mca iteration count (default: ${ITERATIONS})
  --out-dir DIR         Output directory (default: ${OUT_DIR})
  --summary-only        Compact summary table only
  --timeline            Show timeline view
  --timeline-max-cycles N
  --timeline-max-iterations N
  --bottleneck-analysis Enable bottleneck analysis
  --ce-helper           Write Compiler Explorer helper files
  --ce-out-dir DIR      Directory for CE helper files
  --list                Print available kernels and exit
  --help                Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compiler)     COMPILER="$2";              shift 2 ;;
    --llvm-mca)     LLVM_MCA_BIN="$2";          shift 2 ;;
    --cxxflags)     CXXFLAGS="$2";              shift 2 ;;
    --mcpu)         MCPU_LIST="$2";             shift 2 ;;
    --kernel)       KERNEL_LIST="$2";           shift 2 ;;
    --iterations)   ITERATIONS="$2";            shift 2 ;;
    --out-dir)      OUT_DIR="$2";               shift 2 ;;
    --summary-only) SUMMARY_ONLY=1;             shift   ;;
    --timeline)     TIMELINE=1;                 shift   ;;
    --timeline-max-cycles)
                    TIMELINE_MAX_CYCLES="$2";   shift 2 ;;
    --timeline-max-iterations)
                    TIMELINE_MAX_ITERATIONS="$2"; shift 2 ;;
    --bottleneck-analysis) BOTTLENECK_ANALYSIS=1; shift  ;;
    --ce-helper)    CE_HELPER=1;                shift   ;;
    --ce-out-dir)   CE_OUT_DIR="$2";            shift 2 ;;
    --list)         LIST_ONLY=1;                shift   ;;
    --help)         usage; exit 0               ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# --- --list (early exit, no compile needed) ---
if [[ "${LIST_ONLY}" -eq 1 ]]; then
  for kernel in "${BUILTIN_KERNELS[@]}"; do
    printf "%-38s %s\n" "${kernel}" "$(describe_kernel "${kernel}")"
  done
  exit 0
fi

# --- Kernel selection ---
SELECTED_KERNELS=("${BUILTIN_KERNELS[@]}")
if [[ -n "${KERNEL_LIST}" ]]; then
  IFS=',' read -r -a SELECTED_KERNELS <<< "${KERNEL_LIST}"
  for kernel in "${SELECTED_KERNELS[@]}"; do
    if ! is_builtin_kernel "${kernel}"; then
      echo "Unknown kernel: ${kernel}" >&2
      echo >&2
      echo "Available kernels:" >&2
      for known in "${BUILTIN_KERNELS[@]}"; do
        printf "  %-38s %s\n" "${known}" "$(describe_kernel "${known}")" >&2
      done
      exit 1
    fi
  done
fi

# --- AVX-512 flag check ---
need_avx512_compile=0
selected_has_non_avx512=0
for kernel in "${SELECTED_KERNELS[@]}"; do
  if [[ "${kernel}" == zorro_mca_avx512_* ]]; then
    need_avx512_compile=1
  else
    selected_has_non_avx512=1
  fi
done

if [[ "${need_avx512_compile}" -eq 1 ]] && ! compile_flags_enable_avx512 "${CXXFLAGS}"; then
  cat >&2 <<EOF
Selected kernel set includes AVX-512 kernels, but the compile flags do not
enable AVX-512 code generation.

Accepted examples include:
  --cxxflags "-O3 -std=c++20 -mavx512f -fno-unroll-loops -I${REPO_ROOT} -I${REPO_ROOT}/include -fno-exceptions -fno-rtti"
  --cxxflags "-O3 -std=c++20 -march=skylake-avx512 -fno-unroll-loops -I${REPO_ROOT} -I${REPO_ROOT}/include -fno-exceptions -fno-rtti"
EOF
  exit 1
fi

if [[ "${TIMELINE}" -eq 1 ]] && [[ "${SUMMARY_ONLY}" -eq 1 ]]; then
  echo "--timeline and --summary-only cannot be used together." >&2
  exit 1
fi

# --- --list ---
if [[ "${LIST_ONLY}" -eq 1 ]]; then
  for kernel in "${BUILTIN_KERNELS[@]}"; do
    printf "%-38s %s\n" "${kernel}" "$(describe_kernel "${kernel}")"
  done
  exit 0
fi

# --- Compile ---
mkdir -p "${OUT_DIR}"
ASM_FILE="${OUT_DIR}/llvm_mca_harness.s"

echo "Compiling harness with ${COMPILER}"
echo "  flags: ${CXXFLAGS}"
echo "  output: ${ASM_FILE}"

if [[ "${need_avx512_compile}" -eq 1 ]] && [[ "${selected_has_non_avx512}" -eq 1 ]]; then
  cat <<EOF
  note: mixed AVX-512 and non-AVX-512 kernels selected in one compile.
        Scalar/AVX2 rows are useful for shape comparisons here, but they are
        not a pure AVX2 baseline when the translation unit is built with
        AVX-512-enabled flags.
EOF
fi

"${COMPILER}" ${CXXFLAGS} -S -o "${ASM_FILE}" "${HARNESS}"

# --- Extract kernels (same awk as the generic tool, inlined here) ---
extract_kernel() {
  local kernel="$1"
  local snippet="${OUT_DIR}/${kernel}.s"

  awk -v kernel="${kernel}" '
    $0 ~ ("^" kernel ":[[:space:]]*(#.*)?$") { in_fn = 1 }
    in_fn {
      if ($0 ~ "^[[:space:]]*\\.size[[:space:]]+" kernel ",") { exit }
      if ($0 ~ ("^" kernel ":[[:space:]]*(#.*)?$")) { print; next }
      if ($0 ~ "^[.]L[^:]*:[[:space:]]*(#.*)?$")   { print; next }
      if ($0 ~ "^[[:space:]]+[A-Za-z].*")           { print; next }
    }
  ' "${ASM_FILE}" > "${snippet}"

  if [[ ! -s "${snippet}" ]]; then
    echo "Failed to extract assembly for ${kernel}" >&2
    exit 1
  fi
}

for kernel in "${SELECTED_KERNELS[@]}"; do
  extract_kernel "${kernel}"
done

# --- CE helpers ---
if [[ "${CE_HELPER}" -eq 1 ]]; then
  if [[ -z "${CE_OUT_DIR}" ]]; then
    CE_OUT_DIR="${OUT_DIR}/compiler-explorer"
  fi
  mkdir -p "${CE_OUT_DIR}"
  cp "${HARNESS}" "${CE_OUT_DIR}/llvm_mca_harness.cpp"

  cat > "${CE_OUT_DIR}/README.md" <<EOF
# Compiler Explorer Helper

Generated from:
- harness: \`${HARNESS}\`
- compiler: \`${COMPILER}\`
- compile flags: \`${CXXFLAGS}\`
- selected kernels: \`${SELECTED_KERNELS[*]}\`
- llvm-mca cpu list: \`${MCPU_LIST}\`
EOF

  for kernel in "${SELECTED_KERNELS[@]}"; do
    {
      echo "# Kernel: ${kernel}"
      echo "# Description: $(describe_kernel "${kernel}")"
      echo "# Suggested compiler: ${COMPILER}"
      echo "# Suggested compile flags: ${CXXFLAGS}"
      echo "# Suggested llvm-mca cpus: ${MCPU_LIST}"
      echo
      cat "${OUT_DIR}/${kernel}.s"
    } > "${CE_OUT_DIR}/${kernel}.s"
  done

  echo
  echo "Compiler Explorer helper files:"
  echo "  ${CE_OUT_DIR}"
fi

echo
echo "Extracted kernels:"
for kernel in "${SELECTED_KERNELS[@]}"; do
  printf "  %-38s %s\n" "${kernel}" "$(describe_kernel "${kernel}")"
done

# --- Find llvm-mca ---
if [[ -z "${LLVM_MCA_BIN}" ]]; then
  if command -v llvm-mca >/dev/null 2>&1; then
    LLVM_MCA_BIN="$(command -v llvm-mca)"
  else
    for v in 20 19 18 17 16 15; do
      if [[ -x "/usr/lib/llvm-${v}/bin/llvm-mca" ]]; then
        LLVM_MCA_BIN="/usr/lib/llvm-${v}/bin/llvm-mca"
        break
      fi
    done
  fi
fi

if [[ -z "${LLVM_MCA_BIN}" ]]; then
  cat <<EOF

llvm-mca was not found in PATH.
The harness assembly and per-kernel snippets were still written to:
  ${OUT_DIR}
EOF
  exit 0
fi

echo
echo "Using llvm-mca at: ${LLVM_MCA_BIN}"

# --- Run llvm-mca ---
IFS=',' read -r -a CPU_TARGETS <<< "${MCPU_LIST}"

summary_header_printed=0

print_summary_header() {
  if [[ "${summary_header_printed}" -eq 0 ]]; then
    printf "%-38s %-16s %12s %12s %12s %10s %12s %10s %10s\n" \
      "kernel" "cpu" "cycles" "uops" "IPC" "uops/cyc" "RThroughput" "cyc/out" "out/cyc"
    printf "%-38s %-16s %12s %12s %12s %10s %12s %10s %10s\n" \
      "------" "---" "------" "----" "---" "--------" "-----------" "-------" "-------"
    summary_header_printed=1
  fi
}

print_summary_row() {
  local kernel="$1"
  local cpu="$2"
  local report="$3"
  local outputs
  outputs="$(kernel_outputs_per_iter "${kernel}")"
  awk -v kernel="${kernel}" -v cpu="${cpu}" -v outputs="${outputs}" '
    /^Total Cycles:/      { cycles = $3 }
    /^Total uOps:/        { uops = $3 }
    /^IPC:/               { ipc = $2 }
    /^uOps Per Cycle:/    { upc = $4 }
    /^Block RThroughput:/ { rtp = $3 }
    END {
      cyc_per_out = "n/a"
      out_per_cyc = "n/a"
      if (rtp != "" && outputs > 0) {
        cyc_per_out = sprintf("%.3f", rtp / outputs)
        out_per_cyc = sprintf("%.3f", outputs / rtp)
      }
      printf "%-38s %-16s %12s %12s %12s %10s %12s %10s %10s\n",
             kernel, cpu, cycles, uops, ipc, upc, rtp, cyc_per_out, out_per_cyc
    }
  ' "${report}"
}

for cpu in "${CPU_TARGETS[@]}"; do
  echo
  echo "=== llvm-mca for -mcpu=${cpu} ==="
  for kernel in "${SELECTED_KERNELS[@]}"; do
    local_snippet="${OUT_DIR}/${kernel}.s"
    report="${OUT_DIR}/${kernel}.${cpu}.txt"
    mca_args=(
      -mcpu="${cpu}"
      --iterations="${ITERATIONS}"
    )
    if [[ "${TIMELINE}" -eq 1 ]]; then
      mca_args+=(--timeline "--timeline-max-cycles=${TIMELINE_MAX_CYCLES}")
      if [[ -n "${TIMELINE_MAX_ITERATIONS}" ]]; then
        mca_args+=("--timeline-max-iterations=${TIMELINE_MAX_ITERATIONS}")
      fi
    fi
    if [[ "${BOTTLENECK_ANALYSIS}" -eq 1 ]]; then
      mca_args+=(--bottleneck-analysis)
    fi
    if [[ "${SUMMARY_ONLY}" -eq 0 ]]; then
      echo
      echo "--- ${kernel} ($(describe_kernel "${kernel}")) ---"
      "${LLVM_MCA_BIN}" "${mca_args[@]}" "${local_snippet}" | tee "${report}"
    else
      "${LLVM_MCA_BIN}" "${mca_args[@]}" "${local_snippet}" > "${report}"
    fi
  done

  echo
  echo "Summary for -mcpu=${cpu}:"
  print_summary_header
  for kernel in "${SELECTED_KERNELS[@]}"; do
    report="${OUT_DIR}/${kernel}.${cpu}.txt"
    print_summary_row "${kernel}" "${cpu}" "${report}"
  done
done
