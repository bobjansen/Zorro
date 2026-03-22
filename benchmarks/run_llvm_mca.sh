#!/usr/bin/env bash
set -euo pipefail

# Compile C++ to assembly, split it into one file per kernel, and optionally
# run llvm-mca on each kernel.
#
# Why this script exists:
# - LLVM-MCA wants compact assembly regions.
# - Looking at one giant compiler output file is noisy and brittle.
# - The repo has several natural comparison points (scalar, raw AVX2, AVX2 +
#   float conversion, interleaved x8 AVX2), and it is convenient to automate
#   them together.
#
# The script is intentionally verbose and heavily commented because it is meant
# to be read, adapted, and rerun when experimenting with scheduling changes.
#
# Typical usage (built-in harness):
#   ./benchmarks/run_llvm_mca.sh
#   ./benchmarks/run_llvm_mca.sh --mcpu znver4
#   ./benchmarks/run_llvm_mca.sh --mcpu skylake,znver4
#   ./benchmarks/run_llvm_mca.sh --cxxflags "-O3 -std=c++20 -mavx2 -mavx512vl"
#   ./benchmarks/run_llvm_mca.sh --list
#
# Bring-your-own source — no special harness needed:
#   ./benchmarks/run_llvm_mca.sh --source my_kernel.cpp
#   ./benchmarks/run_llvm_mca.sh --source my_kernel.cpp --function 'my_hot_loop'
#   ./benchmarks/run_llvm_mca.sh --source my_kernel.cpp --function 'kernel_*'
#   ./benchmarks/run_llvm_mca.sh --source already_compiled.s
#
# LLVM-MCA region markers (in your source):
#   asm volatile("# LLVM-MCA-BEGIN my_region" ::: "memory");
#   // ... hot code ...
#   asm volatile("# LLVM-MCA-END my_region" ::: "memory");
#   When markers are detected the script extracts those named regions instead
#   of whole functions, giving you precise control without extern "C" wrappers.
#
# If llvm-mca is not installed, the script still compiles the source and writes
# per-kernel assembly snippets. That is useful on machines where you want to
# inspect the compiler output first or copy the snippets elsewhere.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_HARNESS="${SCRIPT_DIR}/llvm_mca_harness.cpp"
SOURCE_FILE=""

COMPILER="clang++"
LLVM_MCA_BIN="${LLVM_MCA_BIN:-}"
# `-fno-unroll-loops` keeps the extracted kernels close to the source shape.
# That makes MCA comparisons easier to interpret. If you want to study the
# compiler's preferred unrolled version instead, pass your own `--cxxflags`.
CXXFLAGS="-O3 -std=c++20 -mavx2 -fno-unroll-loops -I${REPO_ROOT} -I${REPO_ROOT}/include -fno-exceptions -fno-rtti"
MCPU_LIST="skylake,znver4"
ITERATIONS=100
OUT_DIR="${REPO_ROOT}/build/llvm-mca"
LIST_ONLY=0
SUMMARY_ONLY=0
KERNEL_LIST=""
FUNCTION_FILTER=""
CE_HELPER=0
CE_OUT_DIR=""
TIMELINE=0
TIMELINE_MAX_CYCLES=80
TIMELINE_MAX_ITERATIONS=""
BOTTLENECK_ANALYSIS=0
LOOP_ONLY=0

# Known kernels for the built-in harness. When --source is used with an
# external file these are ignored and functions are auto-discovered.
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

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --source FILE         C++ source or .s assembly to analyze (default: built-in harness)
  --compiler PATH       C++ compiler to use (default: ${COMPILER})
  --llvm-mca PATH       llvm-mca binary to use
  --cxxflags FLAGS      Override compile flags (default includes repo include paths)
  --mcpu LIST           Comma-separated llvm-mca cpu list (default: ${MCPU_LIST})
  --kernel LIST         Comma-separated kernel list (built-in harness only)
  --function PATTERN    Glob pattern to filter auto-discovered functions
                        (e.g. 'my_kernel_*', '*avx2*'; applied to --source files)
  --iterations N        llvm-mca iteration count (default: ${ITERATIONS})
  --out-dir DIR         Output directory (default: ${OUT_DIR})
  --summary-only        Print the compact summary table, not full llvm-mca dumps
  --timeline            Print llvm-mca's timeline view in the console
  --timeline-max-cycles N
                        Max cycles shown in the timeline (default: ${TIMELINE_MAX_CYCLES})
  --timeline-max-iterations N
                        Max iterations shown in the timeline
  --bottleneck-analysis Enable llvm-mca bottleneck analysis
  --loop                Extract only the first inner loop from each function,
                        stripping preamble, tail handling, and epilogue
  --ce-helper           Write Compiler Explorer helper files
  --ce-out-dir DIR      Directory for Compiler Explorer helper files
  --list                Print available kernels and exit
  --help                Show this message

When --source points at a .cpp file it is compiled to assembly. When it points
at a .s file compilation is skipped. Functions are auto-discovered from the
assembly unless --kernel or --function restricts the set.

If the assembly contains LLVM-MCA region markers (# LLVM-MCA-BEGIN / END),
those named regions are extracted instead of whole functions.
EOF
}

is_builtin_kernel() {
  local needle="$1"
  local kernel
  for kernel in "${BUILTIN_KERNELS[@]}"; do
    if [[ "${kernel}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

# Return true if name matches a shell glob pattern.
matches_glob() {
  local name="$1" pattern="$2"
  # shellcheck disable=SC2254
  case "${name}" in
    ${pattern}) return 0 ;;
  esac
  return 1
}

# Auto-discover function symbols from an assembly file. Finds labels that
# look like function entry points — either preceded by a .globl directive
# or by a .type @function directive. Filters out compiler-generated helpers.
discover_functions() {
  local asm_file="$1"
  awk '
    /^[[:space:]]*\.globl[[:space:]]+/ {
      sym = $2
      gsub(/[[:space:]]/, "", sym)
      is_func[sym] = 1
    }
    /^[[:space:]]*\.type[[:space:]]+.*,@function/ {
      sym = $2
      sub(/,.*/, "", sym)
      gsub(/[[:space:]]/, "", sym)
      is_func[sym] = 1
    }
    /^[A-Za-z_][A-Za-z0-9_]*:[[:space:]]*(#.*)?$/ {
      sym = $0
      sub(/:.*/, "", sym)
      if (sym in is_func && sym !~ /^_?(GLOBAL|__cxx_global|__clang|_ZTV|_ZTS|_ZTI|_ZSt|_ZZN|_ZL|__do_global|__cxa|__dso|GCC_except|_fini|_init|_start|__libc|_mca_force_emit)/)
        print sym
    }
  ' "${asm_file}"
}

# Detect LLVM-MCA region markers in an assembly file.
has_mca_regions() {
  local asm_file="$1"
  grep -q '# LLVM-MCA-BEGIN' "${asm_file}" 2>/dev/null
}

# Extract named regions delimited by # LLVM-MCA-BEGIN name / # LLVM-MCA-END.
discover_mca_regions() {
  local asm_file="$1"
  grep '# LLVM-MCA-BEGIN' "${asm_file}" | sed 's/.*# LLVM-MCA-BEGIN[[:space:]]*//' | sed 's/[[:space:]]*$//'
}

# Extract a named MCA region (everything between BEGIN and END markers,
# inclusive) into a snippet file.
extract_mca_region() {
  local region="$1"
  local asm_file="$2"
  local snippet="${OUT_DIR}/${region}.s"

  awk -v region="${region}" '
    $0 ~ ("# LLVM-MCA-BEGIN[[:space:]]+" region "([[:space:]]|$)") { in_region = 1 }
    in_region { print }
    in_region && $0 ~ ("# LLVM-MCA-END[[:space:]]+" region "([[:space:]]|$)") { exit }
    in_region && $0 ~ ("# LLVM-MCA-END[[:space:]]*$") { exit }
  ' "${asm_file}" > "${snippet}"

  if [[ ! -s "${snippet}" ]]; then
    echo "Failed to extract MCA region '${region}'" >&2
    exit 1
  fi
}

kernel_outputs_per_iter() {
  case "$1" in
    zorro_mca_scalar_step)
      echo 1
      ;;
    zorro_mca_scalar_step_unroll2)
      echo 2
      ;;
    zorro_mca_avx2_next_x4|zorro_mca_avx2_next_x4_uniform01)
      echo 4
      ;;
    zorro_mca_avx2_interleaved_uniform_x8|zorro_mca_avx512_next_x8|zorro_mca_avx512_next_x8_uniform01)
      echo 8
      ;;
    zorro_mca_avx512_interleaved_uniform_x16)
      echo 16
      ;;
    *)
      echo 1
      ;;
  esac
}

compile_flags_enable_avx512() {
  local flags="$1"
  local token
  [[ "${flags}" == *-mavx512f* ]] && return 0
  [[ "${flags}" == *-march=native* ]] && return 0
  for token in ${flags}; do
    case "${token}" in
      -march=*avx512*|-mcpu=*avx512*)
        return 0
        ;;
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
        return 0
        ;;
    esac
  done
  return 1
}

describe_kernel() {
  case "$1" in
    zorro_mca_scalar_step)
      echo "scalar xoshiro256++ operator()"
      ;;
    zorro_mca_scalar_step_unroll2)
      echo "scalar xoshiro256++ operator(), two draws per loop"
      ;;
    zorro_mca_avx2_next_x4)
      echo "raw AVX2 4-lane xoshiro256++ core step"
      ;;
    zorro_mca_avx2_next_x4_uniform01)
      echo "raw AVX2 core step plus integer-to-double uniform conversion"
      ;;
    zorro_mca_avx2_interleaved_uniform_x8)
      echo "two interleaved AVX2 streams, matching the x8 uniform shape"
      ;;
    zorro_mca_avx512_next_x8)
      echo "raw AVX-512 8-lane xoshiro256++ core step"
      ;;
    zorro_mca_avx512_next_x8_uniform01)
      echo "raw AVX-512 x8 core step plus integer-to-double uniform conversion"
      ;;
    zorro_mca_avx512_interleaved_uniform_x16)
      echo "two interleaved AVX-512 streams, matching the x16 uniform shape"
      ;;
    *)
      echo "unknown kernel"
      ;;
  esac
}

write_ce_helpers() {
  local ce_dir="$1"
  local kernel

  mkdir -p "${ce_dir}"
  cp "${HARNESS}" "${ce_dir}/llvm_mca_harness.cpp"

  cat > "${ce_dir}/README.md" <<EOF
# Compiler Explorer Helper

These files are generated by \`$(basename "$0")\` to make it easier to inspect
the selected kernels in Compiler Explorer's LLVM-MCA pane.

Generated from:
- harness: \`${HARNESS}\`
- compiler: \`${COMPILER}\`
- compile flags used for the local extraction: \`${CXXFLAGS}\`
- selected kernels: \`${SELECTED_KERNELS[*]}\`
- local llvm-mca cpu list: \`${MCPU_LIST}\`

Suggested CE workflow:
1. Open https://godbolt.org/
2. Paste one of the per-kernel \`.s\` files into an x86-64 assembly pane if you
   want the closest match to the local script output.
3. Enable the LLVM-MCA pane and choose the microarchitecture you care about.
4. If you prefer working from source, use \`llvm_mca_harness.cpp\` only as a
   reference copy. It depends on repo headers and is not a standalone CE input.
5. If you rebuild the source setup manually in CE, use Clang with flags close
   to:
   \`${CXXFLAGS}\`
6. Search for one of the selected function names in the generated assembly pane.
EOF

  for kernel in "${SELECTED_KERNELS[@]}"; do
    {
      echo "# Kernel: ${kernel}"
      echo "# Description: $(describe_kernel "${kernel}")"
      echo "# Suggested compiler: ${COMPILER}"
      echo "# Suggested compile flags: ${CXXFLAGS}"
      echo "# Suggested llvm-mca cpus: ${MCPU_LIST}"
      echo "#"
      echo "# Paste this file into Compiler Explorer as x86-64 assembly if you want"
      echo "# to inspect the exact snippet extracted by the local script."
      echo
      cat "${OUT_DIR}/${kernel}.s"
    } > "${ce_dir}/${kernel}.s"
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE_FILE="$2"
      shift 2
      ;;
    --compiler)
      COMPILER="$2"
      shift 2
      ;;
    --llvm-mca)
      LLVM_MCA_BIN="$2"
      shift 2
      ;;
    --cxxflags)
      CXXFLAGS="$2"
      shift 2
      ;;
    --mcpu)
      MCPU_LIST="$2"
      shift 2
      ;;
    --kernel)
      KERNEL_LIST="$2"
      shift 2
      ;;
    --function)
      FUNCTION_FILTER="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --summary-only)
      SUMMARY_ONLY=1
      shift
      ;;
    --timeline)
      TIMELINE=1
      shift
      ;;
    --timeline-max-cycles)
      TIMELINE_MAX_CYCLES="$2"
      shift 2
      ;;
    --timeline-max-iterations)
      TIMELINE_MAX_ITERATIONS="$2"
      shift 2
      ;;
    --bottleneck-analysis)
      BOTTLENECK_ANALYSIS=1
      shift
      ;;
    --loop)
      LOOP_ONLY=1
      shift
      ;;
    --ce-helper)
      CE_HELPER=1
      shift
      ;;
    --ce-out-dir)
      CE_OUT_DIR="$2"
      shift 2
      ;;
    --list)
      LIST_ONLY=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# Determine whether we are using the built-in harness or an external source.
USE_BUILTIN=1
SOURCE_IS_ASM=0
if [[ -n "${SOURCE_FILE}" ]]; then
  USE_BUILTIN=0
  if [[ ! -f "${SOURCE_FILE}" ]]; then
    echo "Source file not found: ${SOURCE_FILE}" >&2
    exit 1
  fi
  case "${SOURCE_FILE}" in
    *.s|*.S|*.asm)
      SOURCE_IS_ASM=1
      ;;
  esac
  # Make SOURCE_FILE absolute so later steps are simpler.
  SOURCE_FILE="$(cd "$(dirname "${SOURCE_FILE}")" && pwd)/$(basename "${SOURCE_FILE}")"
fi

HARNESS="${SOURCE_FILE:-${DEFAULT_HARNESS}}"

# --- Kernel selection: built-in harness path ---
if [[ "${USE_BUILTIN}" -eq 1 ]]; then
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
fi

if [[ "${TIMELINE}" -eq 1 ]] && [[ "${SUMMARY_ONLY}" -eq 1 ]]; then
  cat >&2 <<EOF
--timeline and --summary-only pull in opposite directions.
Use --timeline by itself if you want the full llvm-mca report with the console
timeline view.
EOF
  exit 1
fi

mkdir -p "${OUT_DIR}"

ASM_FILE="${OUT_DIR}/$(basename "${HARNESS}" | sed 's/\.[^.]*$//').s"

# --- Compile or copy assembly ---
if [[ "${SOURCE_IS_ASM}" -eq 1 ]]; then
  cp "${HARNESS}" "${ASM_FILE}"
  echo "Using pre-compiled assembly: ${HARNESS}"
  echo "  copied to: ${ASM_FILE}"
else
  # If the source is a header file (.h, .hpp, .hxx, .hh), generate a thin
  # wrapper that includes it and instantiates the class so the compiler emits
  # the methods we want to analyze. This lets users point --source directly
  # at a library header without writing a separate .cpp.
  COMPILE_INPUT="${HARNESS}"
  case "${HARNESS}" in
    *.h|*.hpp|*.hxx|*.hh)
      WRAPPER="${OUT_DIR}/_mca_wrapper.cpp"
      # Each method gets its own extern "C" noinline wrapper so it survives as
      # a distinct symbol, while everything it CALLS is inlined at full -O3.
      # This gives llvm-mca self-contained loops with no artificial callq stubs.
      # The #define private public hack lets us reach internal _avx2 methods.
      cat > "${WRAPPER}" <<CPPEOF
#define private public
#include "$(basename "${HARNESS}")"
#undef private

#define MCA_WRAP(name, call) \\
  extern "C" __attribute__((noinline, used)) \\
  void name(zorro::Rng& rng, double* __restrict__ out, std::size_t n) { call; }

MCA_WRAP(mca_fill_uniform,              rng.fill_uniform(out, n))
MCA_WRAP(mca_fill_normal,               rng.fill_normal(out, n))
MCA_WRAP(mca_fill_exponential,          rng.fill_exponential(out, n))
MCA_WRAP(mca_fill_bernoulli,            rng.fill_bernoulli(out, n, 0.5))
#if defined(__AVX2__)
MCA_WRAP(mca_fill_bernoulli_avx2,       rng.fill_bernoulli_avx2(out, n, 0x7fffffffffffffffULL))
MCA_WRAP(mca_fill_uniform_avx2,         rng.fill_uniform(out, n, 0.0, 1.0))
MCA_WRAP(mca_fill_normal_avx2,          rng.fill_normal_avx2(out, n, 0.0, 1.0))
MCA_WRAP(mca_fill_exponential_avx2,     rng.fill_exponential_avx2(out, n, 1.0))
#endif
CPPEOF
      COMPILE_INPUT="${WRAPPER}"
      # Add the header's directory to include paths.
      CXXFLAGS="${CXXFLAGS} -I$(dirname "${HARNESS}")"
      echo "Source is a header — generated wrapper: ${WRAPPER}"
      ;;
  esac

  echo "Compiling with ${COMPILER}"
  echo "  source: ${COMPILE_INPUT}"
  echo "  flags: ${CXXFLAGS}"
  echo "  output: ${ASM_FILE}"

  if [[ "${USE_BUILTIN}" -eq 1 ]] && [[ "${need_avx512_compile}" -eq 1 ]] && [[ "${selected_has_non_avx512}" -eq 1 ]]; then
    cat <<EOF
  note: mixed AVX-512 and non-AVX-512 kernels selected in one compile.
        Scalar/AVX2 rows are useful for shape comparisons here, but they are
        not a pure AVX2 baseline when the translation unit is built with
        AVX-512-enabled flags.
EOF
  fi

  ${COMPILER} ${CXXFLAGS} -S -o "${ASM_FILE}" "${COMPILE_INPUT}"
fi

# --- Auto-discover kernels for external sources ---
# When using --source, detect whether the assembly contains LLVM-MCA region
# markers. If so, extract those named regions. Otherwise, discover function
# symbols from the assembly.
USE_MCA_REGIONS=0
if [[ "${USE_BUILTIN}" -eq 0 ]]; then
  if has_mca_regions "${ASM_FILE}"; then
    USE_MCA_REGIONS=1
    mapfile -t DISCOVERED < <(discover_mca_regions "${ASM_FILE}")
  else
    mapfile -t DISCOVERED < <(discover_functions "${ASM_FILE}")
  fi

  if [[ "${#DISCOVERED[@]}" -eq 0 ]]; then
    echo "No functions or MCA regions found in ${ASM_FILE}" >&2
    echo "Tip: use extern \"C\" __attribute__((noinline)) for stable symbol names," >&2
    echo "or add # LLVM-MCA-BEGIN / # LLVM-MCA-END markers in your source." >&2
    exit 1
  fi

  # Build an associative array mapping mangled -> demangled names so that
  # --function and --list can work with readable C++ names.
  declare -A DEMANGLED=()
  if command -v c++filt >/dev/null 2>&1; then
    local_idx=0
    while IFS= read -r line; do
      DEMANGLED["${DISCOVERED[${local_idx}]}"]="${line}"
      local_idx=$((local_idx + 1))
    done < <(printf '%s\n' "${DISCOVERED[@]}" | c++filt)
  else
    for fn in "${DISCOVERED[@]}"; do
      DEMANGLED["${fn}"]="${fn}"
    done
  fi

  # Apply --function glob filter if provided, otherwise use --kernel list.
  # The filter matches against both mangled and demangled names.
  if [[ -n "${FUNCTION_FILTER}" ]]; then
    SELECTED_KERNELS=()
    for fn in "${DISCOVERED[@]}"; do
      if matches_glob "${fn}" "${FUNCTION_FILTER}" || \
         matches_glob "${DEMANGLED[${fn}]}" "${FUNCTION_FILTER}"; then
        SELECTED_KERNELS+=("${fn}")
      fi
    done
    if [[ "${#SELECTED_KERNELS[@]}" -eq 0 ]]; then
      echo "No functions matching '${FUNCTION_FILTER}' found. Available:" >&2
      for fn in "${DISCOVERED[@]}"; do
        if [[ "${DEMANGLED[${fn}]}" != "${fn}" ]]; then
          printf "  %s  (%s)\n" "${fn}" "${DEMANGLED[${fn}]}" >&2
        else
          printf "  %s\n" "${fn}" >&2
        fi
      done
      exit 1
    fi
  elif [[ -n "${KERNEL_LIST}" ]]; then
    IFS=',' read -r -a SELECTED_KERNELS <<< "${KERNEL_LIST}"
  else
    SELECTED_KERNELS=("${DISCOVERED[@]}")
  fi
fi

if [[ "${LIST_ONLY}" -eq 1 ]]; then
  if [[ "${USE_BUILTIN}" -eq 1 ]]; then
    for kernel in "${BUILTIN_KERNELS[@]}"; do
      printf "%-38s %s\n" "${kernel}" "$(describe_kernel "${kernel}")"
    done
  else
    local_kind="functions"
    [[ "${USE_MCA_REGIONS}" -eq 1 ]] && local_kind="MCA regions"
    echo "Discovered ${local_kind} in $(basename "${HARNESS}"):"
    for fn in "${SELECTED_KERNELS[@]}"; do
      if [[ -n "${DEMANGLED[${fn}]+x}" ]] && [[ "${DEMANGLED[${fn}]}" != "${fn}" ]]; then
        printf "  %-40s  %s\n" "${fn}" "${DEMANGLED[${fn}]}"
      else
        printf "  %s\n" "${fn}"
      fi
    done
  fi
  exit 0
fi

# Extract exactly one function into one snippet file.  We keep labels and
# instructions, but drop assembler bookkeeping directives such as .cfi and
# .size because llvm-mca does not need them.  Local labels are preserved so
# loop back-edges still work.
extract_kernel() {
  local kernel="$1"
  local snippet="${OUT_DIR}/${kernel}.s"

  awk -v kernel="${kernel}" '
    $0 ~ ("^" kernel ":[[:space:]]*(#.*)?$") { in_fn = 1 }
    in_fn {
      if ($0 ~ "^[[:space:]]*\\.size[[:space:]]+" kernel ",") {
        exit
      }
      if ($0 ~ ("^" kernel ":[[:space:]]*(#.*)?$")) {
        print
        next
      }
      if ($0 ~ "^[.]L[^:]*:[[:space:]]*(#.*)?$") {
        print
        next
      }
      if ($0 ~ "^[[:space:]]+[A-Za-z].*") {
        print
        next
      }
    }
  ' "${ASM_FILE}" > "${snippet}"

  if [[ ! -s "${snippet}" ]]; then
    echo "Failed to extract assembly for ${kernel}" >&2
    exit 1
  fi
}

# Extract only the first inner loop from an already-extracted function snippet.
# Looks for the LLVM "Inner Loop Header" comment to find the loop, then grabs
# everything from that label through the backwards branch that closes it.
extract_loop_from_snippet() {
  local snippet="$1"
  local tmp="${snippet}.loop"

  awk '
    # Find the first inner loop header label.
    !found_header && /Inner Loop Header/ {
      found_header = 1
      loop_label = $0
      sub(/:.*/, "", loop_label)         # e.g. ".LBB4_5"
      gsub(/^[[:space:]]+/, "", loop_label)
      print
      next
    }
    # Once inside the loop, collect instructions and local labels until we
    # see a branch back to the loop header (the back-edge).
    found_header && !done {
      print
      # Detect the back-edge: a jump targeting the loop header label.
      if ($0 ~ loop_label && /^[[:space:]]+(j[a-z]+|jmp)[[:space:]]/) {
        done = 1
      }
    }
  ' "${snippet}" > "${tmp}"

  if [[ -s "${tmp}" ]]; then
    mv "${tmp}" "${snippet}"
  else
    # No inner loop found — keep the full function.
    rm -f "${tmp}"
  fi
}

for kernel in "${SELECTED_KERNELS[@]}"; do
  if [[ "${USE_MCA_REGIONS}" -eq 1 ]]; then
    extract_mca_region "${kernel}" "${ASM_FILE}"
  else
    extract_kernel "${kernel}"
  fi
  if [[ "${LOOP_ONLY}" -eq 1 ]]; then
    extract_loop_from_snippet "${OUT_DIR}/${kernel}.s"
  fi
done

if [[ "${CE_HELPER}" -eq 1 ]]; then
  if [[ -z "${CE_OUT_DIR}" ]]; then
    CE_OUT_DIR="${OUT_DIR}/compiler-explorer"
  fi
  write_ce_helpers "${CE_OUT_DIR}"
fi

echo
local_kind="kernels"
[[ "${USE_MCA_REGIONS}" -eq 1 ]] && local_kind="MCA regions"
echo "Extracted ${local_kind}:"
for kernel in "${SELECTED_KERNELS[@]}"; do
  local_desc="$(describe_kernel "${kernel}")"
  if [[ "${local_desc}" == "unknown kernel" ]]; then
    printf "  %s\n" "${kernel}"
  else
    printf "  %-38s %s\n" "${kernel}" "${local_desc}"
  fi
done

if [[ "${CE_HELPER}" -eq 1 ]]; then
  echo
  echo "Compiler Explorer helper files:"
  echo "  ${CE_OUT_DIR}"
fi

if [[ -z "${LLVM_MCA_BIN}" ]]; then
  if command -v llvm-mca >/dev/null 2>&1; then
    LLVM_MCA_BIN="$(command -v llvm-mca)"
  elif [[ -x /usr/lib/llvm-20/bin/llvm-mca ]]; then
    # Common Ubuntu/Debian install layout when the LLVM toolchain is present
    # but not added to PATH.
    LLVM_MCA_BIN="/usr/lib/llvm-20/bin/llvm-mca"
  fi
fi

if [[ -z "${LLVM_MCA_BIN}" ]]; then
  cat <<EOF

llvm-mca was not found in PATH.
The harness assembly and per-kernel snippets were still written to:
  ${OUT_DIR}

Once llvm-mca is installed, rerun:
  $(basename "$0") --out-dir "${OUT_DIR}"
EOF
  exit 0
fi

echo
echo "Using llvm-mca at: ${LLVM_MCA_BIN}"

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
      local_desc="$(describe_kernel "${kernel}")"
      echo
      if [[ "${local_desc}" != "unknown kernel" ]]; then
        echo "--- ${kernel} (${local_desc}) ---"
      elif [[ -n "${DEMANGLED+x}" ]] && [[ -n "${DEMANGLED[${kernel}]+x}" ]] && [[ "${DEMANGLED[${kernel}]}" != "${kernel}" ]]; then
        echo "--- ${kernel} (${DEMANGLED[${kernel}]}) ---"
      else
        echo "--- ${kernel} ---"
      fi
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
