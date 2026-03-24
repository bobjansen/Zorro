#!/usr/bin/env bash
set -euo pipefail

# llvm-mca-tool — compile C++ to assembly, split it into one snippet per
# function (or MCA region), and run llvm-mca on each snippet.
#
# Typical usage:
#   llvm-mca-tool.sh --source my_kernel.cpp
#   llvm-mca-tool.sh --source my_kernel.cpp --function 'my_hot_*'
#   llvm-mca-tool.sh --source my_kernel.cpp --function '*avx2*' --loop
#   llvm-mca-tool.sh --source already_compiled.s
#   llvm-mca-tool.sh --source my_kernel.cpp --mcpu skylake,znver4
#
# LLVM-MCA region markers (in your source):
#   asm volatile("# LLVM-MCA-BEGIN my_region" ::: "memory");
#   // ... hot code ...
#   asm volatile("# LLVM-MCA-END my_region" ::: "memory");
#   When markers are detected, named regions are extracted instead of whole
#   functions, giving precise control without extern "C" wrappers.
#
# If llvm-mca is not installed, the script still compiles the source and writes
# per-kernel assembly snippets.

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
COMPILER="clang++"
LLVM_MCA_BIN="${LLVM_MCA_BIN:-}"
CXXFLAGS="${MCA_CXXFLAGS:-"-O3 -std=c++20 -fno-unroll-loops"}"
MCPU_LIST="${MCA_MCPU:-skylake,znver4}"
ITERATIONS=100
OUT_DIR=""
SOURCE_FILE=""
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

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
  cat <<EOF
Usage: $(basename "$0") --source FILE [options]

Options:
  --source FILE         C++ source or .s assembly to analyze (required)
  --compiler PATH       C++ compiler (default: ${COMPILER})
  --llvm-mca PATH       llvm-mca binary (or set LLVM_MCA_BIN)
  --cxxflags FLAGS      Compile flags (default: "${CXXFLAGS}", or set MCA_CXXFLAGS)
  --mcpu LIST           Comma-separated llvm-mca cpu list (default: ${MCPU_LIST}, or set MCA_MCPU)
  --kernel LIST         Comma-separated function/region list to analyze
  --function PATTERN    Glob pattern to filter discovered functions
                        (e.g. 'my_kernel_*', '*avx2*'); matches both
                        mangled and demangled names
  --loop                Extract only the first inner loop from each function,
                        stripping preamble, tail handling, and epilogue
  --iterations N        llvm-mca iteration count (default: ${ITERATIONS})
  --out-dir DIR         Output directory (default: build/llvm-mca under cwd)
  --summary-only        Print the compact summary table, not full llvm-mca dumps
  --timeline            Print llvm-mca's timeline view
  --timeline-max-cycles N
                        Max cycles in timeline (default: ${TIMELINE_MAX_CYCLES})
  --timeline-max-iterations N
                        Max iterations in timeline
  --bottleneck-analysis Enable llvm-mca bottleneck analysis
  --ce-helper           Write Compiler Explorer helper files
  --ce-out-dir DIR      Directory for CE helper files
  --list                Print discovered functions/regions and exit
  --help                Show this message

When --source points at a .cpp/.cc file it is compiled to assembly. When it
points at a .s/.S/.asm file, compilation is skipped. Functions are auto-
discovered from the assembly unless --kernel or --function restricts the set.

If the assembly contains LLVM-MCA region markers (# LLVM-MCA-BEGIN / END),
those named regions are extracted instead of whole functions.
EOF
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Return 0 if name matches a shell glob pattern.
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
      if (sym in is_func && sym !~ /^_?(GLOBAL|__cxx_global|__clang|_ZTV|_ZTS|_ZTI|_ZSt|_ZZN|_ZL|__do_global|__cxa|__dso|GCC_except|_fini|_init|_start|__libc)/)
        print sym
    }
  ' "${asm_file}"
}

# Detect LLVM-MCA region markers in an assembly file.
has_mca_regions() {
  local asm_file="$1"
  grep -q '# LLVM-MCA-BEGIN' "${asm_file}" 2>/dev/null
}

# List named regions from # LLVM-MCA-BEGIN <name> markers.
discover_mca_regions() {
  local asm_file="$1"
  grep '# LLVM-MCA-BEGIN' "${asm_file}" \
    | sed 's/.*# LLVM-MCA-BEGIN[[:space:]]*//' \
    | sed 's/[[:space:]]*$//'
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

# Extract exactly one function into one snippet file. Keeps labels and
# instructions, drops assembler bookkeeping (.cfi, .size, etc). Local
# labels are preserved so loop back-edges still work.
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
      sub(/:.*/, "", loop_label)
      gsub(/^[[:space:]]+/, "", loop_label)
      print
      next
    }
    # Once inside the loop, collect instructions and local labels until we
    # see a branch back to the loop header (the back-edge).
    found_header && !done {
      print
      if ($0 ~ loop_label && /^[[:space:]]+(j[a-z]+|jmp)[[:space:]]/) {
        done = 1
      }
    }
  ' "${snippet}" > "${tmp}"

  if [[ -s "${tmp}" ]]; then
    mv "${tmp}" "${snippet}"
  else
    rm -f "${tmp}"
  fi
}

# Write Compiler Explorer helper files (per-kernel .s files with comments).
write_ce_helpers() {
  local ce_dir="$1"
  local kernel

  mkdir -p "${ce_dir}"

  for kernel in "${SELECTED_KERNELS[@]}"; do
    {
      echo "# Kernel: ${kernel}"
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

# ---------------------------------------------------------------------------
# Summary table helpers
# ---------------------------------------------------------------------------
summary_header_printed=0

print_summary_header() {
  if [[ "${summary_header_printed}" -eq 0 ]]; then
    printf "%-38s %-16s %12s %12s %12s %10s %12s\n" \
      "kernel" "cpu" "cycles" "uops" "IPC" "uops/cyc" "RThroughput"
    printf "%-38s %-16s %12s %12s %12s %10s %12s\n" \
      "------" "---" "------" "----" "---" "--------" "-----------"
    summary_header_printed=1
  fi
}

print_summary_row() {
  local kernel="$1"
  local cpu="$2"
  local report="$3"
  awk -v kernel="${kernel}" -v cpu="${cpu}" '
    /^Total Cycles:/      { cycles = $3 }
    /^Total uOps:/        { uops = $3 }
    /^IPC:/               { ipc = $2 }
    /^uOps Per Cycle:/    { upc = $4 }
    /^Block RThroughput:/ { rtp = $3 }
    END {
      printf "%-38s %-16s %12s %12s %12s %10s %12s\n",
             kernel, cpu, cycles, uops, ipc, upc, rtp
    }
  ' "${report}"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)       SOURCE_FILE="$2";           shift 2 ;;
    --compiler)     COMPILER="$2";              shift 2 ;;
    --llvm-mca)     LLVM_MCA_BIN="$2";          shift 2 ;;
    --cxxflags)     CXXFLAGS="$2";              shift 2 ;;
    --mcpu)         MCPU_LIST="$2";             shift 2 ;;
    --kernel)       KERNEL_LIST="$2";           shift 2 ;;
    --function)     FUNCTION_FILTER="$2";       shift 2 ;;
    --iterations)   ITERATIONS="$2";            shift 2 ;;
    --out-dir)      OUT_DIR="$2";               shift 2 ;;
    --summary-only) SUMMARY_ONLY=1;             shift   ;;
    --timeline)     TIMELINE=1;                 shift   ;;
    --timeline-max-cycles)
                    TIMELINE_MAX_CYCLES="$2";   shift 2 ;;
    --timeline-max-iterations)
                    TIMELINE_MAX_ITERATIONS="$2"; shift 2 ;;
    --bottleneck-analysis) BOTTLENECK_ANALYSIS=1; shift  ;;
    --loop)         LOOP_ONLY=1;                shift   ;;
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

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
if [[ -z "${SOURCE_FILE}" ]]; then
  echo "Error: --source is required." >&2
  echo >&2
  usage >&2
  exit 1
fi

if [[ ! -f "${SOURCE_FILE}" ]]; then
  echo "Source file not found: ${SOURCE_FILE}" >&2
  exit 1
fi

SOURCE_IS_ASM=0
case "${SOURCE_FILE}" in
  *.s|*.S|*.asm) SOURCE_IS_ASM=1 ;;
esac

# Make SOURCE_FILE absolute.
SOURCE_FILE="$(cd "$(dirname "${SOURCE_FILE}")" && pwd)/$(basename "${SOURCE_FILE}")"

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="$(pwd)/build/llvm-mca"
fi

if [[ "${TIMELINE}" -eq 1 ]] && [[ "${SUMMARY_ONLY}" -eq 1 ]]; then
  echo "--timeline and --summary-only cannot be used together." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Compile or copy assembly
# ---------------------------------------------------------------------------
mkdir -p "${OUT_DIR}"

ASM_FILE="${OUT_DIR}/$(basename "${SOURCE_FILE}" | sed 's/\.[^.]*$//').s"

if [[ "${SOURCE_IS_ASM}" -eq 1 ]]; then
  cp "${SOURCE_FILE}" "${ASM_FILE}"
  echo "Using pre-compiled assembly: ${SOURCE_FILE}"
  echo "  copied to: ${ASM_FILE}"
else
  echo "Compiling with ${COMPILER}"
  echo "  source: ${SOURCE_FILE}"
  echo "  flags: ${CXXFLAGS}"
  echo "  output: ${ASM_FILE}"
  ${COMPILER} ${CXXFLAGS} -S -o "${ASM_FILE}" "${SOURCE_FILE}"
fi

# ---------------------------------------------------------------------------
# Discover functions / MCA regions
# ---------------------------------------------------------------------------
USE_MCA_REGIONS=0
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

# Demangle C++ names when c++filt is available.
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

# ---------------------------------------------------------------------------
# Select kernels
# ---------------------------------------------------------------------------
# --function: glob filter (matches mangled and demangled names)
# --kernel:   explicit comma-separated list
# neither:    all discovered
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

# ---------------------------------------------------------------------------
# --list: print and exit
# ---------------------------------------------------------------------------
if [[ "${LIST_ONLY}" -eq 1 ]]; then
  local_kind="functions"
  [[ "${USE_MCA_REGIONS}" -eq 1 ]] && local_kind="MCA regions"
  echo "Discovered ${local_kind} in $(basename "${SOURCE_FILE}"):"
  for fn in "${SELECTED_KERNELS[@]}"; do
    if [[ -n "${DEMANGLED[${fn}]+x}" ]] && [[ "${DEMANGLED[${fn}]}" != "${fn}" ]]; then
      printf "  %-40s  %s\n" "${fn}" "${DEMANGLED[${fn}]}"
    else
      printf "  %s\n" "${fn}"
    fi
  done
  exit 0
fi

# ---------------------------------------------------------------------------
# Extract snippets
# ---------------------------------------------------------------------------
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
  if [[ -n "${DEMANGLED[${kernel}]+x}" ]] && [[ "${DEMANGLED[${kernel}]}" != "${kernel}" ]]; then
    printf "  %-40s  %s\n" "${kernel}" "${DEMANGLED[${kernel}]}"
  else
    printf "  %s\n" "${kernel}"
  fi
done

if [[ "${CE_HELPER}" -eq 1 ]]; then
  echo
  echo "Compiler Explorer helper files:"
  echo "  ${CE_OUT_DIR}"
fi

# ---------------------------------------------------------------------------
# Find llvm-mca
# ---------------------------------------------------------------------------
if [[ -z "${LLVM_MCA_BIN}" ]]; then
  if command -v llvm-mca >/dev/null 2>&1; then
    LLVM_MCA_BIN="$(command -v llvm-mca)"
  else
    # Try common versioned paths on Debian/Ubuntu.
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
The per-kernel assembly snippets were written to:
  ${OUT_DIR}

Once llvm-mca is installed, rerun with the same arguments.
EOF
  exit 0
fi

echo
echo "Using llvm-mca at: ${LLVM_MCA_BIN}"

# ---------------------------------------------------------------------------
# Run llvm-mca
# ---------------------------------------------------------------------------
IFS=',' read -r -a CPU_TARGETS <<< "${MCPU_LIST}"

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
      local_desc="${kernel}"
      if [[ -n "${DEMANGLED[${kernel}]+x}" ]] && [[ "${DEMANGLED[${kernel}]}" != "${kernel}" ]]; then
        local_desc="${kernel} (${DEMANGLED[${kernel}]})"
      fi
      echo
      echo "--- ${local_desc} ---"
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
