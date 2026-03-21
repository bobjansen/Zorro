#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
OUT_DIR=""
STREAMS=("scalar" "x2" "x4")

usage() {
    cat <<'EOF'
Usage: benchmarks/run_dieharder.sh [options]

Options:
  --build-dir DIR   CMake build directory (default: ./build)
  --out-dir DIR     Directory for dieharder reports
  --stream NAME     One of: scalar, x2, x4, all (default: all)
  --help            Show this message

This runs a targeted dieharder battery against the raw bitstreams exported by
benchmarks/dieharder_stream.cpp. If a run reports WEAK or FAILED, the script
automatically reruns that test in ambiguity-resolution mode (-k 2 -Y 1).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --stream)
            case "$2" in
                scalar|x2|x4)
                    STREAMS=("$2")
                    ;;
                all)
                    STREAMS=("scalar" "x2" "x4")
                    ;;
                *)
                    echo "Unknown stream: $2" >&2
                    usage
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if ! command -v dieharder >/dev/null 2>&1; then
    echo "dieharder is not installed or not on PATH" >&2
    exit 1
fi

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$BUILD_DIR/dieharder/$(date +%Y%m%d-%H%M%S)"
fi

mkdir -p "$OUT_DIR"

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
fi
cmake --build "$BUILD_DIR" --target dieharder_stream -j >/dev/null

STREAM_BIN="$BUILD_DIR/dieharder_stream"
if [[ ! -x "$STREAM_BIN" ]]; then
    echo "Missing executable: $STREAM_BIN" >&2
    exit 1
fi

run_dieharder() {
    local stream="$1"
    local name="$2"
    shift 2

    local outfile="$OUT_DIR/${stream}_${name}.txt"
    local statuses
    local stream_status
    local dieharder_status

    echo "[$stream] $name"
    set +o pipefail
    "$STREAM_BIN" "$stream" | dieharder "$@" -g 200 >"$outfile"
    statuses=("${PIPESTATUS[@]}")
    set -o pipefail

    stream_status="${statuses[0]}"
    dieharder_status="${statuses[1]}"

    if [[ "$dieharder_status" -ne 0 ]]; then
        return "$dieharder_status"
    fi

    if [[ "$stream_status" -ne 0 && "$stream_status" -ne 1 && "$stream_status" -ne 141 ]]; then
        return "$stream_status"
    fi

    if grep -Eq 'WEAK|FAILED' "$outfile"; then
        local ra_outfile="$OUT_DIR/${stream}_${name}_ra.txt"
        echo "[$stream] $name rerun in resolution mode"
        set +o pipefail
        "$STREAM_BIN" "$stream" | dieharder "$@" -g 200 -k 2 -Y 1 >"$ra_outfile"
        statuses=("${PIPESTATUS[@]}")
        set -o pipefail

        stream_status="${statuses[0]}"
        dieharder_status="${statuses[1]}"

        if [[ "$dieharder_status" -ne 0 ]]; then
            return "$dieharder_status"
        fi

        if [[ "$stream_status" -ne 0 && "$stream_status" -ne 1 && "$stream_status" -ne 141 ]]; then
            return "$stream_status"
        fi
    fi
}

print_summary() {
    local file
    local ra_file
    local line
    local test_name
    local ntup
    local rerun_line
    local weak=0

    echo
    echo "Reports written to $OUT_DIR"
    echo "Summary"

    while IFS= read -r -d '' file; do
        if [[ "$file" == *_ra.txt ]]; then
            continue
        fi

        if grep -Eq 'WEAK|FAILED' "$file"; then
            weak=1
            ra_file="${file%.txt}_ra.txt"
            echo "$(basename "$file")"
            while IFS= read -r line; do
                echo "$line"
                if [[ -f "$ra_file" ]]; then
                    test_name="$(printf '%s\n' "$line" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1}')"
                    ntup="$(printf '%s\n' "$line" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')"
                    rerun_line="$(
                        awk -F'|' -v test_name="$test_name" -v ntup="$ntup" '
                            function trim(s) {
                                gsub(/^[ \t]+|[ \t]+$/, "", s)
                                return s
                            }
                            trim($1) == test_name && trim($2) == ntup { last = $0 }
                            END { print last }
                        ' "$ra_file"
                    )"
                    if [[ -n "$rerun_line" ]]; then
                        echo "  rerun -> $rerun_line"
                    fi
                fi
            done < <(grep -E 'WEAK|FAILED' "$file" || true)
        fi
    done < <(find "$OUT_DIR" -maxdepth 1 -type f -name '*.txt' -print0 | sort -z)

    if [[ "$weak" -eq 0 ]]; then
        echo "All recorded runs passed."
    fi
}

for stream in "${STREAMS[@]}"; do
    run_dieharder "$stream" "rank32" -d 2
    run_dieharder "$stream" "bitstream" -d 4
    run_dieharder "$stream" "sts_serial" -d 102
    run_dieharder "$stream" "rgb_bitdist10" -d 200 -n 10
    run_dieharder "$stream" "rgb_permutations" -d 202
done

print_summary
