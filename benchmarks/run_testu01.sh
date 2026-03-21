#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
OUT_DIR=""
STREAMS=("scalar" "x2" "x4")
BATTERY="smallcrush"

usage() {
    cat <<'EOF'
Usage: benchmarks/run_testu01.sh [options]

Options:
  --build-dir DIR   CMake build directory (default: ./build)
  --out-dir DIR     Directory for TestU01 reports
  --stream NAME     One of: scalar, x2, x4, all (default: all)
  --battery NAME    One of: smallcrush, crush, bigcrush (default: smallcrush)
  --help            Show this message

This runs TestU01 against the raw bit generators exposed by the project. The
runner links directly against the TestU01 libraries and writes one report per
stream to the output directory.
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
        --battery)
            case "$2" in
                smallcrush|crush|bigcrush)
                    BATTERY="$2"
                    ;;
                *)
                    echo "Unknown battery: $2" >&2
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

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$BUILD_DIR/testu01/${BATTERY}-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$OUT_DIR"

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
fi
cmake --build "$BUILD_DIR" --target testu01_runner -j >/dev/null

RUNNER="$BUILD_DIR/testu01_runner"
if [[ ! -x "$RUNNER" ]]; then
    echo "Missing executable: $RUNNER" >&2
    echo "TestU01 support may not have been detected by CMake." >&2
    exit 1
fi

for stream in "${STREAMS[@]}"; do
    local_out="$OUT_DIR/${stream}_${BATTERY}.txt"
    echo "[$stream] $BATTERY"
    "$RUNNER" "$BATTERY" "$stream" >"$local_out" 2>&1
done

echo
echo "Reports written to $OUT_DIR"
echo "Summary"
for stream in "${STREAMS[@]}"; do
    local_out="$OUT_DIR/${stream}_${BATTERY}.txt"
    echo "$(basename "$local_out")"
    if ! grep -E "All tests were passed|NOT passed|suspicious p-values|rejected" "$local_out" | tail -n 1; then
        tail -n 5 "$local_out"
    fi
done
