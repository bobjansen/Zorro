cmake_minimum_required(VERSION 3.20)

project(zorro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-mavx2" RNG_BENCH_HAS_MAVX2)
check_cxx_compiler_flag("-mavx512f" RNG_BENCH_HAS_MAVX512F)

option(
    RNG_BENCH_ENABLE_STEPHANFR_AVX2
    "Benchmark the handwritten AVX2 Xoshiro256Plus implementation"
    ${RNG_BENCH_HAS_MAVX2}
)

# The compiler may accept -mavx512f even when the host CPU lacks AVX-512.
# Compile AND run a tiny AVX-512 program to detect actual hardware support.
set(_rng_avx512_default OFF)
if(RNG_BENCH_HAS_MAVX512F AND NOT CMAKE_CROSSCOMPILING)
    include(CheckCXXSourceRuns)
    set(CMAKE_REQUIRED_FLAGS "-mavx512f")
    check_cxx_source_runs("
#include <immintrin.h>
int main() {
    __m512i a = _mm512_set1_epi64(1);
    __m512i b = _mm512_add_epi64(a, a);
    (void)b;
    return 0;
}
" RNG_BENCH_CPU_HAS_AVX512F)
    unset(CMAKE_REQUIRED_FLAGS)
    if(RNG_BENCH_CPU_HAS_AVX512F)
        set(_rng_avx512_default ON)
    endif()
endif()

option(
    RNG_BENCH_ENABLE_AVX512
    "Enable AVX-512F kernels (also enables VL for 256-bit rol, DQ for rotate)"
    ${_rng_avx512_default}
)

option(
    RNG_BENCH_ENABLE_NATIVE
    "Tune benchmark builds for the host CPU with -march=native -mtune=native"
    OFF
)

add_executable(
    benchmark_distributions
    benchmarks/benchmark_distributions.cpp
    benchmarks/zorro_benchmark_kernels.cpp
)

target_include_directories(
    benchmark_distributions
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(
    benchmark_distributions
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-Wall;-Wextra;-Wpedantic>
)

# libmvec provides vectorized math functions (_ZGVdN4v_log etc.)
target_link_libraries(
    benchmark_distributions
    PRIVATE
        m
        mvec
)

if(RNG_BENCH_ENABLE_AVX512)
    target_compile_options(
        benchmark_distributions
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx512f;-mavx512vl;-mavx512dq>
    )
endif()

if(RNG_BENCH_ENABLE_NATIVE)
    target_compile_options(
        benchmark_distributions
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native;-mtune=native>
    )
endif()

# Optional SIMD override flags appended after -march=native so they win.
# e.g. -DRNG_BENCH_MARCH_FLAGS="-mno-avx512f" or "-mno-avx2 -mno-avx512f"
set(RNG_BENCH_MARCH_FLAGS "" CACHE STRING "Extra SIMD override flags (appended after -march=native)")
if(RNG_BENCH_MARCH_FLAGS)
    separate_arguments(_march_flags UNIX_COMMAND "${RNG_BENCH_MARCH_FLAGS}")
    target_compile_options(benchmark_distributions PRIVATE ${_march_flags})
endif()

if(RNG_BENCH_ENABLE_STEPHANFR_AVX2)
    target_sources(
        benchmark_distributions
        PRIVATE
            benchmarks/stephanfr_adapters.cpp
    )

    target_include_directories(
        benchmark_distributions
        SYSTEM PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stephanfr_xoshiro256plussimd/include
    )

    target_compile_definitions(
        benchmark_distributions
        PRIVATE
            RNG_BENCH_ENABLE_STEPHANFR_AVX2
            __AVX2_AVAILABLE__
    )

    target_compile_options(
        benchmark_distributions
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2>
    )
endif()

if(RNG_BENCH_ENABLE_AVX512)
    add_executable(
        test_avx512_determinism
        benchmarks/test_avx512_determinism.cpp
        benchmarks/zorro_benchmark_kernels.cpp
    )

    target_include_directories(
        test_avx512_determinism
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_compile_options(
        test_avx512_determinism
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-O2;-Wall;-Wextra;-mavx512f;-mavx512vl;-mavx512dq>
    )

    target_link_libraries(
        test_avx512_determinism
        PRIVATE
            m
            mvec
    )
endif()
