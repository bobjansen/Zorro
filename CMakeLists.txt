cmake_minimum_required(VERSION 3.20)

project(zorro LANGUAGES CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckCXXCompilerFlag)
include(CheckIncludeFileCXX)

check_cxx_compiler_flag("-mavx2" RNG_BENCH_HAS_MAVX2)
check_cxx_compiler_flag("-mavx512f" RNG_BENCH_HAS_MAVX512F)
check_include_file_cxx("testu01/unif01.h" RNG_BENCH_HAS_TESTU01_HEADERS)

find_library(RNG_BENCH_TESTU01_LIBRARY NAMES testu01)
find_library(RNG_BENCH_TESTU01_PROBDIST_LIBRARY NAMES testu01probdist)
find_library(RNG_BENCH_TESTU01_MYLIB_LIBRARY NAMES testu01mylib)

set(RNG_BENCH_HAS_TESTU01 OFF)
if(RNG_BENCH_HAS_TESTU01_HEADERS
   AND RNG_BENCH_TESTU01_LIBRARY
   AND RNG_BENCH_TESTU01_PROBDIST_LIBRARY
   AND RNG_BENCH_TESTU01_MYLIB_LIBRARY)
    set(RNG_BENCH_HAS_TESTU01 ON)
endif()

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

add_executable(
    high_dimensional_random_variables
    examples/high_dimensional_random_variables.cpp
)

add_executable(
    importance_sampling
    examples/importance_sampling.cpp
)

add_executable(
    normal_transform_comparison
    examples/normal_transform_comparison.cpp
    benchmarks/zorro_benchmark_kernels.cpp
)

add_executable(
    quality_tests
    benchmarks/quality_tests.cpp
)

add_executable(
    dieharder_stream
    benchmarks/dieharder_stream.cpp
)

add_executable(
    normal_transform_tests
    tests/normal_transform_tests.cpp
    benchmarks/zorro_benchmark_kernels.cpp
)

add_executable(
    exponential_transform_tests
    tests/exponential_transform_tests.cpp
)

if(RNG_BENCH_HAS_TESTU01)
    add_executable(
        testu01_runner
        benchmarks/testu01_runner.cpp
    )
endif()

target_include_directories(
    benchmark_distributions
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(
    high_dimensional_random_variables
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(
    importance_sampling
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(
    normal_transform_comparison
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(
    quality_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(
    dieharder_stream
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(
    normal_transform_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(
    exponential_transform_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(RNG_BENCH_HAS_TESTU01)
    target_include_directories(
        testu01_runner
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endif()

target_compile_options(
    benchmark_distributions
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-Wall;-Wextra;-Wpedantic>
)

target_compile_options(
    high_dimensional_random_variables
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-Wall;-Wextra;-Wpedantic>
)

target_compile_options(
    importance_sampling
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-Wall;-Wextra;-Wpedantic>
)

target_compile_options(
    normal_transform_comparison
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-Wall;-Wextra;-Wpedantic>
)

target_compile_options(
    quality_tests
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-Wall;-Wextra;-Wpedantic>
)

target_compile_options(
    dieharder_stream
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-Wall;-Wextra;-Wpedantic>
)

target_compile_options(
    normal_transform_tests
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-Wall;-Wextra;-Wpedantic>
)

target_compile_options(
    exponential_transform_tests
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-Wall;-Wextra;-Wpedantic>
)

if(RNG_BENCH_HAS_TESTU01)
    target_compile_options(
        testu01_runner
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-Wall;-Wextra;-Wpedantic>
    )
endif()

# libmvec provides vectorized math functions (_ZGVdN4v_log etc.)
target_compile_definitions(
    benchmark_distributions
    PRIVATE
        ZORRO_USE_LIBMVEC
)

target_compile_definitions(
    normal_transform_comparison
    PRIVATE
        ZORRO_USE_LIBMVEC
)

target_compile_definitions(
    normal_transform_tests
    PRIVATE
        ZORRO_USE_LIBMVEC
)

target_compile_definitions(
    exponential_transform_tests
    PRIVATE
        ZORRO_USE_LIBMVEC
)

target_link_libraries(
    benchmark_distributions
    PRIVATE
        m
        mvec
)

target_link_libraries(
    normal_transform_comparison
    PRIVATE
        m
        mvec
)

target_link_libraries(
    high_dimensional_random_variables
    PRIVATE
        m
        mvec
)

target_link_libraries(
    importance_sampling
    PRIVATE
        m
        mvec
)

target_link_libraries(
    quality_tests
    PRIVATE
        m
        mvec
)

target_link_libraries(
    dieharder_stream
    PRIVATE
        m
        mvec
)

target_link_libraries(
    normal_transform_tests
    PRIVATE
        m
        mvec
)

target_link_libraries(
    exponential_transform_tests
    PRIVATE
        m
        mvec
)

if(RNG_BENCH_HAS_TESTU01)
    target_link_libraries(
        testu01_runner
        PRIVATE
            m
            mvec
            ${RNG_BENCH_TESTU01_LIBRARY}
            ${RNG_BENCH_TESTU01_PROBDIST_LIBRARY}
            ${RNG_BENCH_TESTU01_MYLIB_LIBRARY}
    )
endif()

if(RNG_BENCH_ENABLE_AVX512)
    target_compile_options(
        benchmark_distributions
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx512f;-mavx512vl;-mavx512dq>
    )

    target_compile_options(
        high_dimensional_random_variables
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx512f;-mavx512vl;-mavx512dq>
    )

    target_compile_options(
        importance_sampling
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx512f;-mavx512vl;-mavx512dq>
    )

    target_compile_options(
        normal_transform_comparison
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx512f;-mavx512vl;-mavx512dq>
    )

    target_compile_options(
        quality_tests
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx512f;-mavx512vl;-mavx512dq>
    )

    target_compile_options(
        normal_transform_tests
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx512f;-mavx512vl;-mavx512dq>
    )

    target_compile_options(
        exponential_transform_tests
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx512f;-mavx512vl;-mavx512dq>
    )

    target_compile_options(
        dieharder_stream
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx512f;-mavx512vl;-mavx512dq>
    )

    if(RNG_BENCH_HAS_TESTU01)
        target_compile_options(
            testu01_runner
            PRIVATE
                $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx512f;-mavx512vl;-mavx512dq>
        )
    endif()
endif()

if(RNG_BENCH_ENABLE_NATIVE)
    target_compile_options(
        benchmark_distributions
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native;-mtune=native>
    )

    target_compile_options(
        high_dimensional_random_variables
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native;-mtune=native>
    )

    target_compile_options(
        importance_sampling
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native;-mtune=native>
    )

    target_compile_options(
        normal_transform_comparison
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native;-mtune=native>
    )

    target_compile_options(
        quality_tests
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native;-mtune=native>
    )

    target_compile_options(
        normal_transform_tests
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native;-mtune=native>
    )

    target_compile_options(
        exponential_transform_tests
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native;-mtune=native>
    )

    target_compile_options(
        dieharder_stream
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native;-mtune=native>
    )

    if(RNG_BENCH_HAS_TESTU01)
        target_compile_options(
            testu01_runner
            PRIVATE
                $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native;-mtune=native>
        )
    endif()
endif()

# Optional SIMD override flags appended after -march=native so they win.
# e.g. -DRNG_BENCH_MARCH_FLAGS="-mno-avx512f" or "-mno-avx2 -mno-avx512f"
set(RNG_BENCH_MARCH_FLAGS "" CACHE STRING "Extra SIMD override flags (appended after -march=native)")
if(RNG_BENCH_MARCH_FLAGS)
    separate_arguments(_march_flags UNIX_COMMAND "${RNG_BENCH_MARCH_FLAGS}")
    target_compile_options(benchmark_distributions PRIVATE ${_march_flags})
    target_compile_options(high_dimensional_random_variables PRIVATE ${_march_flags})
    target_compile_options(importance_sampling PRIVATE ${_march_flags})
    target_compile_options(normal_transform_comparison PRIVATE ${_march_flags})
    target_compile_options(quality_tests PRIVATE ${_march_flags})
    target_compile_options(normal_transform_tests PRIVATE ${_march_flags})
    target_compile_options(exponential_transform_tests PRIVATE ${_march_flags})
    target_compile_options(dieharder_stream PRIVATE ${_march_flags})
    if(RNG_BENCH_HAS_TESTU01)
        target_compile_options(testu01_runner PRIVATE ${_march_flags})
    endif()
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
            $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2;-mfma>
    )
endif()

if(RNG_BENCH_HAS_MAVX2)
    target_compile_options(
        normal_transform_comparison
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2;-mfma>
    )

    target_compile_options(
        normal_transform_tests
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2;-mfma>
    )

    target_compile_options(
        exponential_transform_tests
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2;-mfma>
    )
endif()

add_test(NAME normal_transform_tests COMMAND normal_transform_tests)
add_test(NAME exponential_transform_tests COMMAND exponential_transform_tests)

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
