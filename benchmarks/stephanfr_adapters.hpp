#pragma once

#include <cstddef>

namespace zorro_bench {

void fill_uniform_stephanfr_avx2_52(double* out, std::size_t count);
void fill_normals_stephanfr_avx2_52(double* out, std::size_t count);
void fill_normals_stephanfr_avx2_52_batched(double* out, std::size_t count);
void fill_normals_stephanfr_avx2_52_veclog(double* out, std::size_t count);
void fill_normals_stephanfr_avx2_52_vecpolar(double* out, std::size_t count);

}  // namespace zorro_bench
