#include <iostream>
#include <vector>

#include "zorro/zorro.hpp"

// $ g++ -O3 -std=c++20 -I./include -mavx2 examples/print.cpp && time ./a.out && rm a.out
// 4.99994e+06
// 2276.04
// 1164.45
// 9.99971e+08
// 9.9992e+06
// 2.00121e+07
// 2.9985e+06
// 2.00055e+07
// -2389.47

// real    0m0.424s
// user    0m0.406s
// sys     0m0.021s

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);

  zorro::Rng rng(42);
  std::vector<double> buf(10'000'000);

  rng.fill_uniform(buf.data(), buf.size()); // Uniform [0, 1)
  std::cout << std::accumulate(buf.begin(), buf.end(), 0.) << "\n";
  rng.fill_uniform(buf.data(), buf.size(), -1.0, 1.0); // Uniform [-1, 1)
  std::cout << std::accumulate(buf.begin(), buf.end(), 0.) << "\n";
  rng.fill_normal(buf.data(), buf.size()); // Normal N(0, 1)
  std::cout << std::accumulate(buf.begin(), buf.end(), 0.) << "\n";
  rng.fill_normal(buf.data(), buf.size(), 100.0, 15.0); // Normal N(100, 15^2)
  std::cout << std::accumulate(buf.begin(), buf.end(), 0.) << "\n";
  rng.fill_exponential(buf.data(), buf.size()); // Exponential lambda = 1
  std::cout << std::accumulate(buf.begin(), buf.end(), 0.) << "\n";
  rng.fill_exponential(buf.data(), buf.size(), 0.5); // Exponential lambda = 0.5
  std::cout << std::accumulate(buf.begin(), buf.end(), 0.) << "\n";
  rng.fill_bernoulli(buf.data(), buf.size(), 0.3); // 0.0 / 1.0 with P(1) = 0.3
  std::cout << std::accumulate(buf.begin(), buf.end(), 0.) << "\n";
  rng.fill_gamma(buf.data(), buf.size(), 2.0); // Gamma(alpha, 1), alpha >= 1
  std::cout << std::accumulate(buf.begin(), buf.end(), 0.) << "\n";
  rng.fill_student_t(buf.data(), buf.size(), 5.0); // Student's t(nu)
  std::cout << std::accumulate(buf.begin(), buf.end(), 0.) << "\n";
}
