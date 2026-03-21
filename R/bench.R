library(zorro)
library(microbenchmark)

rng <- create_zorro_rng(42)

sizes <- c(1e4, 1e6, 1e8)

for (n in sizes) {
  cat(sprintf("\n=== n = %s ===\n", formatC(n, format = "d", big.mark = ",")))

  cat("\n-- Uniform [0, 1) --\n")
  print(microbenchmark(
    base   = runif(n),
    zorro  = draw_uniform(rng, n),
    times  = 21L
  ))

  cat("\n-- Normal(0, 1) --\n")
  print(microbenchmark(
    base   = rnorm(n),
    zorro  = draw_normal(rng, n),
    times  = 21L
  ))

  cat("\n-- Exponential(1) --\n")
  print(microbenchmark(
    base   = rexp(n),
    zorro  = draw_exponential(rng, n),
    times  = 21L
  ))

  cat("\n-- Bernoulli(0.3) --\n")
  print(microbenchmark(
    base   = rbinom(n, 1L, 0.3),
    zorro  = draw_bernoulli(rng, n, prob = 0.3),
    times  = 21L
  ))

  cat("\n-- Gamma(2) --\n")
  print(microbenchmark(
    base   = rgamma(n, shape = 2),
    zorro  = draw_gamma(rng, n, alpha = 2),
    times  = 21L
  ))

  cat("\n-- Student's t(5) --\n")
  print(microbenchmark(
    base   = rt(n, df = 5),
    zorro  = draw_student_t(rng, n, df = 5),
    times  = 21L
  ))
}
