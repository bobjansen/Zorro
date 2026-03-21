library(zorro)
library(microbenchmark)

rng <- zorro_create(42)

sizes <- c(1e4, 1e6, 1e8)

for (n in sizes) {
  cat(sprintf("\n=== n = %s ===\n", formatC(n, format = "d", big.mark = ",")))

  cat("\n-- Uniform [0, 1) --\n")
  print(microbenchmark(
    base   = runif(n),
    zorro  = zorro_uniform(rng, n),
    times  = 21L
  ))

  cat("\n-- Normal(0, 1) --\n")
  print(microbenchmark(
    base   = rnorm(n),
    zorro  = zorro_normal(rng, n),
    times  = 21L
  ))

  cat("\n-- Exponential(1) --\n")
  print(microbenchmark(
    base   = rexp(n),
    zorro  = zorro_exponential(rng, n),
    times  = 21L
  ))

  cat("\n-- Bernoulli(0.3) --\n")
  print(microbenchmark(
    base   = rbinom(n, 1L, 0.3),
    zorro  = zorro_bernoulli(rng, n, prob = 0.3),
    times  = 21L
  ))
}
