#' Create a Zorro RNG Instance
#'
#' @param seed Integer seed for the generator.
#' @return An external pointer to the RNG state.
#' @export
create_zorro_rng <- function(seed = 42L) {
  .Call(C_zorro_create, as.double(seed))
}

#' Generate Uniform Random Numbers
#'
#' @param rng External pointer from \code{zorro_create}.
#' @param n Number of values to generate.
#' @param low Lower bound (default 0).
#' @param high Upper bound (default 1).
#' @return Numeric vector of length \code{n}.
#' @export
draw_uniform <- function(rng, n, low = 0, high = 1) {
  .Call(C_zorro_uniform, rng, as.integer(n), as.double(low), as.double(high))
}

#' Generate Normal Random Numbers
#'
#' @param rng External pointer from \code{zorro_create}.
#' @param n Number of values to generate.
#' @param mean Mean (default 0).
#' @param sd Standard deviation (default 1).
#' @return Numeric vector of length \code{n}.
#' @export
draw_normal <- function(rng, n, mean = 0, sd = 1) {
  .Call(C_zorro_normal, rng, as.integer(n), as.double(mean), as.double(sd))
}

#' Generate Exponential Random Numbers
#'
#' @param rng External pointer from \code{zorro_create}.
#' @param n Number of values to generate.
#' @param rate Rate parameter lambda (default 1).
#' @return Numeric vector of length \code{n}.
#' @export
draw_exponential <- function(rng, n, rate = 1) {
  .Call(C_zorro_exponential, rng, as.integer(n), as.double(rate))
}

#' Generate Bernoulli Random Numbers
#'
#' @param rng External pointer from \code{zorro_create}.
#' @param n Number of values to generate.
#' @param prob Probability of success (default 0.5).
#' @return Numeric vector of 0s and 1s of length \code{n}.
#' @export
draw_bernoulli <- function(rng, n, prob = 0.5) {
  .Call(C_zorro_bernoulli, rng, as.integer(n), as.double(prob))
}

#' Generate Gamma Random Numbers
#'
#' @param rng External pointer from \code{create_zorro_rng}.
#' @param n Number of values to generate.
#' @param alpha Shape parameter (must be >= 1).
#' @return Numeric vector of length \code{n}.
#' @export
draw_gamma <- function(rng, n, alpha = 1) {
  .Call(C_zorro_gamma, rng, as.integer(n), as.double(alpha))
}

#' Generate Student's t Random Numbers
#'
#' @param rng External pointer from \code{create_zorro_rng}.
#' @param n Number of values to generate.
#' @param df Degrees of freedom.
#' @return Numeric vector of length \code{n}.
#' @export
draw_student_t <- function(rng, n, df = 1) {
  .Call(C_zorro_student_t, rng, as.integer(n), as.double(df))
}
