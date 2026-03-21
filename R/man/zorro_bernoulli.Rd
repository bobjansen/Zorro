\name{draw_bernoulli}
\alias{draw_bernoulli}
\title{Generate Bernoulli Random Numbers}
\usage{
draw_bernoulli(rng, n, prob = 0.5)
}
\arguments{
\item{rng}{External pointer from \code{create_zorro_rng}.}
\item{n}{Number of values to generate.}
\item{prob}{Probability of success (default 0.5).}
}
\value{
Numeric vector of 0s and 1s of length \code{n}.
}
\description{
Generate Bernoulli random trials using threshold comparison with
SIMD-accelerated xoshiro256++ engine.
}
\examples{
rng <- create_zorro_rng(42)
x <- draw_bernoulli(rng, 1000, prob = 0.3)
mean(x)
}
