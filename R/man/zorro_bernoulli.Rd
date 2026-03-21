\name{zorro_bernoulli}
\alias{zorro_bernoulli}
\title{Generate Bernoulli Random Numbers}
\usage{
zorro_bernoulli(rng, n, prob = 0.5)
}
\arguments{
\item{rng}{External pointer from \code{zorro_create}.}
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
rng <- zorro_create(42)
x <- zorro_bernoulli(rng, 1000, prob = 0.3)
mean(x)  # approximately 0.3
}
