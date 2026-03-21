\name{draw_normal}
\alias{draw_normal}
\title{Generate Normal Random Numbers}
\usage{
draw_normal(rng, n, mean = 0, sd = 1)
}
\arguments{
\item{rng}{External pointer from \code{create_zorro_rng}.}
\item{n}{Number of values to generate.}
\item{mean}{Mean of the distribution (default 0).}
\item{sd}{Standard deviation (default 1).}
}
\value{
Numeric vector of length \code{n}.
}
\description{
Generate normal random numbers using the Marsaglia polar method with
SIMD-accelerated xoshiro256++ engine.
}
\examples{
rng <- create_zorro_rng(42)
x <- draw_normal(rng, 1000)
hist(x)
}
