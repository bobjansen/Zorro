\name{zorro_normal}
\alias{zorro_normal}
\title{Generate Normal Random Numbers}
\usage{
zorro_normal(rng, n, mean = 0, sd = 1)
}
\arguments{
\item{rng}{External pointer from \code{zorro_create}.}
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
rng <- zorro_create(42)
x <- zorro_normal(rng, 1000)
hist(x)
}
