\name{zorro_exponential}
\alias{zorro_exponential}
\title{Generate Exponential Random Numbers}
\usage{
zorro_exponential(rng, n, rate = 1)
}
\arguments{
\item{rng}{External pointer from \code{zorro_create}.}
\item{n}{Number of values to generate.}
\item{rate}{Rate parameter lambda (default 1).}
}
\value{
Numeric vector of length \code{n}.
}
\description{
Generate exponential random numbers using inverse transform sampling with
SIMD-accelerated xoshiro256++ engine.
}
\examples{
rng <- zorro_create(42)
x <- zorro_exponential(rng, 1000, rate = 2)
hist(x)
}
