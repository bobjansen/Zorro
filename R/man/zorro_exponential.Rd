\name{draw_exponential}
\alias{draw_exponential}
\title{Generate Exponential Random Numbers}
\usage{
draw_exponential(rng, n, rate = 1)
}
\arguments{
\item{rng}{External pointer from \code{create_zorro_rng}.}
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
rng <- create_zorro_rng(42)
x <- draw_exponential(rng, 1000, rate = 2)
hist(x)
}
