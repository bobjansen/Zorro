\name{draw_gamma}
\alias{draw_gamma}
\title{Generate Gamma Random Numbers}
\usage{
draw_gamma(rng, n, alpha = 1)
}
\arguments{
\item{rng}{External pointer from \code{create_zorro_rng}.}
\item{n}{Number of values to generate.}
\item{alpha}{Shape parameter (must be >= 1, default 1).}
}
\value{
Numeric vector of length \code{n}.
}
\description{
Generate Gamma random numbers using the Marsaglia & Tsang method with
SIMD-accelerated xoshiro256++ engine.
}
\examples{
rng <- create_zorro_rng(42)
x <- draw_gamma(rng, 1000, alpha = 2)
hist(x)
}
