\name{draw_uniform}
\alias{draw_uniform}
\title{Generate Uniform Random Numbers}
\usage{
draw_uniform(rng, n, low = 0, high = 1)
}
\arguments{
\item{rng}{External pointer from \code{create_zorro_rng}.}
\item{n}{Number of values to generate.}
\item{low}{Lower bound (default 0).}
\item{high}{Upper bound (default 1).}
}
\value{
Numeric vector of length \code{n} with values in [low, high).
}
\description{
Generate uniform random numbers using the zorro xoshiro256++ engine.
}
\examples{
rng <- create_zorro_rng(42)
x <- draw_uniform(rng, 1000)
hist(x)
}
