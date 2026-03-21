\name{zorro_uniform}
\alias{zorro_uniform}
\title{Generate Uniform Random Numbers}
\usage{
zorro_uniform(rng, n, low = 0, high = 1)
}
\arguments{
\item{rng}{External pointer from \code{zorro_create}.}
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
rng <- zorro_create(42)
x <- zorro_uniform(rng, 1000)
hist(x)
}
