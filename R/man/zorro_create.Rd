\name{create_zorro_rng}
\alias{create_zorro_rng}
\title{Create a Zorro RNG Instance}
\usage{
create_zorro_rng(seed = 42L)
}
\arguments{
\item{seed}{Integer seed for the generator.}
}
\value{
An external pointer to the RNG state.
}
\description{
Creates a new xoshiro256++ random number generator with automatic SIMD
dispatch (AVX-512, AVX2, or portable fallback).
}
\examples{
rng <- create_zorro_rng(123)
draw_uniform(rng, 10)
}
