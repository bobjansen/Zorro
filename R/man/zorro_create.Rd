\name{zorro_create}
\alias{zorro_create}
\title{Create a Zorro RNG Instance}
\usage{
zorro_create(seed = 42L)
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
rng <- zorro_create(123)
zorro_uniform(rng, 10)
}
