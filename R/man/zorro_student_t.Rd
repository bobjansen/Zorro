\name{draw_student_t}
\alias{draw_student_t}
\title{Generate Student's t Random Numbers}
\usage{
draw_student_t(rng, n, df = 1)
}
\arguments{
\item{rng}{External pointer from \code{create_zorro_rng}.}
\item{n}{Number of values to generate.}
\item{df}{Degrees of freedom (default 1).}
}
\value{
Numeric vector of length \code{n}.
}
\description{
Generate Student's t random numbers using the ratio of a normal and
the square root of a scaled gamma, with SIMD-accelerated xoshiro256++ engine.
}
\examples{
rng <- create_zorro_rng(42)
x <- draw_student_t(rng, 1000, df = 5)
hist(x)
}
