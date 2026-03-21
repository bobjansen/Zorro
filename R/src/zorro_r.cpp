#include "zorro/zorro.hpp" // vendored copy in src/zorro/

#include <R.h>
#include <Rinternals.h>

// Prevent C++ name mangling for R entry points
extern "C" {

static void zorro_finalizer(SEXP ptr) {
  auto *rng = static_cast<zorro::Rng *>(R_ExternalPtrAddr(ptr));
  if (rng) {
    delete rng;
    R_ClearExternalPtr(ptr);
  }
}

SEXP C_zorro_create(SEXP seed_sexp) {
  double seed_dbl = REAL(seed_sexp)[0];
  auto seed = static_cast<std::uint64_t>(seed_dbl);

  auto *rng = new zorro::Rng(seed);

  SEXP ptr = PROTECT(R_MakeExternalPtr(rng, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ptr, zorro_finalizer, TRUE);
  UNPROTECT(1);
  return ptr;
}

static zorro::Rng *get_rng(SEXP ptr) {
  auto *rng = static_cast<zorro::Rng *>(R_ExternalPtrAddr(ptr));
  if (!rng)
    Rf_error("zorro RNG pointer is NULL (already garbage collected?)");
  return rng;
}

SEXP C_zorro_uniform(SEXP rng_ptr, SEXP n_sexp, SEXP low_sexp,
                     SEXP high_sexp) {
  zorro::Rng *rng = get_rng(rng_ptr);
  int n = INTEGER(n_sexp)[0];
  double low = REAL(low_sexp)[0];
  double high = REAL(high_sexp)[0];

  if (n <= 0)
    return Rf_allocVector(REALSXP, 0);

  SEXP out = PROTECT(Rf_allocVector(REALSXP, n));
  rng->fill_uniform(REAL(out), static_cast<std::size_t>(n), low, high);
  UNPROTECT(1);
  return out;
}

SEXP C_zorro_normal(SEXP rng_ptr, SEXP n_sexp, SEXP mean_sexp,
                    SEXP sd_sexp) {
  zorro::Rng *rng = get_rng(rng_ptr);
  int n = INTEGER(n_sexp)[0];
  double mean = REAL(mean_sexp)[0];
  double sd = REAL(sd_sexp)[0];

  if (n <= 0)
    return Rf_allocVector(REALSXP, 0);

  SEXP out = PROTECT(Rf_allocVector(REALSXP, n));
  rng->fill_normal(REAL(out), static_cast<std::size_t>(n), mean, sd);
  UNPROTECT(1);
  return out;
}

SEXP C_zorro_exponential(SEXP rng_ptr, SEXP n_sexp, SEXP rate_sexp) {
  zorro::Rng *rng = get_rng(rng_ptr);
  int n = INTEGER(n_sexp)[0];
  double rate = REAL(rate_sexp)[0];

  if (n <= 0)
    return Rf_allocVector(REALSXP, 0);

  SEXP out = PROTECT(Rf_allocVector(REALSXP, n));
  rng->fill_exponential(REAL(out), static_cast<std::size_t>(n), rate);
  UNPROTECT(1);
  return out;
}

SEXP C_zorro_bernoulli(SEXP rng_ptr, SEXP n_sexp, SEXP prob_sexp) {
  zorro::Rng *rng = get_rng(rng_ptr);
  int n = INTEGER(n_sexp)[0];
  double prob = REAL(prob_sexp)[0];

  if (n <= 0)
    return Rf_allocVector(REALSXP, 0);

  SEXP out = PROTECT(Rf_allocVector(REALSXP, n));
  rng->fill_bernoulli(REAL(out), static_cast<std::size_t>(n), prob);
  UNPROTECT(1);
  return out;
}

// R method registration table
static const R_CallMethodDef CallEntries[] = {
    {"C_zorro_create", (DL_FUNC)&C_zorro_create, 1},
    {"C_zorro_uniform", (DL_FUNC)&C_zorro_uniform, 4},
    {"C_zorro_normal", (DL_FUNC)&C_zorro_normal, 4},
    {"C_zorro_exponential", (DL_FUNC)&C_zorro_exponential, 3},
    {"C_zorro_bernoulli", (DL_FUNC)&C_zorro_bernoulli, 3},
    {NULL, NULL, 0}};

void R_init_zorro(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}

} // extern "C"
