#!/usr/bin/env Rscript
# Wrapper around several R changepoint-detection packages used in the TCPD
# benchmark.  Called from benchmark_covid_cpd_winner.py.
#
# Usage:
#   Rscript cpd_r_wrapper.R <algorithm> <input_data.csv> <output_cps.csv> [param]
#
# The optional fourth argument is a numeric hyperparameter:
#   - For PELT/BinSeg/SegNeigh/CPNP/FPOP/WBS it scales the penalty.
#   - For ECP it is the significance level.
#   - For BOCPD_bcp / BOCPDMS_ocp / RBOCPDMS_ocp it is the probability threshold.
#   - For AMOC it is ignored.
#
# The input CSV has one column per channel (ch0, ch1, ...).
# The output CSV has a single column "cp" with 1-based changepoint indices.

suppressPackageStartupMessages({
  library(changepoint)
  library(changepoint.np)
  library(wbs)
  library(ecp)
  library(fpop)
  library(bcp)
  library(ocp)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript cpd_r_wrapper.R <algorithm> <input.csv> <output.csv> [param]")
}
ALGO <- args[1]
IN_CSV <- args[2]
OUT_CSV <- args[3]
PARAM <- if (length(args) >= 4) args[4] else NA_character_

# Try to parse the optional parameter as numeric; leave as NA if missing/invalid.
PARAM_NUM <- suppressWarnings(as.numeric(PARAM))
if (is.na(PARAM_NUM)) PARAM_NUM <- NA_real_

X <- as.matrix(read.csv(IN_CSV, header = TRUE))
n <- nrow(X)
MIN_CP <- 11
MAX_CP <- n - 10

feasible <- function(cps) {
  cps <- unique(as.integer(cps))
  cps <- cps[cps >= MIN_CP & cps <= MAX_CP]
  sort(cps)
}

merge_nearby <- function(cps, min_gap = 3) {
  cps <- sort(unique(cps))
  if (length(cps) == 0) return(integer(0))
  clusters <- list()
  current <- cps[1]
  for (c in cps[-1]) {
    if (c - current[length(current)] <= min_gap) {
      current <- c(current, c)
    } else {
      clusters <- c(clusters, list(current))
      current <- c
    }
  }
  clusters <- c(clusters, list(current))
  as.integer(sapply(clusters, function(x) as.integer(median(x))))
}

# -----------------------------------------------------------------------------
# Univariate algorithms: run on each channel and union nearby CPs.
# -----------------------------------------------------------------------------
run_per_channel <- function(func, X, ...) {
  cps <- integer(0)
  for (k in seq_len(ncol(X))) {
    res <- tryCatch(func(X[, k], ...), error = function(e) integer(0))
    cps <- c(cps, res)
  }
  feasible(merge_nearby(cps, min_gap = 3))
}

run_amoc <- function(x, pen_scale = NA_real_) {
  fit <- changepoint::cpt.mean(x, method = "AMOC")
  cpts(fit)
}

run_pelt <- function(x, pen_scale = NA_real_) {
  nloc <- length(x)
  pen <- if (!is.na(pen_scale)) pen_scale * log(nloc) else "BIC"
  if (!is.na(pen_scale)) {
    fit <- changepoint::cpt.mean(x, method = "PELT", penalty = "Manual", pen.value = pen)
  } else {
    fit <- changepoint::cpt.mean(x, method = "PELT", penalty = "BIC")
  }
  cpts(fit)
}

run_binseg <- function(x, pen_scale = NA_real_) {
  nloc <- length(x)
  pen <- if (!is.na(pen_scale)) pen_scale * log(nloc) else "BIC"
  if (!is.na(pen_scale)) {
    fit <- changepoint::cpt.mean(x, method = "BinSeg", penalty = "Manual", pen.value = pen, Q = 15)
  } else {
    fit <- changepoint::cpt.mean(x, method = "BinSeg", penalty = "BIC", Q = 15)
  }
  cpts(fit)
}

run_segneigh <- function(x, pen_scale = NA_real_) {
  nloc <- length(x)
  Q <- max(2L, min(15L, as.integer(nloc / 10) - 1L))
  pen <- if (!is.na(pen_scale)) pen_scale * log(nloc) else "BIC"
  if (!is.na(pen_scale)) {
    fit <- changepoint::cpt.mean(x, method = "SegNeigh", Q = Q, penalty = "Manual", pen.value = pen)
  } else {
    fit <- changepoint::cpt.mean(x, method = "SegNeigh", Q = Q, penalty = "BIC")
  }
  cpts(fit)
}

run_cpnp <- function(x, pen_scale = NA_real_) {
  nloc <- length(x)
  pen <- if (!is.na(pen_scale)) pen_scale * log(nloc) else "BIC"
  if (!is.na(pen_scale)) {
    fit <- changepoint.np::cpt.np(x, method = "PELT", penalty = "Manual", pen.value = pen)
  } else {
    fit <- changepoint.np::cpt.np(x, method = "PELT", penalty = "BIC")
  }
  cpts(fit)
}

run_wbs <- function(x, pen_scale = NA_real_) {
  fit <- wbs::wbs(x)
  if (!is.na(pen_scale)) {
    res <- wbs::changepoints(fit, penalty = "MBIC", C = pen_scale)
  } else {
    res <- wbs::changepoints(fit)
  }
  as.integer(res$cpt.ic$mbic.penalty)
}

run_fpop <- function(x, pen_scale = NA_real_) {
  nloc <- length(x)
  scale <- if (!is.na(pen_scale)) pen_scale else 1.0
  lam <- scale * log(nloc) * max(var(x), 1e-6)
  fit <- fpop::Fpop(x, lambda = lam)
  fit$t.est
}

run_rfpop <- function(x, pen_scale = NA_real_) {
  if (!requireNamespace("robseg", quietly = TRUE)) {
    warning("robseg package not installed; skipping RFPOP")
    return(integer(0))
  }
  nloc <- length(x)
  scale <- if (!is.na(pen_scale)) pen_scale else 1.0
  # Standardise by a robust MAD estimate, as recommended by robseg.
  est.std <- max(mad(diff(x) / sqrt(2)), 1e-6)
  z <- x / est.std
  lam <- scale * log(nloc)
  fit <- robseg::Rob_seg.std(x = z, loss = "L2", lambda = lam, lthreshold = NULL)
  as.integer(fit$t.est)
}

run_bocpd <- function(x, thresh = NA_real_) {
  fit <- bcp::bcp(x)
  threshold <- if (!is.na(thresh)) thresh else 0.5
  which(fit$posterior.prob > threshold)
}

# -----------------------------------------------------------------------------
# Multivariate algorithms.
# -----------------------------------------------------------------------------
run_ecp <- function(X, sig = NA_real_) {
  sig.lvl <- if (!is.na(sig)) sig else 0.05
  fit <- ecp::e.divisive(X, sig.lvl = sig.lvl, R = 199, min.size = 10)
  as.integer(fit$estimates)
}

ocp_multivariate <- function(X, cpthreshold = 0.5) {
  d <- ncol(X)
  pm <- rep(list("g"), d)
  ip <- rep(list(list(m = 0, k = 0.01, a = 0.01, b = 1e-4)), d)
  fit <- ocp::onlineCPD(X, multivariate = TRUE, probModel = pm,
                        init_params = ip, cpthreshold = cpthreshold)
  fit
}

run_bocpdms_ocp <- function(X, thresh = NA_real_) {
  cpthresh <- if (!is.na(thresh)) thresh else 0.5
  fit <- ocp_multivariate(X, cpthreshold = cpthresh)
  cps <- fit$threshcps
  if (is.list(cps)) cps <- cps[[1]]
  unique(as.integer(cps))
}

run_rbocpdms_ocp <- function(X, thresh = NA_real_) {
  cpthresh <- if (!is.na(thresh)) thresh else 0.5
  fit <- ocp_multivariate(X, cpthreshold = cpthresh)
  cps <- fit$changepoint_lists[[1]][[1]]
  merge_nearby(unique(as.integer(cps)), min_gap = 3)
}

# -----------------------------------------------------------------------------
# Dispatch.
# -----------------------------------------------------------------------------
cps <- tryCatch({
  switch(ALGO,
    "AMOC"          = run_per_channel(run_amoc, X, pen_scale = PARAM_NUM),
    "PELT"          = run_per_channel(run_pelt, X, pen_scale = PARAM_NUM),
    "BinSeg"        = run_per_channel(run_binseg, X, pen_scale = PARAM_NUM),
    "SegNeigh"      = run_per_channel(run_segneigh, X, pen_scale = PARAM_NUM),
    "CPNP"          = run_per_channel(run_cpnp, X, pen_scale = PARAM_NUM),
    "WBS"           = run_per_channel(run_wbs, X, pen_scale = PARAM_NUM),
    "FPOP"          = run_per_channel(run_fpop, X, pen_scale = PARAM_NUM),
    "BOCPD"         = run_per_channel(run_bocpd, X, thresh = PARAM_NUM),
    "RFPOP"         = run_per_channel(run_rfpop, X, pen_scale = PARAM_NUM),
    "ECP"           = feasible(run_ecp(X, sig = PARAM_NUM)),
    "BOCPDMS_ocp"   = feasible(run_bocpdms_ocp(X, thresh = PARAM_NUM)),
    "RBOCPDMS_ocp"  = feasible(run_rbocpdms_ocp(X, thresh = PARAM_NUM)),
    stop("Unknown algorithm: ", ALGO)
  )
}, error = function(e) {
  warning("Algorithm ", ALGO, " failed: ", conditionMessage(e))
  integer(0)
})

write.csv(data.frame(cp = cps), OUT_CSV, row.names = FALSE)
