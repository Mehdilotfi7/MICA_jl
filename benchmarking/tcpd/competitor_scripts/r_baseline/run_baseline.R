#!/usr/bin/env Rscript
# run_baseline.R
# TCPD-paper (van den Burg & Williams, 2020) baseline changepoint runner.
# Accepts a JSON configuration so the Python orchestrator can pass full
# hyperparameter grids for every algorithm.
#
# The config argument may be either:
#   - a single JSON object (legacy / default mode), in which case the output
#     is a CSV with one column 'cp' containing 1-based changepoint indices; or
#   - a JSON array of objects (batch / grid mode), in which case the output is
#     a JSON object {"results": [{"config": {...}, "cps": [...]}, ...]}.
#
# Usage:
#   Rscript run_baseline.R <algorithm> <input.csv> <output> '<json_config>'

suppressPackageStartupMessages(library(changepoint))
suppressPackageStartupMessages(library(changepoint.np))
suppressPackageStartupMessages(library(wbs))
suppressPackageStartupMessages(library(ecp))
suppressPackageStartupMessages(library(fpop))
suppressPackageStartupMessages(library(robseg))
suppressPackageStartupMessages(library(ocp))
suppressPackageStartupMessages(library(jsonlite))

`%||%` <- function(x, y) if (is.null(x)) y else x

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript run_baseline.R <algorithm> <input.csv> <output> ['<json_config>']")
}

algo <- toupper(args[1])
infile <- args[2]
outfile <- args[3]
config_arg <- if (length(args) >= 4) args[4] else ""
raw_config <- tryCatch({
  if (file.exists(config_arg)) {
    fromJSON(config_arg, simplifyDataFrame = FALSE)
  } else {
    fromJSON(config_arg, simplifyDataFrame = FALSE)
  }
}, error = function(e) list())
if (is.null(raw_config)) raw_config <- list()

batch_mode <- is.list(raw_config) && length(raw_config) > 0 && is.list(raw_config[[1]])
configs <- if (batch_mode) raw_config else list(raw_config)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

filter_cps <- function(cps, n) {
  cps <- sort(unique(as.integer(cps)))
  cps <- cps[!is.na(cps) & cps >= 1 & cps <= (n - 1)]
  cps
}

merge_close <- function(cps, min_gap = 3) {
  cps <- sort(unique(cps))
  if (length(cps) == 0) return(cps)
  keep <- cps[1]
  for (i in seq_along(cps)[-1]) {
    if (cps[i] - tail(keep, 1) >= min_gap) {
      keep <- c(keep, cps[i])
    }
  }
  keep
}

# Read input -------------------------------------------------------------
df <- tryCatch(read.csv(infile, stringsAsFactors = FALSE), error = function(e) NULL)
if (is.null(df) || !("y" %in% names(df))) {
  if (batch_mode) {
    write(toJSON(list(results = list()), auto_unbox = TRUE, null = "null"), outfile)
  } else {
    write.csv(data.frame(cp = integer(0)), outfile, row.names = FALSE)
  }
  quit(status = 0)
}

x <- as.numeric(df$y)
x <- x[!is.na(x)]
n <- length(x)

if (n < 2) {
  if (batch_mode) {
    write(toJSON(list(results = list()), auto_unbox = TRUE, null = "null"), outfile)
  } else {
    write.csv(data.frame(cp = integer(0)), outfile, row.names = FALSE)
  }
  quit(status = 0)
}

# Standardize (z-score) like the TCPD wrappers ---------------------------
sdx <- sd(x, na.rm = TRUE)
if (is.na(sdx) || sdx < 1e-12) {
  xs <- x - mean(x, na.rm = TRUE)
} else {
  xs <- as.vector(scale(x, center = TRUE, scale = TRUE))
}

# ---------------------------------------------------------------------------
# Per-config dispatcher
# ---------------------------------------------------------------------------

run_one <- function(config) {
  cps <- integer(0)

  # ----- changepoint package: AMOC, PELT, BinSeg, SegNeigh -------------
  if (algo %in% c("AMOC", "PELT", "BINSEG", "SEGNEIGH")) {

    cpt_type <- config$cpt_type %||% "mean"
    penalty  <- config$penalty %||% "MBIC"
    test_stat <- config$test_stat %||% "Normal"
    pen_scale <- config$pen_scale
    Q <- config$Q %||% 5

    cpt_func <- switch(cpt_type,
                       "mean"    = changepoint::cpt.mean,
                       "var"     = changepoint::cpt.var,
                       "meanvar" = changepoint::cpt.meanvar,
                       changepoint::cpt.mean)

    # changepoint v2.3 requires exact mixed case for BinSeg and SegNeigh.
    method_name <- switch(algo,
                          "BINSEG" = "BinSeg",
                          "SEGNEIGH" = "SegNeigh",
                          algo)

    common_args <- list(method = method_name, test.stat = test_stat,
                        minseglen = 1, penalty = penalty)

    if (penalty == "Manual" && !is.null(pen_scale)) {
      common_args$pen.value <- as.numeric(pen_scale) * log(n)
    } else if (penalty == "Asymptotic") {
      common_args$pen.value <- 0.05
    }

    if (algo %in% c("BINSEG", "SEGNEIGH")) {
      common_args$Q <- as.integer(Q)
    }

    cps <- tryCatch({
      fit <- do.call(cpt_func, c(list(xs), common_args))
      cp <- changepoint::cpts(fit)
      cp <- cp[!is.na(cp) & cp >= 1 & cp <= n]
      cp
    }, error = function(e) integer(0))

  # ----- CPNP (changepoint.np) -------------------------------------------
  } else if (algo == "CPNP") {

    penalty   <- config$penalty %||% "MBIC"
    pen_scale <- config$pen_scale
    nquantiles <- config$nquantiles %||% 10

    cps <- tryCatch({
      args <- list(xs, method = "PELT", penalty = penalty, minseglen = 1,
                   nquantiles = as.integer(nquantiles))
      if (penalty == "Manual" && !is.null(pen_scale)) {
        args$pen.value <- as.numeric(pen_scale) * log(n)
      } else if (penalty == "Asymptotic") {
        args$pen.value <- 0.05
      }
      fit <- do.call(changepoint.np::cpt.np, args)
      cp <- changepoint::cpts(fit)
      cp <- cp[!is.na(cp) & cp >= 1 & cp <= n]
      cp
    }, error = function(e) integer(0))

  # ----- RFPOP (robseg) --------------------------------------------------
  } else if (algo == "RFPOP") {

    loss       <- config$loss %||% "Outlier"
    lambda     <- config$lambda
    if (is.null(lambda) || lambda == "log_n") {
      lambda <- log(n)
    }
    lthreshold <- config$lthreshold %||% 3.0

    cps <- tryCatch({
      fit <- robseg::Rob_seg.std(xs, loss = loss,
                                  lambda = as.numeric(lambda),
                                  lthreshold = as.numeric(lthreshold))
      cp <- as.integer(fit$t.est)
      cp <- cp[!is.na(cp) & cp >= 1 & cp <= n]
      cp
    }, error = function(e) integer(0))

  # ----- ECP / KCPA (ecp) ------------------------------------------------
  } else if (algo == "ECP") {

    alg      <- config$algorithm %||% "e.divisive"
    sig.lvl  <- config$sig.lvl %||% 0.05
    min.size <- config$min.size %||% 30
    alpha    <- config$alpha %||% 1.0

    cps <- tryCatch({
      mat <- matrix(xs, ncol = 1)
      if (alg == "e.agglo") {
        fit <- ecp::e.agglo(mat, alpha = as.numeric(alpha))
        est <- as.integer(fit$estimates)
      } else {
        fit <- ecp::e.divisive(mat, sig.lvl = as.numeric(sig.lvl),
                                R = 199, min.size = as.integer(min.size),
                                alpha = as.numeric(alpha))
        est <- as.integer(fit$estimates)
      }
      cp <- est[est > 1 & est < (n + 1)]
      cp
    }, error = function(e) integer(0))

  } else if (algo == "KCPA") {

    cost   <- config$cost %||% 1.0
    max_cp <- config$max_cp
    if (is.null(max_cp) || max_cp == "max") {
      max_cp <- max(1, n - 1)
    }

    cps <- tryCatch({
      fit <- ecp::kcpa(matrix(xs, ncol = 1), L = as.integer(max_cp),
                       C = as.numeric(cost))
      cp <- as.integer(fit)
      cp <- cp[!is.na(cp) & cp >= 1 & cp <= n]
      cp
    }, error = function(e) integer(0))

  # ----- WBS (wbs) -------------------------------------------------------
  } else if (algo == "WBS") {

    penalty    <- config$penalty %||% "SSIC"
    integrated <- config$integrated %||% TRUE
    max.cp     <- config$max.cp %||% 50

    cps <- tryCatch({
      pen_map <- c("SSIC" = "ssic.penalty",
                   "BIC"  = "bic.penalty",
                   "MBIC" = "mbic.penalty")
      pen_name <- pen_map[[toupper(penalty)]]
      if (is.null(pen_name)) pen_name <- "ssic.penalty"

      wb <- wbs::wbs(xs, integrated = as.logical(integrated))
      fit <- wbs::changepoints(wb, penalty = pen_name,
                               Kmax = as.integer(max.cp))
      cp <- sort(unique(as.integer(unlist(fit$cpt.ic))))
      cp <- cp[cp >= 1 & cp <= n]
      cp
    }, error = function(e) integer(0))

  # ----- BOCPD (ocp package) ---------------------------------------------
  } else if (algo == "BOCPD") {

    intensity <- config$intensity %||% 100
    alpha0    <- config$alpha0 %||% 1
    beta0     <- config$beta0 %||% 1
    kappa0    <- config$kappa0 %||% 1

    cps <- tryCatch({
      fit <- ocp::onlineCPD(matrix(xs, ncol = 1), multivariate = FALSE,
                            hazard_func = function(x) ocp::const_hazard(x, lambda = as.numeric(intensity)),
                            init_params = list(list(m = 0,
                                                     k = as.numeric(kappa0),
                                                     a = as.numeric(alpha0),
                                                     b = as.numeric(beta0))),
                            cpthreshold = 0.5,
                            truncRlim = 0,
                            getR = FALSE,
                            optionalOutputs = FALSE)
      cp <- as.integer(sort(unique(unlist(fit$threshcps))))
      cp <- cp[cp >= 1 & cp <= n]
      merge_close(cp, min_gap = 3)
    }, error = function(e) integer(0))
  }

  filter_cps(cps, n)
}

# ---------------------------------------------------------------------------
# Run and write output
# ---------------------------------------------------------------------------

if (batch_mode) {
  results <- lapply(configs, function(cfg) {
    list(config = cfg, cps = as.list(run_one(cfg)))
  })
  write(toJSON(list(results = results), auto_unbox = TRUE, null = "null"),
       outfile)
} else {
  cps <- run_one(configs[[1]])
  write.csv(data.frame(cp = cps), outfile, row.names = FALSE)
}
