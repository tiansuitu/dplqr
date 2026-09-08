# R reconstruction of Zhong and Wang (2024), Section 5.1.
# This adapts the fitting, prediction, loss, and Python-to-R CSV handoff in the
# authors' original plaqr.R. The simulation DGP, repetitions, and tables are new.

script_arg <- grep("^--file=", commandArgs(FALSE), value=TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[1]))) else getwd()
local_library <- file.path(script_dir, "_r_libs")
if (dir.exists(local_library)) .libPaths(c(local_library, .libPaths()))

# R's historical default density grid is the one available when the paper
# was written. Keep it when running R >= 4.4 as well.
residual_density_zero <- function(residuals) {
  if (length(residuals) < 2L || any(!is.finite(residuals)))
    stop("Density estimation needs at least two finite residuals.")
  options <- list(x=as.numeric(residuals))
  if ("old.coords" %in% names(formals(stats::density.default))) options$old.coords <- TRUE
  estimate <- do.call(stats::density, options)
  value <- stats::approx(estimate$x, estimate$y, xout=0)$y
  if (!is.finite(value) || value <= 0) stop("Residual density at zero is not positive and finite.")
  as.numeric(value)
}

parse_args <- function() {
  # ACTIVE PROFILE: FIRST PASS. Section 9 of the notebook passes its current
  # settings explicitly; these defaults also match a standalone R invocation.
  # *%%* FULL REPLICATION marks preserved assignments; replace the active line
  # below each comment when restoring the full profile for standalone R use.
  out <- list(
    # *%%* FULL REPLICATION: repetitions=200L,
    repetitions=2L,
    # *%%* FULL REPLICATION: sample_sizes=c(1000L, 2000L),
    sample_sizes=c(1000L),
    # *%%* FULL REPLICATION: test_size=5000L,
    test_size=1000L,
    cases=1:3,
    # *%%* FULL REPLICATION: taus=c(0.25, 0.50, 0.75),
    taus=c(0.5),
    seed=20260904L,
    # *%%* FULL REPLICATION: output_dir=file.path(script_dir, "full-run"),
    output_dir=file.path(script_dir, "first-pass-run"),
    # *%%* FULL REPLICATION: python_results=file.path(script_dir, "full-run", "raw_results.csv"),
    python_results=file.path(script_dir, "first-pass-run", "raw_results.csv"),
    # *%%* FULL REPLICATION: data_dir=file.path(script_dir, "full-run", "data_for_r")
    data_dir=file.path(script_dir, "first-pass-run", "data_for_r"))
  for (arg in commandArgs(trailingOnly=TRUE)) {
    bits <- strsplit(sub("^--", "", arg), "=", fixed=TRUE)[[1]]
    if (length(bits) != 2) stop("Use --name=value arguments: ", arg)
    name <- gsub("-", "_", bits[1]); value <- bits[2]
    if (!name %in% names(out)) stop("Unknown argument: ", bits[1])
    if (name %in% c("output_dir", "python_results", "data_dir")) out[[name]] <- value
    else if (name %in% c("sample_sizes", "cases")) out[[name]] <- as.integer(strsplit(value, ",")[[1]])
    else if (name == "taus") out[[name]] <- as.numeric(strsplit(value, ",")[[1]])
    else out[[name]] <- as.integer(value)
  }
  out
}

require_replication_packages <- function() {
  missing <- c("quantreg", "plaqr")[!vapply(c("quantreg", "plaqr"), requireNamespace,
                                             logical(1), quietly=TRUE)]
  if (length(missing)) {
    stop("Missing R package(s): ", paste(missing, collapse=", "),
         ". See README.md for installation commands. No fallback estimator is substituted.")
  }
  # plaqr 2.0 calls rq() without a namespace qualifier internally.
  suppressPackageStartupMessages(library("quantreg", character.only=TRUE))
  suppressPackageStartupMessages(library("plaqr", character.only=TRUE))
}

nonlinear_truth <- function(z, case) {
  if (case == 1) return(0.56 * rowSums(z))
  if (case == 2) {
    inside <- (z[,1]-1)^2-z[,2]^2+3*abs(z[,3]-1)+0.6*sin(pi*z[,4])+
      log(z[,5]+0.5)+sqrt(z[,6]+0.5)+3*cos(0.1*pi*z[,7])+
      3*(z[,8]-1+abs(z[,8]-1))
    return(0.82 * inside)
  }
  if (case == 3) {
    inside <- exp(z[,1]*(1+z[,2]-pi*z[,3]*z[,4])/2)*(z[,5]+0.2)+
      z[,5]*(z[,4]-0.3)/(abs(2*z[,4]-1)+1)+
      2*sin(z[,5])*abs(z[,5]*z[,6]-0.6)+log(z[,6]+z[,7]*z[,8])
    return(0.61 * inside)
  }
  stop("Unknown case")
}

generate_covariates <- function(n) {
  sigma <- matrix(0.5, 10, 10); diag(sigma) <- 1
  gaussian <- matrix(rnorm(n*10), n, 10) %*% chol(sigma)
  z_tilde <- 2 * pnorm(gaussian)
  list(x=cbind(as.numeric(z_tilde[,9] > 1), z_tilde[,10]), z=z_tilde[,1:8])
}

generate_dataset <- function(n, case, error=TRUE) {
  covariates <- generate_covariates(n)
  m <- nonlinear_truth(covariates$z, case)
  y <- as.vector(covariates$x %*% c(1,-1) + m + if (error) rt(n, df=3) else 0)
  list(x=covariates$x, z=covariates$z, y=y, m=m)
}

as_frame <- function(d) {
  frame <- data.frame(y=d$y, x1=d$x[,1], x2=d$x[,2], d$z)
  names(frame)[4:11] <- paste0("z", 1:8)
  frame
}

checkLoss_mean <- function(errors, tau=0.5) {
  mat <- cbind(tau*errors, (tau-1)*errors)
  mean(apply(mat, 1, max))
}

fit_plaqr <- function(train, test, tau) {
  fit <- plaqr::plaqr(y ~ ., nonlinVars=~z1+z2+z3+z4+z5+z6+z7+z8,
                      data=as_frame(train), tau=tau)
  # summary.rq defaults to rank intervals below 1001 fitting observations,
  # and an SE table for larger samples. Set alpha explicitly: rank's default
  # alpha=0.1 would otherwise produce 90%, not the paper's 95%, intervals.
  smry <- summary(fit, alpha=0.05)$coefficients
  theta <- smry[c("x1", "x2"), 1]
  critical <- stats::qnorm(0.975)
  if (all(c("lower bd", "upper bd") %in% colnames(smry))) {
    lower <- smry[c("x1", "x2"), "lower bd"]
    upper <- smry[c("x1", "x2"), "upper bd"]
    # This is an interval-width equivalent SE for the shared output schema;
    # rank interval bounds themselves are authoritative for coverage.
    se <- (upper-lower)/(2*critical)
  } else if ("Std. Error" %in% colnames(smry)) {
    se <- smry[c("x1", "x2"), "Std. Error"]
    lower <- theta-critical*se
    upper <- theta+critical*se
  } else stop("Unrecognized plaqr/quantreg summary column names: ", paste(colnames(smry), collapse=", "))
  pred <- as.vector(predict(fit, newdata=as_frame(test)))
  list(theta=theta, se=se, lower=lower, upper=upper,
       pred=pred, mhat=pred-as.vector(test$x %*% theta))
}

read_exported_dataset <- function(path, expected_rows=NULL) {
  frame <- read.csv(path)
  required <- c("y", "x1", "x2", paste0("z", 1:8), "true_m")
  if (!identical(names(frame), required)) stop("Unexpected data columns in ", path)
  if (!is.null(expected_rows) && nrow(frame) != expected_rows)
    stop("Unexpected row count in ", path, ": expected ", expected_rows, ", got ", nrow(frame))
  if (!all(vapply(frame, is.numeric, logical(1))) || any(!is.finite(as.matrix(frame))))
    stop("Exported data must contain only finite numeric values: ", path)
  if (any(!frame$x1 %in% c(0, 1)) || any(frame$x2 < 0 | frame$x2 > 2) ||
      any(as.matrix(frame[paste0("z",1:8)]) < 0 | as.matrix(frame[paste0("z",1:8)]) > 2))
    stop("Exported covariates are outside the simulation support: ", path)
  list(y=frame$y, x=as.matrix(frame[c("x1","x2")]),
       z=as.matrix(frame[paste0("z",1:8)]), m=frame$true_m)
}

summarize_results <- function(raw, output_dir) {
  keys <- unique(raw[c("case","n","tau","method")])
  values <- lapply(seq_len(nrow(keys)), function(i) {
    key <- keys[i,]; rows <- raw[raw$case==key$case & raw$n==key$n &
                                  raw$tau==key$tau & raw$method==key$method,]
    data.frame(key, bias=mean(rows$theta1)-1, sd=sd(rows$theta1),
               coverage=mean(rows$covered_theta1), rmse=mean(rows$rmse_m),
               successful_reps=nrow(rows))
  })
  summary <- do.call(rbind, values)
  write.csv(summary, file.path(output_dir, "simulation_summary_long.csv"), row.names=FALSE)
  summary$bias_sd <- ifelse(is.na(summary$sd), sprintf("%.4f (NA)", summary$bias),
                            sprintf("%.4f (%.4f)", summary$bias, summary$sd))
  for (entry in list(c("table1_bias","bias"), c("table1_sd","sd"),
                     c("table2_coverage","coverage"), c("table3_rmse","rmse"))) {
    selected <- summary[c("case","n","tau","method",entry[2])]
    selected$column <- paste0("tau_",sprintf("%.2f",selected$tau),"_",selected$method)
    selected$tau <- NULL; selected$method <- NULL
    wide <- reshape(selected, idvar=c("case","n"), timevar="column", direction="wide")
    names(wide) <- sub(paste0("^",entry[2],"\\."), "", names(wide))
    write.csv(wide, file.path(output_dir, paste0(entry[1], ".csv")), row.names=FALSE)
  }
  selected <- summary[c("case","n","tau","method","bias_sd")]
  selected$column <- paste0("tau_",sprintf("%.2f",selected$tau),"_",selected$method)
  selected$tau <- NULL; selected$method <- NULL
  wide <- reshape(selected,idvar=c("case","n"),timevar="column",direction="wide")
  names(wide) <- sub("^bias_sd\\.","",names(wide))
  write.csv(wide,file.path(output_dir,"table1_bias_sd.csv"),row.names=FALSE)
  summary
}

result_key <- function(frame) {
  paste(frame$case, frame$n, frame$rep, sprintf("%.12g", frame$tau), frame$method, sep="|")
}

parse_coverage <- function(value, label) {
  if (is.logical(value) && !anyNA(value)) return(value)
  strings <- tolower(trimws(as.character(value)))
  if (anyNA(strings) || any(!strings %in% c("true", "false", "1", "0")))
    stop("Invalid coverage values in ", label)
  strings %in% c("true", "1")
}

expected_results <- function(args, methods) {
  expand.grid(case=args$cases, n=args$sample_sizes, rep=seq_len(args$repetitions),
              tau=args$taus, method=methods, KEEP.OUT.ATTRS=FALSE, stringsAsFactors=FALSE)
}

validate_results <- function(raw, expected, complete=TRUE, label="results") {
  required <- c("case", "n", "rep", "tau", "method", "theta1", "theta2",
                "se_theta1", "se_theta2", "lower_theta1", "upper_theta1",
                "lower_theta2", "upper_theta2", "covered_theta1", "covered_theta2",
                "rmse_m", "test_check_loss")
  if (!all(required %in% names(raw)))
    stop("Missing columns in ", label, ": ", paste(setdiff(required, names(raw)), collapse=", "))
  if (!nrow(raw)) stop("No rows in ", label)
  numeric_columns <- setdiff(required, c("method", "covered_theta1", "covered_theta2"))
  if (!all(vapply(raw[numeric_columns], is.numeric, logical(1))) ||
      any(!is.finite(as.matrix(raw[numeric_columns])))) stop("Nonfinite/nonnumeric values in ", label)
  if (any(raw$case != as.integer(raw$case) | raw$n != as.integer(raw$n) | raw$rep != as.integer(raw$rep)))
    stop("Noninteger simulation keys in ", label)
  keys <- result_key(raw); wanted <- result_key(expected)
  if (anyDuplicated(keys)) stop("Duplicate simulation keys in ", label)
  if (any(!keys %in% wanted)) stop("Unexpected simulation keys in ", label)
  if (complete && !setequal(keys, wanted))
    stop("Incomplete ", label, ": expected ", length(wanted), " rows; got ", length(keys))
  for (j in 1:2) {
    lower <- raw[[paste0("lower_theta", j)]]; upper <- raw[[paste0("upper_theta", j)]]
    if (any(lower > upper) || any(raw[[paste0("se_theta", j)]] < 0)) stop("Invalid intervals in ", label)
    column <- paste0("covered_theta", j)
    raw[[column]] <- parse_coverage(raw[[column]], paste(label, column))
    truth <- c(1, -1)[j]
    if (any(raw[[column]] != (lower <= truth & truth <= upper))) stop("Coverage/bounds mismatch in ", label)
  }
  if (any(raw$rmse_m < 0 | raw$test_check_loss < 0)) stop("Negative loss in ", label)
  raw[required]
}

replace_csv <- function(frame, path) {
  temporary <- tempfile(pattern="csv-", tmpdir=dirname(path))
  on.exit(unlink(temporary), add=TRUE)
  write.csv(frame, temporary, row.names=FALSE)
  # Immutable per-fit RDS checkpoints remain authoritative if interrupted
  # while replacing this convenient, cumulative CSV view on Windows.
  if (!file.copy(temporary, path, overwrite=TRUE)) stop("Cannot write ", path)
}

run_simulation <- function(args) {
  require_replication_packages()
  if (length(args$repetitions) != 1L || is.na(args$repetitions) || args$repetitions < 1L ||
      length(args$test_size) != 1L || is.na(args$test_size) || args$test_size < 1L ||
      anyNA(args$cases) || any(!args$cases %in% 1:3) || anyDuplicated(args$cases) ||
      anyNA(args$sample_sizes) || any(args$sample_sizes < 20L) || anyDuplicated(args$sample_sizes) ||
      anyNA(args$taus) || any(args$taus <= 0 | args$taus >= 1) || anyDuplicated(args$taus))
    stop("Invalid or duplicate simulation settings.")
  if (!nzchar(args$data_dir) || !nzchar(args$python_results))
    stop("Run Python first and provide its data directory and raw results for paired comparisons.")
  args$data_dir <- normalizePath(args$data_dir, mustWork=TRUE)
  args$python_results <- normalizePath(args$python_results, mustWork=TRUE)
  expected_data_dir <- normalizePath(file.path(dirname(args$python_results), "data_for_r"), mustWork=TRUE)
  if (!identical(args$data_dir, expected_data_dir))
    stop("Data exports and Python raw results must come from the same Python output folder.")
  python_config <- file.path(dirname(args$python_results), "run_config.json")
  if (!file.exists(python_config)) stop("Missing Python run configuration: ", python_config)
  py <- validate_results(read.csv(args$python_results), expected_results(args, c("LQR", "DPLQR")),
                         label="Python results")
  panels <- expand.grid(case=args$cases, n=args$sample_sizes, rep=seq_len(args$repetitions),
                        KEEP.OUT.ATTRS=FALSE)
  stems <- sprintf("case_%d_n_%d_rep_%04d", panels$case, panels$n, panels$rep)
  inputs <- c(file.path(args$data_dir, paste0(stems, "_train.csv.gz")),
              file.path(args$data_dir, paste0(stems, "_test.csv.gz")), args$python_results)
  if (any(!file.exists(inputs))) stop("Missing Python export: ", inputs[which(!file.exists(inputs))[1]])
  hashes <- tools::md5sum(inputs)
  if (anyNA(hashes)) stop("Cannot hash all Python input files.")
  dir.create(args$output_dir, recursive=TRUE, showWarnings=FALSE)
  args$output_dir <- normalizePath(args$output_dir, mustWork=TRUE)
  signature <- list(format_version=2L, arguments=args, input_hashes=hashes,
                    r_script_hash=tools::md5sum(file.path(script_dir, "simulate_homoscedastic.R")))
  manifest_path <- file.path(args$output_dir, "r_input_manifest.rds")
  checkpoint_dir <- file.path(args$output_dir, "r_checkpoints")
  if (file.exists(manifest_path)) {
    if (!identical(readRDS(manifest_path), signature))
      stop("R resume inputs, settings, or code have changed. Choose a new output directory.")
  } else {
    if (file.exists(file.path(args$output_dir, "raw_results_r.csv")) ||
        (dir.exists(checkpoint_dir) && length(list.files(checkpoint_dir, all.files=TRUE, no..=TRUE))))
      stop("Existing R results have no matching manifest. Choose a new output directory.")
    saveRDS(signature, manifest_path)
  }
  dir.create(checkpoint_dir, showWarnings=FALSE)
  capture.output(dput(args), file=file.path(args$output_dir, "run_config.R"))
  capture.output(sessionInfo(), file=file.path(args$output_dir, "r_session_info.txt"))
  expected <- expected_results(args, "PLAQR")
  checkpoints <- list.files(checkpoint_dir, pattern="\\.rds$", full.names=TRUE)
  rows <- lapply(checkpoints, readRDS)
  if (length(rows)) {
    existing <- validate_results(do.call(rbind, rows), expected, complete=FALSE, label="R checkpoints")
    completed <- result_key(existing)
    cat("Resuming", length(completed), "PLAQR fits.\n")
  } else completed <- character()
  started <- proc.time()[3]
  for (case in args$cases) for (n in args$sample_sizes) for (rep in seq_len(args$repetitions)) {
    set.seed(args$seed + case*10000000L + n*1000L + rep)
    stem <- sprintf("case_%d_n_%d_rep_%04d", case, n, rep)
    train <- read_exported_dataset(file.path(args$data_dir, paste0(stem, "_train.csv.gz")), floor(0.8*n))
    test <- read_exported_dataset(file.path(args$data_dir, paste0(stem, "_test.csv.gz")), args$test_size)
    for (tau in args$taus) {
      key_frame <- data.frame(case=case, n=n, rep=rep, tau=tau, method="PLAQR")
      key <- result_key(key_frame)
      if (key %in% completed) next
      true_m_tau <- test$m + qt(tau, df=3)
      f <- fit_plaqr(train, test, tau)
      row <- data.frame(key_frame, theta1=f$theta[1], theta2=f$theta[2],
        se_theta1=f$se[1], se_theta2=f$se[2],
        lower_theta1=f$lower[1], upper_theta1=f$upper[1], lower_theta2=f$lower[2], upper_theta2=f$upper[2],
        covered_theta1=f$lower[1] <= 1 & 1 <= f$upper[1],
        covered_theta2=f$lower[2] <= -1 & -1 <= f$upper[2],
        rmse_m=mean((f$mhat-true_m_tau)^2)/mean(true_m_tau^2),
        test_check_loss=checkLoss_mean(test$y-f$pred, tau))
      row <- validate_results(row, key_frame, label="PLAQR fit")
      checkpoint <- file.path(checkpoint_dir, paste0(gsub("|", "_", key, fixed=TRUE), ".rds"))
      temporary <- paste0(checkpoint, ".tmp")
      saveRDS(row, temporary)
      if (file.exists(checkpoint) || !file.rename(temporary, checkpoint)) stop("Cannot save checkpoint: ", checkpoint)
      rows[[length(rows)+1L]] <- row
      completed <- c(completed, key)
      replace_csv(do.call(rbind, rows), file.path(args$output_dir, "raw_results_r.csv"))
    }
    cat(sprintf("case=%d n=%d rep=%d/%d PLAQR fits=%d elapsed=%.1f min\n", case, n, rep,
                args$repetitions, length(completed), (proc.time()[3]-started)/60))
  }
  raw_r <- validate_results(do.call(rbind, rows), expected, label="R results")
  replace_csv(raw_r, file.path(args$output_dir, "raw_results_r.csv"))
  raw <- rbind(py, raw_r)
  raw <- raw[order(raw$case, raw$n, raw$tau, match(raw$method, c("LQR", "PLAQR", "DPLQR")), raw$rep), ]
  replace_csv(raw, file.path(args$output_dir, "raw_results_combined.csv"))
  summarize_results(raw, args$output_dir)
}

if (sys.nframe() == 0) {
  if (identical(commandArgs(trailingOnly=TRUE), "--density-only")) {
    residuals <- scan(file("stdin"), what=double(), quiet=TRUE)
    cat(sprintf("%.17g\n", residual_density_zero(residuals)))
  } else run_simulation(parse_args())
}
