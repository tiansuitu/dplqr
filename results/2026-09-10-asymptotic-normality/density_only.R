# Exact density function from the September 4 simulation. Base R only.
# No benchmark code, packages, or simulation entry point is sourced.
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
residuals <- scan(file("stdin"), what=double(), quiet=TRUE)
cat(sprintf("%.17g\n", residual_density_zero(residuals)))
