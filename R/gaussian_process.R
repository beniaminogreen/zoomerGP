instantiate_gp <- function(form, data, sparse = F, n_points = 50, noiseless = F) {
  parsed_formula <- r_parse_formula(form, data)
  outcome <- as.numeric(data[[parsed_formula$outcome]])
  colnames <- names(data)[parsed_formula$columns]
  specification <- parsed_formula$specification
  
  x <- list(
    y = outcome,
    n = length(outcome),
    formula = form,
    colnames = colnames,
    X = as.matrix(data[,colnames]),
    specification = specification,
    sparse = sparse
  )
  if (sparse) {
    x$X_inducing <- as.matrix(x$X[sample(1:nrow(x$X),n_points), ])
    gpr_object <- SparseGPRegression$new(x$X, x$X_inducing, x$y, specification)
  } else {
    gpr_object <- GPRegression$r_new(x$X, x$y, specification, noiseless)
  }

  x[['gpr_object']] <- gpr_object
  class(x) <- "gaussian_process"
  
  
  return(x)
}

#' Fit a Gaussian process in R
#'
#' @param form the regression formula. See the vingette on the gaussian process
#' formula sub-language the package uses.
#' @param data the dataframe from which to source regression data.
#' @param sparse a boolean indicating whether to use the projected process
#' approximation to speed up training. Recommended for N>500.
#'
#' @param n_points The number of inducing points used for the projected process
#' approximation. Defaults to 5. If set too large, may cause numerical
#' stability issues when calculating the gradient of the marginal likelihood.
#'
#' @export
gaussian_process <- function(form, data, sparse=F, n_points = 5, training_method = "bfgs", noiseless = F) {
  x <- instantiate_gp(form, data, sparse, n_points, noiseless)
  gp_taining_loop(x$gpr_object, sparse, training_method)
  
  return(x)
}