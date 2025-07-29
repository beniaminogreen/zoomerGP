#' Fit a logistic Gaussian process in R for a binary outcome
#' 
#' @param form the regression formula for the Gaussian process. The left-hand-side 
#' gives determines outcome you are fitting the GP to, and the right-hand side gives the kernel specification. 
#' 
#' @param data the dataframe from which to source regression data.
#' 
#' @param n_particles number of particles to be used in the stein variational gradient descent simulation
#' 
#' @param sparse a Boolean indicating whether to use the projected process
#' approximation to speed up training. Recommended for N>100.
#'
#' @param n_points The number of inducing points used for the projected process
#' approximation. Defaults to 5. If set too large, may cause numerical
#' stability issues when calculating the gradient of the marginal likelihood.
#'
#' @export
logistic_gaussian_process <- function(formula, data, sparse= T, n_points = 10, n_particles = 10, n_iter = 2000) {
  parsed_formula <- r_parse_formula(formula, data)
  outcome <- as.numeric(as.logical(data[[parsed_formula$outcome]]))
  colnames <- names(data)[parsed_formula$columns]
  specification <- parsed_formula$specification
  
  X <- as.matrix(data[,colnames])
  colnames(X) <- colnames
  
  x <- list(
    y = outcome,
    n = length(outcome),
    formula = formula,
    colnames = colnames,
    X = X,
    specification = specification,
    sparse = sparse, 
    likelihood = "binomial likelihood (logit link)"
  )
  
  if (sparse) {
    x$X_inducing <- as.matrix(x$X[sample(1:nrow(x$X),n_points), ])
    coin_latent_gpr <- CoinSVGP$r_new_sparse_latent(X, outcome, x$X_inducing, specification, n_particles)
  } else {
    coin_latent_gpr <- CoinSVGP$r_new_latent(X, outcome, specification, n_particles)
  }
  
  coin_latent_gpr$run(n_iter)
  
  x[['gpr_object']] <- coin_latent_gpr
  class(x) <- "gaussian_process"
  
  x
}
