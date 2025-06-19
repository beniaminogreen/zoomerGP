instantiate_gp <- function(form, data, sparse = F, n_points = 50, noise = T) {
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
    gpr_object <- GPRegression$r_new(x$X, x$y, specification, noise)
  }

  x[['gpr_object']] <- gpr_object
  class(x) <- "gaussian_process"


  return(x)
}

#' Fit a Gaussian process in R
#'
#' @param form the regression formula for the gaussian process. The left-hand-side 
#' gives determines outcome you are fitting the GP to, and the right-hand side gives the kernel specification. 
#' 
#' @param data the dataframe from which to source regression data.
#' @param sparse a boolean indicating whether to use the projected process
#' approximation to speed up training. Recommended for N>500.
#'
#' @param n_points The number of inducing points used for the projected process
#' approximation. Defaults to 5. If set too large, may cause numerical
#' stability issues when calculating the gradient of the marginal likelihood.
#'
#' @examples
#' \donttest{
#' library(tibble)
#' library(ggplot2)
#' n <- 400
#' data <- tibble(
#'  x = runif(n), 
#'  y = runif(n), 
#'  z = x^2 + rnorm(n, 0,.1)
#' )
#' gp_model <- gaussian_process(z ~ rbf(x,y), data = data)
#' data$predictions <- predict(gp_model)[,1]
#' ggplot(data, aes(x=x)) + 
#'   geom_point(aes(y= z)) + 
#'   geom_line(aes(y=predictions))
#'  }
#' @export
gaussian_process <- function(form, data, sparse=F, n_points = 5, training_method = "bfgs", noise = T) {
  x <- instantiate_gp(form, data, sparse, n_points, noise)
  gp_taining_loop(x$gpr_object, sparse, training_method)
  
  x$ll <- x$gpr_object$log_like(FALSE)$get_x()
  
  TSS <- sum(((x$y)-mean(x$y))^2)
  RSS <- sum((predict(x)[,1]- x$y)^2)
  x$rsq <- 1 - RSS/TSS
  

  return(x)
}
