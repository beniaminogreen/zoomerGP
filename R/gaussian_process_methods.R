#' Print information about a gaussian process
#'
#' @method print gaussian_process
#' @export
print.gaussian_process <- function(x, ...) {
  if (x$sparse) {
    cat(sprintf("A Gaussian Process model with %d observations\n", x$n))
    cat("Using the projected process approximation\n")
  } else {
    cat(sprintf("A Gaussian Process model with %d observations\n", x$n))
  }
  cat(sprintf("Log-Likelihood: %f\n", x$ll))
  cat(sprintf("R-Squared: %f\n", x$rsq))
  cat("Kernel Specification:\n")
  cat(x$gpr_object$display_kernel())
}

#' Predict Function Values at a Test Point
#'
#' @method predict gaussian_process
#' 
#' @param newdata a new data set to generate predictions for. Must have all of 
#' the predictor columns of the original data set or an error will be returned. 
#' 
#' @param kernel_str the label of the kernel you would like to extract 
#' predictions for. This can be useful if you are (for example) attempting to 
#' isolate a specific signal from a data set such as a periodic component from a 
#' time series.  Use the print method of the Gaussian process to determine the 
#' labels for the individual kernels. 
#' 
#' 
#' @export
predict.gaussian_process <- function(object, newdata = NULL, kernel_str=NA,...) {
  if (is.null(newdata)) {
    preds <- object$gpr_object$predict(object$X, kernel_str)
    n_pred_points <- object$n
  } else{
    newdata <- as.matrix(newdata[,object$colnames])
    preds <- object$gpr_object$predict(newdata, kernel_str)
    n_pred_points <- nrow(newdata)
  }
  
  out <- matrix(nrow=n_pred_points, ncol=3)
  out[,1] <- preds$get_response()
  out[,2] <- preds$get_f_var()
  out[,3] <- preds$get_pred_var()
  colnames(out) <- c("prediction", "f_var", "y_var")
   
  return(out)
}