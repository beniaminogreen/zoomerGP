#' Print information about a gaussian process
#'
#' @method print gaussian_process
#' @export
print.gaussian_process <- function(x, ...) {
  if (x$sparse) {
    print(stringr::str_glue("A Gaussian Process model with {x$n} observations"))
    print("Using the projected porcess approximation")
  } else {
    print(stringr::str_glue("A Gaussian Process model with {x$n} observations"))
  }
  cat("Kernel Specification:\n")
  cat(x$gpr_object$display_kernel())
}

#' Predict Function Values at a Test Point
#'
#' @method predict gaussian_process
#' @export
predict.gaussian_process <- function(x, newdata = NULL, kernel_str = NA) {
  
  if (is.null(newdata)) {
    preds <- x$gpr_object$predict(x$X, kernel_str)
    n_pred_points <- x$n
  } else{
    newdata <- as.matrix(newdata[,x$colnames])
    preds <- x$gpr_object$predict(newdata, kernel_str)
    n_pred_points <- nrow(newdata)
  }
  
  out <- matrix(nrow=n_pred_points, ncol=3)
  out[,1] <- preds$get_response()
  out[,2] <- preds$get_f_var()
  out[,3] <- preds$get_pred_var()
  colnames(out) <- c("prediction", "f_var", "y_var")
   
  return(out)
}