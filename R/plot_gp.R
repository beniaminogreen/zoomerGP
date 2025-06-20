#' Plot Gaussian Process Fit
#' 
#' @param x Gaussian Process object
#' @param col column name for the predictor you want to plot. All other 
#' predictors are set to their medians
#' 
#' @param add_points whether to plot datapoints in the background
#' 
#' @param ... additional arguments (ignored)
#'
#' @importFrom ggplot2 ggplot geom_point geom_line aes ylab xlab theme_bw
#' @method plot gaussian_process
#' @export
plot.gaussian_process <- function(x, col = 1, add_points = T, ...){
  X <- x$X
  medians <- apply(X,2, stats::median)
  n_preds <- 400
  
  min_val <- min(X[,col])
  max_val <- max(X[, col])
  
  predictor_matrix <- matrix(rep(medians, times = n_preds), nrow = n_preds, byrow = TRUE)
  colnames(predictor_matrix) <- colnames(X)
  
  predictor_matrix[,col] <- seq(min_val, max_val, length.out=n_preds)
  
  predictions <- predict(x, predictor_matrix)
  ci_upper <- predictions[,1] + 2*sqrt(predictions[,3])
  ci_lower <- predictions[,1] - 2*sqrt(predictions[,3])
  
  base_plot <- ggplot() 
  
  if (add_points) {
    base_plot <- base_plot + 
      geom_point(aes(x=X[,col], y = x$y), alpha = .4) 
  }
  
  base_plot + 
    geom_line(aes(x=predictor_matrix[,col], y = predictions[,1])) + 
    geom_line(aes(x=predictor_matrix[,col], y = ci_lower), alpha = .5) + 
    geom_line(aes(x=predictor_matrix[,col], y = ci_upper), alpha = .5) + 
    ylab("Outcome") + 
    xlab(col) + 
    theme_bw()
}