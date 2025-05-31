clip_gradient <- function(grad, max_grad = 10) {
  ifelse(abs(grad)<max_grad, grad, sign(grad)*max_grad)
}

bfgs_training <- function(fn, grad, num_params, sparse=F) {
  if(sparse){
    control = list(fnscale = -1, factr = 1e11, maxit = 300, lmm=50)
  } else {
    control =  list(fnscale = -1, lmm=30)
  }
  
  optim_output <- stats::optim(
    stats::rexp(num_params, .5),
    fn=fn,
    gr=grad,
    method = "L-BFGS-B",
    lower = rep(0.01, num_params),
    upper = rep(100, num_params),
    control = control
  )
  
  stopifnot("Optimization did not converge!" = optim_output$convergence == 0)
  
  return(optim_output$par)
}

rgenoud_training <- function(fn, grad, num_params, sparse =F){
  domain <- matrix(c(rep(0.001, num_params), rep(5, num_params)), ncol=2)
  gen_out <- rgenoud::genoud(
    fn,gr = grad,
    nvars = num_params,
    Domains = domain,
    max = T,
    pop.size = 130,
    max.generations = 90,
    boundary.enforcement = 2,
    solution.tolerance=0.2,
    print.level =  0,
    gradient.check = FALSE)
  
  return(gen_out$par)
  
}



gp_taining_loop <- function(rust_gpr_object, sparse = F, training_method = c("bfgs", "rgenoud", "coin")) {
  gpr_model <- rust_gpr_object
  num_params <- gpr_model$get_n_params()
  
  function_to_optimize <- function(params){
    gpr_model$set_params(params)
    gpr_model$update()
    return(gpr_model$log_like(TRUE))
  }
  
  memoized_func <- memoise::memoise(function_to_optimize)
  
  fn <- function(param) {
    memoized_func(param)$get_x()
  }
  
  grad <- function(param) {
    memoized_func(param)$get_grad()
  }
  
  
  training_method <- match.arg(training_method)
  if (training_method == "bfgs") {
    optim_output <- bfgs_training(fn, grad, num_params, sparse)
    gpr_model$set_params(optim_output)
  } else if (training_method == "rgenoud") {
    optim_output <- rgenoud_training(fn, grad, num_params, sparse)
    gpr_model$set_params(optim_output)
  } else {
    gpr_model$optimize(1000, T)
  }
  
  gpr_model$update()
  
  return(gpr_model)
}
