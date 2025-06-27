clip_gradient <- function(grad, max_grad = 10) {
  ifelse(abs(grad)<max_grad, grad, sign(grad)*max_grad)
}

bfgs_cg_training <- function(fn, grad, num_params, sparse=F, bfgs = T) {
  if(sparse){
    control = list(fnscale = -1, factr = 1e14, maxit = 500, lmm=50)
  } else {
    control =  list(fnscale = -1, lmm=30, maxit = 500)
  }

  if (bfgs) {
      method <- "L-BFGS-B"
  } else {
      method <- "CG"
  }

  optim_output <- stats::optim(
    stats::runif(num_params,-4,.5),
    fn=fn,
    gr=grad,
    method = method,
    # lower = rep(0.01, num_params),
    # upper = rep(100, num_params),
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
    # Domains = domain,
    max = T,
    pop.size = 130,
    max.generations = 100,
    boundary.enforcement = 2,
    solution.tolerance=0.2,
    print.level =  0,
    gradient.check = FALSE)

  return(gen_out$par)

}

generate_fn_and_grad <- function(gpr_model, use_constraints = TRUE) {
  function_to_optimize <- function(params){
     if (use_constraints) {
        constraints <- gpr_model$recommend_constraints()
        dual <- constraints$constrain(params)
        transformed_params <- dual$get_x()
        param_grad <- dual$get_grad()

        gpr_model$set_params(transformed_params)
    } else {
        gpr_model$set_params(params)
        param_grad <- rep(1, length(params))
    }
    gpr_model$update()
    log_like_dual <- gpr_model$log_like(TRUE)

    list(
         real = log_like_dual$get_x(),
         grad = log_like_dual$get_grad() * param_grad
         )
  }

  memoized_func <- memoise::memoise(function_to_optimize)

  fn <- function(param) {
    result <- memoized_func(param)$real
    
    if (!is.finite(result)) {
      return(-9999999.99999)
    } else
      return(result)
    }

  grad <- function(param) {
    memoized_func(param)$grad
  }

  return(list(fn, grad))
}


gp_taining_loop <- function(
  rust_gpr_object,
  sparse = FALSE,
  training_method = c("cg","bfgs", "rgenoud", "coin"),
  use_constraints=TRUE
) {

  gpr_model <- rust_gpr_object
  num_params <- gpr_model$get_n_params()

  fn_and_grad <- generate_fn_and_grad(gpr_model, use_constraints)
  fn <- fn_and_grad[[1]]
  grad <- fn_and_grad[[2]]
  
  training_method <- match.arg(training_method)
  
  if (training_method == "coin") {
    gpr_model$optimize(1000, T)
  } else { 
    if (training_method == "bfgs") {
      optim_output <- bfgs_cg_training(fn, grad, num_params, sparse, T)
    } else if (training_method == "cg") {
      optim_output <- bfgs_cg_training(fn, grad, num_params, sparse,F)
    } else if (training_method == "rgenoud") {
      optim_output <- rgenoud_training(fn, grad, num_params, sparse)
    } 
    
    if (!use_constraints) {
      gpr_model$set_params(optim_output)
    } else {
      constraints <- gpr_model$recommend_constraints()
      dual <- constraints$constrain(optim_output)
      gpr_model$set_params(dual$get_x());
    }
    
    gpr_model$update()
  }

  

  return(gpr_model)
}
