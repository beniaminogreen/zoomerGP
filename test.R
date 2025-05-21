rextendr::document()
devtools::load_all()

n <- 4*10^2
x <- runif(n, 0,20)
data <- data.frame(
               x = x,
               y = sin(x) + rnorm(n,0,.4)
               )

form <- y ~ rbf(x)
parsed_form <- r_parse_formula(form, data = data)$specification
gp <- GPRegression$r_new(as.matrix(data), data$y, parsed_form)
gp$r_display_kernel()
gp$r_update()

gp$r_optimize(200, TRUE)



#
# new_x  <- data.frame(x = seq(0,40,.2))
#
# model <- gaussian_process(y ~ rbf(x), data = data, training_method = "coin", sparse = T, n_points = 30)
# lines(new_x$x, predict(model, new_x)[,1], col = "red")
#
# model_1 <- gaussian_process(
#                             y ~ rbf(x)*linear(x),
#                             data = data,
#                             training_method = "coin",
#                             sparse = T,
#                             n_points = 30)
# lines(new_x$x, predict(model_1, new_x)[,1], col = "green", lwd=3)
#
# model_2 <- gaussian_process(
#                             y ~ rbf(x)*linear(x),
#                             data = data,
#                             training_method = "coin",
#                             sparse = T,
#                             n_points = 30)
# lines(new_x$x, predict(model_2, new_x)[,1], col = "green", lwd=3)
#
# model_3 <- gaussian_process(y ~ spectral3(x),
#                             data = data,
#                             training_method = "rgen",
#                             sparse = T,
#                             n_points = 50)
# lines(new_x$x, predict(model_3, new_x)[,1], col = "blue", lwd=3)
#
#
#
# # gpr_model <- gp$gpr_object
# # num_params <- gpr_model$get_n_params()
# # function_to_optimize <- function(params){
# #     gpr_model$set_params(params)
# #     return(gpr_model$fit())
# # }
# #  memoized_func <- memoise::memoise(function_to_optimize)
# #  fn <- function(param) {
# #    memoized_func(param)$ll
# #  }
# #  grad <- function(param) {
# #    ll_grad <- memoized_func(param)$ll_grad
# #  }
# #
# #
# # domain <- matrix(c(rep(0.01, num_params), rep(100, num_params)), ncol=2)
# # gen_out <- genoud(fn, nvars = num_params, Domains = domain,max = T, gr = grad, pop.size = 100, max.generations = 90, boundary.enforcement = 2,solution.tolerance=0.1, gradient.check = FALSE)
# # gpr_model$set_params(gen_out$par)
# # gpr_model$fit()
# # gp_preds <- predict(gp, new_x)
# # plot(data$x, data$y, xlim = c(0,800), ylim = c(-5,5), col = "green")
# # lines(new_x, gp_preds[,1])
# # lines(new_x, gp_preds[,1] + 2*sqrt(gp_preds[,2]), lty = 2)
# # lines(new_x, gp_preds[,1] - 2*sqrt(gp_preds[,2]), lty = 2 )
# #
# # ?genoud
# #
# # # # x <- matrix(seq(0,50,length.out=400))
# # # # posterior <- gp$gpr_object$get_posterior(x)
# #
# # # # plot(data$x, data$y)
# # # # for (i in 1:100){
# # # #     samp <- MASS::mvrnorm(1, as.numeric(posterior$preds), posterior$vcov)
# # # #     lines(x, samp + mean(data$y), col=alpha(rgb(0,0,0), 0.1))
# # # # }
# #
# # # # rextendr::document()
# # # # devtools::load_all()
# # # # library(KRLS)
# #
# # # # n <- 800
# # # # p <- 15
# # # # X_mat <- matrix(runif(n*p), nrow=n, ncol=p)
# # # # X <- as.data.frame(X_mat)
# # # # noiseless_y <- (X$V1 > .5) * (X$V2 > .5)
# # # # y <- noiseless_y + rnorm(n,0,.05)
# # # # X$y <- y
# #
# # # # krls_out <- krls(X_mat,y)
# # # # krls_predictions <- predict(krls_out, X_mat)$fit
# # # # mean((krls_predictions - noiseless_y)^2)
# #
# # # # gp_out <- gaussian_process(y~rbf(everything()), data = X)
# # # # gp_out$gpr_object$dbg_kernel()
# # # # gp_preds <- predict(gp_out)[,1]
# # # # mean((gp_preds - noiseless_y)^2)
# #
# #
# # # # tibble(
# # # #        x = X[,1],
# # # #        y = X[,2],
# # # #        z= krls_predictions
# # # #        )  %>%
# # # #      ggplot(aes(x=x, y=y)) +
# # # #      geom_point(aes(col = z), size=5)
# #
# # # # tibble(
# # # #        x = X[,1],
# # # #        y = X[,2],
# # # #        z= gp_preds
# # # #        )  %>%
# # # #      ggplot(aes(x=x, y=y)) +
# # # #      geom_point(aes(col = z), size=5)
# #
