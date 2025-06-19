test_grad <- function(form, data, use_constraints = T, ...) {
    gp <- instantiate_gp(form, data, ...)
    fn_and_grad <- generate_fn_and_grad(gp$gpr_object,use_constraints)

    params <- rnorm(gp$gpr_object$get_n_params(), 0,3)
    print(params)

    derivative_check_result <- nloptr::check.derivatives(
        params,
        fn_and_grad[[1]],
        fn_and_grad[[2]],
        check_derivatives_tol = .1,
        check_derivatives_print = "all"
    )
    expect_true(!any(derivative_check_result$flag_derivative_warning))
}
