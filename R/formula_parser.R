expr_type <- function(x) {
  if (rlang::is_syntactic_literal(x)) {
    "constant"
  } else if (is.symbol(x)) {
    "symbol"
  } else if (is.call(x)) {
    "call"
  } else if (is.pairlist(x)) {
    "pairlist"
  } else {
    typeof(x)
  }
}

r_parse_formula <- function(form, data) {
     outcome <- tidyselect::eval_select(rlang::f_lhs(form), data)
     specification <- rlang::f_rhs(form)
     outcome_index <- tidyselect::eval_select(outcome, data)

     seen_columns <- c()
     recursive_parse_kernel <- function(x) {
      switch(expr_type(x),
        call = {
            if (rlang::is_call(x, "+")) {
                list(
                     type = "add",
                     left = recursive_parse_kernel(x[[2]]),
                     right = recursive_parse_kernel(x[[3]]))
              } else if (rlang::is_call(x, "*")){
                list(
                     type = "multiply",
                     left = recursive_parse_kernel(x[[2]]),
                     right = recursive_parse_kernel(x[[3]]))
            } else {

                # Find the columns in the dataset that are referenced
                cols <- c()
                arg_names <- rlang::call_args_names(x)
                args <- rlang::call_args(x)
                
                kwargs <- list()
                for (i in seq(args)) {
                    if (arg_names[[i]] == ""){
                      cols <- c(cols, tidyselect::eval_select(args[[i]], data))
                    } else {
                      kwargs[arg_names[[i]]] <- as.numeric(eval(args[[i]]))
                    }
                }
                cols <- cols[cols != outcome_index]

                # Then add the ones that are not already in seen cols
                cols_to_add <- cols[!(cols %in% seen_columns)]
                seen_columns <<- c(seen_columns, cols_to_add)

                # Finally, get the indexes of these columns in seen_columns
                cols_in_array <- match(cols, seen_columns)

                list(
                     kernel = rlang::call_name(x),
                     cols = cols_in_array, 
                     kwargs = kwargs
                )
              }
        },
        pairlist = x,
        stop("Don't know how to handle type ", typeof(x), call. = FALSE)
      )
    }

    formula_specification = recursive_parse_kernel(specification)

    return(
           list(
                "columns" = seen_columns,
                "outcome" = outcome,
                "specification" = formula_specification
           )
           )
}
