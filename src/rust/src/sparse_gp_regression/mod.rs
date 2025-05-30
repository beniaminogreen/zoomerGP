use extendr_api::prelude::*;
mod sparse_gp_reg;

extendr_module! {
    mod sparse_gp_regression;
    use sparse_gp_reg;
 }