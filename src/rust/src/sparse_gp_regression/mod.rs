use extendr_api::prelude::*;
mod sparse_gp_reg;
mod sparse_gpr_model;

extendr_module! {
    mod sparse_gp_regression;
    use sparse_gp_reg;
    use sparse_gpr_model;
 }