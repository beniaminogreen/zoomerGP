use extendr_api::prelude::*;
pub mod gp_reg;
mod gp_reg_model;

extendr_module!{
    mod gp_regression;
    use gp_reg;
    use gp_reg_model;
}
