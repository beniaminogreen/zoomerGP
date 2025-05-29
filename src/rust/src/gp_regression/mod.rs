use extendr_api::prelude::*;
pub mod gp_reg;

extendr_module!{
    mod gp_regression;
    use gp_reg;
}
