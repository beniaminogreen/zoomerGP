use extendr_api::prelude::*;
pub mod gp_reg;
pub mod r_methods;

extendr_module!{
    mod gp_regression;
    use r_methods;
}
