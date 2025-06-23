use extendr_api::prelude::*;
pub mod gp_reg;
mod model;

extendr_module!{
    mod gp_regression;
    use gp_reg;
    use model;
}
