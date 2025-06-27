pub mod latent_gp_reg;
pub mod latent_model;

use extendr_api::prelude::*;

extendr_module! {
    mod latent_gp_regression;
    use latent_gp_reg;
    use latent_model;
 }