pub mod sparse_latent_gp_reg;
pub mod sparse_latent_model;

use extendr_api::prelude::*;

extendr_module! {
    mod sparse_latent_gp_regression;
    use sparse_latent_gp_reg;
    use sparse_latent_model;
 }