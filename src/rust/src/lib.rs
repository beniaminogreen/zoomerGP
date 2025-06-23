mod kernel;
mod constraint;
mod dual_number;
mod gp_regression;
mod dataset;
mod coin_optimizer;
mod sparse_gp_regression;
mod predict;
mod coin_svgd;
//mod regression_manager;

use extendr_api::prelude::*;

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod zoomerGP;
    use gp_regression;
    use dual_number;
    use constraint;
    use sparse_gp_regression;
    use predict;
    use coin_svgd;
}
