use crate::kernel::kernel::Kernel;
use crate::dataset::{DataManager, UnitStandardizedDataset};
use crate::constraint::constraints::{Constraints};
use crate::constraint::constraint::{Constraint};

use std::sync::Arc;

use ndarray::{Array2, Array1, ArrayView2};
use extendr_api::prelude::*;

use crate::kernel::utils::parse_kernel_recursive;


#[allow(non_snake_case)]
#[extendr]
pub struct SparseGPRegression {
    pub kernel : Box<dyn Kernel>,
    pub dataset : Arc<dyn DataManager>,
    pub stale : bool,
    pub response: Array1<f64>,
    pub n: usize,
    pub m: usize,
    pub K_nm: Array2<f64>,
    pub K_mm: Array2<f64>,
    pub X_inducing : Array2<f64>,
    pub sigma : f64,
    pub n_params : usize,
    pub constraints : Constraints
}

#[extendr]
impl SparseGPRegression {
    #[allow(non_snake_case)]
    pub fn new(
        X: ArrayView2<f64>,
        X_inducing: ArrayView2<f64>,
        y: &[f64],
        kernel_specification: List,
    ) -> Self {
        let y = Array1::from(y.to_owned());
        //let y_bar = y.iter().sum::<f64>() / (y.len() as f64);
        // y -= y_bar;


        let X = X.to_owned();
        let dataset = Arc::new(UnitStandardizedDataset::new(X,y.clone()));

        let kernel = parse_kernel_recursive(kernel_specification, dataset.as_ref());

        let X_inducing = dataset.scale_predictors(X_inducing);

        let n = y.len();
        let m = X_inducing.nrows();

        let n_params = kernel.num_params() + 1;


        let kernel_constraints = kernel.rec_constraint();
        let mut constraints = vec![Constraint::new_positive_sp()];
        constraints.extend(kernel_constraints);

        Self {
            dataset,
            response : y,
            X_inducing,
            n,
            m,
            n_params,
            kernel,
            stale : true,
            K_nm: Array2::zeros((n, m)),
            K_mm: Array2::zeros((m, m)),
            sigma: 1.0,
            constraints : Constraints::new(constraints)
        }
    }
}

extendr_module! {
    mod sparse_gp_reg;
    impl SparseGPRegression;
}
