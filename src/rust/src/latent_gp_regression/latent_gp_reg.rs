use std::sync::Arc;
use ndarray::{Array1, Array2, ArrayView2};
use crate::constraint::constraint::Constraint;
use crate::constraint::constraints::Constraints;
use crate::dataset::{DataManager, UnitStandardizedDataset};
use crate::kernel::kernel::Kernel;
use extendr_api::prelude::*;
use crate::dual_number::Dual;
use crate::likelihood::{Likelihood, BinomialLogit};
use crate::kernel::utils::{build_kernel_tree, parse_kernel_recursive};
use crate::sparse_latent_gp_regression::sparse_latent_gp_reg::SparseLatentGPR;

#[allow(non_snake_case)]
#[extendr]
#[derive(Clone)]
pub struct LatentGPR{
    pub kernel : Box<dyn Kernel>,
    pub dataset : Arc<dyn DataManager>,
    pub v : Array1<f64>,
    pub n_params : usize,
    pub stale : bool,
    pub constraints : Constraints,
    pub L : Array2<f64>,
    pub K : Array2<f64>,
    pub likelihood : Box<dyn Likelihood>
}

#[extendr]
impl LatentGPR{
    pub fn r_new(x: ArrayView2<f64>, y: &[f64], kernel_specification: List) -> Self {
        let y = Array1::from(y.to_owned());
        let x = x.to_owned();

        let dataset = Arc::new(UnitStandardizedDataset::new(x,y.clone()));
        let kernel = parse_kernel_recursive(kernel_specification, dataset.as_ref());

        Self::new(kernel, dataset, y)
    }
}
impl LatentGPR {
    pub fn new(kernel: Box<dyn Kernel>, dataset: Arc<dyn DataManager>, response: Array1<f64>) -> Self {
        let n = dataset.shape().0;
        let n_params = kernel.num_params() + n;
        let stale = true;

        let v: Array1<f64> = Array1::zeros(n);

        let mut constraints = kernel.rec_constraint();
        let latent_constraints = vec![Constraint::new_no_constraint(); n];
        constraints.extend(latent_constraints);



        Self {
            n_params,
            dataset,
            constraints: Constraints::new(constraints),
            stale,
            v,
            L: Array2::zeros((n, n)),
            K: Array2::zeros((n, n)),
            kernel,
            likelihood: Box::new(BinomialLogit { y: response.into_iter().map(|x| x == 1.0).collect() })
        }
    }

}



extendr_module! {
    mod latent_gp_reg;
    impl LatentGPR;
}