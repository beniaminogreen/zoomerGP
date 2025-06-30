use std::sync::Arc;
use ndarray::{Array1, Array2, ArrayView2};
use crate::constraint::constraint::Constraint;
use crate::constraint::constraints::Constraints;
use crate::dataset::{DataManager, UnitStandardizedDataset};
use crate::kernel::kernel::Kernel;
use extendr_api::prelude::*;
use crate::likelihood::{Likelihood, BinomialLogit};
use crate::kernel::utils::parse_kernel_recursive;

#[allow(non_snake_case)]
#[extendr]
pub struct SparseLatentGPR{
    pub kernel : Box<dyn Kernel>,
    pub dataset : Arc<dyn DataManager>,
    pub v : Array1<f64>,
    pub n_params : usize,
    pub stale : bool,
    pub constraints : Constraints,
    pub X_inducing : Array2<f64>,
    pub L : Array2<f64>,
    pub K_mm : Array2<f64>,
    pub K_nm : Array2<f64>,
    pub likelihood : Box<dyn Likelihood>
}

#[extendr]
impl SparseLatentGPR{
    pub fn r_new(x: ArrayView2<f64>, y: &[f64], x_inducing : ArrayView2<f64>, kernel_specification: List) -> Self {
        let y = Array1::from(y.to_owned());
        let x = x.to_owned();

        let dataset = Arc::new(UnitStandardizedDataset::new(x,y.clone()));
        let kernel = parse_kernel_recursive(kernel_specification, dataset.as_ref());

        Self::new(kernel, dataset, x_inducing, y)
    }
}
impl SparseLatentGPR {
    pub fn new(kernel: Box<dyn Kernel>, dataset: Arc<dyn DataManager>, x_inducing : ArrayView2<f64>, response: Array1<f64>) -> Self {
        let n = dataset.shape().0;
        let m = x_inducing.nrows();
        let n_params = kernel.num_params() + m;
        let stale = true;

        let v: Array1<f64> = Array1::zeros(m);

        let mut constraints = kernel.rec_constraint();
        let latent_constraints = vec![Constraint::new_no_constraint(); m];
        constraints.extend(latent_constraints);

        Self {
            n_params,
            dataset,
            constraints: Constraints::new(constraints),
            stale,
            v,
            L: Array2::zeros((m, m)),
            K_mm: Array2::zeros((m, m)),
            K_nm : Array2::zeros((n, m)),
            X_inducing : x_inducing.to_owned(),
            kernel,
            likelihood: Box::new(BinomialLogit { y: response.into_iter().map(|x| x == 1.0).collect() })
        }
    }
}

extendr_module! {
    mod sparse_latent_gp_reg;
    impl SparseLatentGPR;
}