use crate::kernel::kernel::Kernel;
use crate::dataset::{DataManager, UnitStandardizedDataset};
use crate::constraint::constraints::{Constraints};
use crate::constraint::constraint::Constraint;


use std::sync::Arc;

use ndarray::{Array2, Array1, ArrayView2};

use extendr_api::prelude::*;

use crate::coin_optimizer::CoinOptimizer;
use crate::kernel::utils::{build_kernel_tree, parse_kernel_recursive};

#[allow(non_snake_case)]
#[extendr]
#[derive(Clone)]
pub struct GPRegression{
    pub kernel : Box<dyn Kernel>,
    pub dataset : Arc<dyn DataManager>,
    pub response : Array1<f64>,
    pub sigma : f64,
    pub n_params : usize,
    pub noise : bool,
    pub stale : bool,
    pub K: Array2<f64>,
    pub K_inv: Array2<f64>,
    pub constraints : Constraints
}

#[allow(non_snake_case)]
impl GPRegression {
    pub fn new(kernel : Box<dyn Kernel>, dataset : Arc<dyn DataManager>, response : Array1<f64>, noise : bool) -> Self {
        let n = dataset.shape().0;

        let n_params = if noise {
            kernel.num_params() + 1
        } else {
            kernel.num_params()
        };

        let kernel_constraints = kernel.rec_constraint();
        let mut constraints = vec![Constraint::new_positive_sp()];
        constraints.extend(kernel_constraints);



        Self{
            dataset,
            stale : true,
            response,
            noise,
            n_params,
            K_inv : Array2::zeros((n,n)),
            K : Array2::zeros((n,n)),
            sigma : 1.0,
            constraints : Constraints::new(constraints),
            kernel,
        }
    }
}

#[extendr]
impl GPRegression {
    pub fn optimize(&mut self, max_iter : u32, use_constraints : bool) {
        let optimizer = CoinOptimizer::new(self, use_constraints);
        optimizer.run(max_iter as usize);
    }

    pub fn r_new(x: ArrayView2<f64>, y: &[f64], kernel_specification: List, noise: bool) -> Self {
        let y = Array1::from(y.to_owned());
        let x = x.to_owned();

        let dataset = Arc::new(UnitStandardizedDataset::new(x,y.clone()));
        let kernel = parse_kernel_recursive(kernel_specification, dataset.as_ref());

        Self::new(kernel, dataset, y, noise)
    }

}



extendr_module! {
    mod gp_reg;
    impl GPRegression;
}
