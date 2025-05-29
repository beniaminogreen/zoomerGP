use extendr_api::prelude::*;
use super::gp_reg::GPRegression;
use crate::dataset::{UnitStandardizedDataset};

use crate::coin_optimizer::CoinOptimizer;

use crate::kernel::utils::{parse_kernel_recursive, print_kernel_tree};

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView2};

#[extendr]
impl GPRegression {
    pub fn r_new(x: ArrayView2<f64>, y: &[f64], kernel_specification: List) -> Self {
        let y = Array1::from(y.to_owned());
        let x = x.to_owned();

        let kernel = parse_kernel_recursive(kernel_specification);
        let dataset = Arc::new(UnitStandardizedDataset::new(x,y.clone()));

        Self::new(kernel, dataset, y, true)
    }

    pub fn r_set_params(&mut self, params : &[f64]) {
        self.set_params(params);
    }

    pub fn r_get_n_params(&self) -> u64 {
        self.get_n_params() as u64
    }


    pub fn r_update(&mut self) {
        self.update();
    }

    pub fn r_log_like(&self) -> f64 {
        self.log_like(false).x
    }

    pub fn r_log_like_and_grad(&self) -> List {
        let log_like_and_grad = self.log_like(true);

        list!(ll = log_like_and_grad.x, ll_grad = Robj::try_from(log_like_and_grad.grad.unwrap()))
    }

    pub fn r_display_kernel(&self) {
        print_kernel_tree(self.kernel.as_ref(), "", true);
    }

    pub fn r_optimize(&mut self, max_iter : u32, use_constraints : bool) {
        let optimizer = CoinOptimizer::new(self, use_constraints);
        optimizer.run(max_iter as usize);
    }
    
    pub fn r_predict(&self, x: ArrayView2<f64>) -> Vec<f64> {
        self.predict(x.view()).to_vec()
    }

    pub fn r_constrain_parameters(&self, params : &[f64]) -> List {
        todo!()
    }

}


extendr_module!{
    mod r_methods;
    impl GPRegression;
}

