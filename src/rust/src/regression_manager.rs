use crate::dataset::{DataManager, UnitStandardizedDataset};
use crate::gp_regression::gp_regression::GPRegression;

use crate::kernel::utils::parse_kernel_recursive;

use extendr_api::prelude::*;
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView2};

use crate::dual_number::Dual;

pub trait Regression{
    fn get_n_params(&self) -> usize;
    fn set_params(&mut self, params : &[f64]);
    fn update(&mut self);
    fn log_like(&self, gradient : bool) -> Dual;
}

// Is this even necessary? No, because the MLGaussian is adding nothing over the base class.
// Perhaps we can get away with exposing these methods directly to R from the GPRegression struct?
// Only issue here is that this might give us multiple extendr impl blocks, but we can get around
// this with multiple modules.

#[extendr]
struct MLGaussian {
    dataset : Arc<dyn DataManager>,
    model : GPRegression,
}


#[extendr]
impl MLGaussian{
    fn new(x: ArrayView2<f64>, y: &[f64], kernel_specification: List) -> Self {
        let y = Array1::from(y.to_owned());
        let x = x.to_owned();

        let kernel_func = parse_kernel_recursive(kernel_specification);
        let dataset = Arc::new(UnitStandardizedDataset::new(x,y));

        Self {
            dataset : dataset.clone(),
            model : GPRegression::new(kernel_func, dataset)
        }

    }
}


extendr_module!{
    mod regression_manager;
    impl MLGaussian;
}
