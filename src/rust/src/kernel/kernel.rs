use ndarray::ArrayView1;
use std::fmt::Debug;

use crate::dual_number::Dual;
use crate::constraint::constraint::Constraint;

pub trait Kernel : Debug + Sync + Send {
    fn rec_constraint(&self) -> Vec<Constraint>;
    fn num_params(&self) -> usize;
    fn set_params(&mut self, params: &[f64]);
    fn calc(&self, x: ArrayView1<f64>, y: ArrayView1<f64>, gradient : bool) -> Dual;
    fn children(&self) -> &[Box<dyn Kernel>];
    fn describe(&self) -> String;
    fn log_prior(&self, gradient : bool) -> Dual;
    fn clone_box(&self) -> Box<dyn Kernel>;
}

impl Clone for Box<dyn Kernel> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}