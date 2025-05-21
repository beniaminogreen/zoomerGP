use ndarray::ArrayView1;
use std::fmt::Debug;

use crate::dual_number::Dual;
use crate::constraint::constraint::Constraint;

pub trait Kernel : Debug + Sync {
    fn rec_constraint(&self) -> Vec<Constraint>;
    fn num_params(&self) -> usize;
    fn set_params(&mut self, params: &[f64]);
    fn calc(&self, x: ArrayView1<f64>, y: ArrayView1<f64>, gradient : bool) -> Dual;
    fn children(&self) -> &[Box<dyn Kernel>];
    fn describe(&self) -> String;
}
