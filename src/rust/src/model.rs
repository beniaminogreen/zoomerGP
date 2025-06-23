use extendr_api::prelude::*;
use crate::constraint::constraints::Constraints;
use crate::predict::PredictionOutput;
use ndarray::ArrayView2;
use crate::dual_number::Dual;


pub trait Model {
    fn get_n_params(&self) -> usize;

    fn recommend_constraints(&self) -> Constraints;

    fn set_params(&mut self, params: &[f64]);

    fn update(&mut self);

    fn predict(&self, prediction_points: ArrayView2<f64>, sub_kernel: Option<String>) -> PredictionOutput;

    fn log_like(&self, gradient: bool) -> Dual;
}