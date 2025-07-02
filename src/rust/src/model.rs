use ndarray::Array1;
use extendr_api::prelude::*;
use crate::constraint::constraints::Constraints;
use crate::predict::PredictionOutput;
use ndarray::ArrayView2;
use crate::dual_number::Dual;

use crate::constraint::constraint::Constraint;

use std::f64::consts::PI;
use crate::kernel::kernel::Kernel;
use crate::likelihood::Likelihood;

pub trait ClonableModel : Model {
    fn clone_box(&self) -> Box<dyn ClonableModel>;
}

impl Clone for Box<dyn ClonableModel> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

pub trait Model : Send + Sync {
    fn get_n_params(&self) -> usize;

    fn recommend_constraints(&self) -> Constraints;

    fn set_params(&mut self, params: &[f64]);

    fn update(&mut self);

    fn predict(&self, prediction_points: ArrayView2<f64>, sub_kernel: Option<String>) -> PredictionOutput;

    fn log_like(&self, gradient: bool) -> Dual;

    fn display_kernel(&self) -> String;
    }