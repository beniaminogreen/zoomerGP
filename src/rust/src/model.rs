use ndarray::Array1;
use extendr_api::prelude::*;
use crate::constraint::constraints::Constraints;
use crate::predict::PredictionOutput;
use ndarray::ArrayView2;
use crate::dual_number::Dual;

use crate::constraint::constraint::Constraint;

use std::f64::consts::PI;


pub trait Model {
    fn get_n_params(&self) -> usize;

    fn recommend_constraints(&self) -> Constraints;

    fn set_params(&mut self, params: &[f64]);

    fn update(&mut self);

    fn predict(&self, prediction_points: ArrayView2<f64>, sub_kernel: Option<String>) -> PredictionOutput;

    fn log_like(&self, gradient: bool) -> Dual;
}
#[extendr]
pub struct TestModel {
    pub params : [f64; 2],
}

#[extendr]
impl Model for TestModel {
    fn get_n_params(&self) -> usize {2}

    fn set_params(&mut self, params: &[f64]) {
        self.params[0] = params[0];
        self.params[1] = params[1];
    }

    fn update(&mut self) {
    }

    fn predict(&self, prediction_points: ArrayView2<f64>, sub_kernel: Option<String>) -> PredictionOutput {
        todo!()
    }

    fn recommend_constraints(&self) -> Constraints {
        Constraints::new(vec![Constraint::new_no_constraint();2])
    }

    fn log_like(&self, gradient: bool) -> Dual {
        let log_like = -(self.params[0] + 4.0).powi(2) - (self.params[1]-2.0).powi(2);

        if !gradient {
            return Dual::from(log_like);
        }

        let mut grad = Array1::zeros(2);
        grad[0] = self.params[0] + 4.0;
        grad[1] = self.params[1] - 2.0;
        Dual::from((log_like,grad))
    }

    /*fn log_like(&self, gradient: bool) -> Dual {
        let dist_1 : f64 = (self.params[0] - 5.0).powi(2) + (self.params[1]-3.0).powi(2);
        let dist_2 : f64 = (self.params[0] - 8.0).powi(2) + (self.params[1]-3.0).powi(4);

        let w_1 = (0.25 / PI) * (-0.5 * dist_1).exp();
        let w_2 = (0.25 / PI) * (-0.5 * dist_2).exp();

        let p = w_1 + w_2;

        let log_like = (p).ln();

        if !gradient {
            return Dual::from(log_like);
        }

        let gamma_1 = w_1 / p;
        let gamma_2 = w_2 / p;

        let mut grad = Array1::zeros(2);

        grad[0] += gamma_1 * (self.params[0] - 5.0) + gamma_2 * (self.params[1]-8.0);
        grad[1] += gamma_1 * (self.params[0] - 3.0) + gamma_2 * (self.params[1]-3.0);


        Dual::from((log_like, grad))

    }
*/

}

extendr_module! {
    mod model;
    impl TestModel;
}