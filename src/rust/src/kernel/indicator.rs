use ndarray::{ArrayView1, Array1};
use super::kernel::Kernel;

use crate::dual_number::Dual;
use crate::constraint::constraint::Constraint;
use std::fmt::Debug;

#[derive(Debug)]
#[derive(Clone)]
pub struct IndicatorKernel {
    operating_columns: Vec<usize>,
    sigma: f64,
    n_params: usize,
}

impl IndicatorKernel {
    pub fn new(operating_columns: Vec<usize>) -> Self {
        Self {
            sigma: 1.0,
            n_params: 1,
            operating_columns,
        }
    }
}

impl Kernel for IndicatorKernel {
    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
    fn rec_constraint(&self ) -> Vec<Constraint> {
        vec![Constraint::new_positive_sp(); 1]
    }

    fn num_params(&self) -> usize {
        self.n_params
    }

    fn set_params(&mut self, params: &[f64]) {
        self.sigma = params[0];
    }

    fn calc(&self, x: ArrayView1<f64>, y: ArrayView1<f64>, gradient: bool) -> Dual {
        let all_eq = self.operating_columns.iter().all(|i| x[*i] == y[*i]);

        let x = if all_eq {
            self.sigma.powi(2)
        } else {
            0.0
        };

        if !gradient {
            return Dual::from(x)
        }

        let grad = if all_eq {
            Array1::from(vec![2.0 * self.sigma])
        } else {
            Array1::from(vec![0.0])
        };

        Dual::from((x,grad))
    }

    fn children(&self) -> &[Box<dyn Kernel>] {
        &[]
    }

    fn describe(&self) -> String {
        "Indicator Kernel".to_string()
    }

    fn log_prior(&self, gradient : bool) -> Dual {
        todo!()
    }
}

