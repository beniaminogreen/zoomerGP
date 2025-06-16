use ndarray::{Array1, ArrayView1};
use super::kernel::Kernel;

use crate::dual_number::Dual;
use crate::constraint::constraint::Constraint;
use std::fmt::Debug;

#[derive(Debug)]
pub struct LinearKernel {
    sigmas: Array1<f64>,
    n_params: usize,
    operating_columns: Vec<usize>,
}

impl LinearKernel {
    pub fn new(operating_columns: Vec<usize>) -> Self {
        Self {
            sigmas: Array1::ones(operating_columns.len()),
            n_params: operating_columns.len(),
            operating_columns,
        }
    }
}

impl Kernel for LinearKernel {
    fn rec_constraint(&self) -> Vec<Constraint> {
        vec![Constraint::new_positive_sp(); self.n_params]
    }

    fn num_params(&self) -> usize {
        self.n_params
    }

    fn set_params(&mut self, params: &[f64]) {
        self.sigmas = Array1::from(params.to_owned());
    }

    fn calc(&self, x: ArrayView1<f64>, y: ArrayView1<f64>, gradient : bool) -> Dual {
        let products = self.operating_columns.iter().map(|i| (x[*i] * y[*i]));

        let x = self.sigmas
            .iter()
            .zip(products.clone())
            .map(|(a, b)| a.powi(2) * b)
            .sum();

        if !gradient {
            return Dual::from(x)
        };

        let grads = self
            .sigmas
            .iter()
            .zip(products)
            .map(|(sigma, prod)| 2.0 * sigma * prod);

        let grads = Array1::from_iter(grads);

        Dual::from((x, grads))
    }

    fn children(&self) -> &[Box<dyn Kernel>] {
        &[]
    }

    fn describe(&self) -> String {
        "Linear Kernel".to_string()
    }
}

