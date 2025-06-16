use super::kernel::Kernel;
use ndarray::{ArrayView1, Array1};

use crate::dual_number::Dual;
use crate::constraint::constraint::Constraint;

#[derive(Debug)]
pub struct ExpQuadKernel {
    sigma: f64,
    lengthscales: Array1<f64>,
    n_params: usize,
    operating_columns: Vec<usize>,
}

impl ExpQuadKernel {
    pub fn new(operating_columns: Vec<usize>) -> Self {
        Self {
            sigma: 1.0,
            lengthscales: Array1::ones(operating_columns.len()),
            n_params: operating_columns.len() + 1,
            operating_columns,
        }
    }
}

impl Kernel for ExpQuadKernel {
    fn rec_constraint(&self ) -> Vec<Constraint> {
        vec![Constraint::new_positive_sp(); self.n_params]
    }

    fn num_params(&self) -> usize {
        self.n_params
    }

    fn set_params(&mut self, params: &[f64]) {
        let (sigma, lengthscales) = params.split_at(1);
        self.sigma = sigma[0];
        self.lengthscales = Array1::from(lengthscales.to_owned());
    }

    fn calc(&self, x: ArrayView1<f64>, y: ArrayView1<f64>, gradient: bool) -> Dual {
        let numerators = self
            .operating_columns
            .iter()
            .map(|i| (x[*i] - y[*i]).powi(2));

        let ratio: f64 = numerators
            .zip(self.lengthscales.iter())
            .map(|(num, denom)| num / (denom.powi(2)))
            .sum();

        let result = self.sigma.powi(2) * ((-1.0 * ratio).exp());

        if !gradient {
            return Dual::from(result)
        };

        let taus = self.operating_columns.iter().map(|i| (x[*i] - y[*i]));

        let mut grad_lengthscales: Vec<f64> = taus
            .zip(self.lengthscales.iter())
            .map(|(tau, l)| result * 2.0 * (tau.powi(2)) / (l.powi(3)))
            .collect();
        let grad_sigma = 2.0 * result / self.sigma;

        // add the grad_sigma as the first item
        grad_lengthscales.insert(0, grad_sigma);

        let grad = Array1::from_vec(grad_lengthscales);

        Dual::from((result,grad))
    }

    fn children(&self) -> &[Box<dyn Kernel>] {
        &[]
    }

    fn describe(&self) -> String {
        "RBF Kernel".to_string()
    }
}

