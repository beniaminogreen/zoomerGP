use super::kernel::Kernel;
use ndarray::{ArrayView1, Array1, Array2};
use crate::dual_number::Dual;

use std::f64::consts::PI;
const NEG_TWO_PI_SQ: f64 = -2.0 * PI * PI;

use crate::constraint::constraint::Constraint;

#[derive(Debug)]
pub struct SpectralMixtureKernel {
    weights: Array1<f64>,
    n_params: usize,
    operating_columns: Vec<usize>,
    sigmas: Array2<f64>,
    means: Array2<f64>,
    p: usize,
    q: usize,
}

impl SpectralMixtureKernel {
    pub fn new(operating_columns: Vec<usize>, q: usize) -> Self {
        let p = operating_columns.len();
        Self {
            weights: Array1::ones(q),
            n_params: q + 2 * q * p,
            means: Array2::ones((q, p)),
            sigmas: Array2::ones((q, p)),
            operating_columns,
            p,
            q,
        }
    }
}

impl Kernel for SpectralMixtureKernel {
    fn rec_constraint(&self ) -> Vec<Constraint> {
        vec![Constraint::new_positive_sp(); self.n_params]
    }

    fn num_params(&self) -> usize {
        self.n_params
    }

    fn set_params(&mut self, params: &[f64]) {
        let (weights, remainder) = params.split_at(self.q);
        self.weights = Array1::from(weights.to_owned());
        let (means, sigmas) = remainder.split_at(self.p * self.q);
        self.means = Array2::from_shape_vec((self.q, self.p), means.to_owned()).unwrap();
        self.sigmas = Array2::from_shape_vec((self.q, self.p), sigmas.to_owned()).unwrap();
    }

    fn calc(&self, x: ArrayView1<f64>, y: ArrayView1<f64>, gradient : bool) -> Dual {
        let taus: Vec<f64> = self
            .operating_columns
            .iter()
            .map(|i| x[*i] - y[*i])
            .collect();

        let mut total = 0.0;
        for q in 0..self.q {
            let mut prod = 1.0;
            for (p, tau) in taus.iter().enumerate() {
                prod *= (NEG_TWO_PI_SQ * tau.powi(2) * self.sigmas[[q, p]].powi(2)).exp()
                    * (2.0 * PI * tau * self.means[[q, p]]).cos();
            }
            total += self.weights[q] * prod;
        }

        if !gradient {
            return Dual::from(total)
        };

        let mut gradient: Array1<f64> = Array1::zeros(self.n_params);

        // first, calculate derivatives wrt to weights:
        for q in 0..self.q {
            let mut prod = 1.0;
            for (p, tau) in taus.iter().enumerate() {
                prod *= (NEG_TWO_PI_SQ * tau.powi(2) * self.sigmas[[q, p]].powi(2)).exp()
                    * (2.0 * PI * tau * self.means[[q, p]]).cos();
            }
            gradient[q] = prod;
        }

        let mut index = self.q;
        // now calculate derivatives wrt to means
        for q in 0..self.q {
            let lhs = self.weights[q] * gradient[q];
            for (p, tau) in taus.iter().enumerate() {
                let intermediate = lhs / (2.0 * PI * tau * self.means[[q, p]]).cos();
                gradient[index] =
                    intermediate * (-2.0 * tau * PI) * (2.0 * PI * tau * self.means[[q, p]]).sin();
                index += 1;
            }
        }

        // now calculate derivatives wrt to sigmas
        for q in 0..self.q {
            let lhs = self.weights[q] * gradient[q];
            for (p, tau) in taus.iter().enumerate() {
                gradient[index] = lhs * NEG_TWO_PI_SQ * tau.powi(2) * 2.0 * self.sigmas[[q, p]];
                index += 1;
            }
        }

        Dual::from((total,gradient))
    }

    fn children(&self) -> &[Box<dyn Kernel>] {
        &[]
    }

    fn describe(&self) -> String {
        format!("Spectral Mixture Kernel with {}", self.q)
    }

    fn log_prior(&self, gradient : bool) -> Dual {
        todo!()
    }
}

