use crate::kernel::kernel::Kernel;
use crate::dataset::DataManager;
use crate::dual_number::Dual;
use crate::constraint::constraint::Constraint;

use std::sync::Arc;

use ndarray::{Array2, Axis, Array1, Array3};
use ndarray_linalg::solve::{Inverse, Determinant};

use extendr_api::prelude::*;

use std::f64::consts::PI;

#[allow(non_snake_case)]
#[extendr]
pub struct GPRegression{
    pub kernel : Box<dyn Kernel>,
    dataset : Arc<dyn DataManager>,
    response : Array1<f64>,
    sigma : f64,
    n_params : usize,
    noise : bool,
    stale : bool,
    K: Array2<f64>,
    K_inv: Array2<f64>,
}

#[allow(non_snake_case)]
impl GPRegression {
    pub fn new(kernel : Box<dyn Kernel>, dataset : Arc<dyn DataManager>, response : Array1<f64>, noise : bool) -> Self {
        let n = dataset.shape().0;

        let n_params = if noise {
            kernel.num_params() + 1
        } else {
            kernel.num_params()
        };

        Self{
            dataset,
            stale : true,
            response,
            noise : noise,
            n_params,
            K_inv : Array2::zeros((n,n)),
            K : Array2::zeros((n,n)),
            sigma : 1.0,
            kernel
        }
    }
}

impl GPRegression {
    pub fn get_n_params(&self) -> usize {
        return self.n_params
    }

    pub fn reccomend_constraints(&self) -> Vec<Constraint> {
        self.kernel.rec_constraint()
    }

    pub fn set_params(&mut self, params: &[f64]) {
        if self.noise {
            let (first, rest) = params.split_at(1);
            self.sigma = first[0];
            self.kernel.set_params(rest);
        } else {
            self.kernel.set_params(params);
        }
        self.stale = true;
    }

    #[allow(non_snake_case)]
    pub fn update(&mut self) {
        if !self.stale {
            return
        }

        dbg!(&self.kernel);


        let X = self.dataset.get_X();

        self.K
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(i, mut row)| {
                for (j, element) in row.iter_mut().enumerate() {
                    *element = self
                        .kernel
                        .calc(X.row(i).view(), X.row(j).view(), false).x;
                }
            });

        let sigma_squared = self.sigma.powi(2);
        for i in 0..self.K.nrows() {
            self.K[[i, i]] += sigma_squared;
        }

        let K_inv = self.K.inv();
        self.K_inv = K_inv.unwrap();

        self.stale = false;

    }

    #[allow(non_snake_case)]
    pub fn log_like(&self, gradient : bool)  -> Dual {

        if self.stale {
            panic!("Tried to get log likelihood on non-updated model!");
        }

        let y = self.response.view();
        let X = self.dataset.get_X();
        let n = self.dataset.shape().0;

        let lhs = -0.5 * (y.t()).dot(&self.K_inv).dot(&y);

         //-0.5 log |K|
        let (_, log_det) = self
            .K
            .sln_det()
            .expect("Could Not Find Log Determinant of K");
        let center_term = -0.5 * log_det;

        // -0.5 * n log(2pi)
        let normalizer = -0.5 * (n as f64) * ((2.0 * PI).ln());

        let log_likelihood = lhs + center_term + normalizer;

        assert!(log_likelihood.is_finite());

        if !gradient{
            return Dual::from(log_likelihood)
        };

        let d = self.n_params;
        let mut matrix_differentials: Array3<f64> = Array3::zeros((n, n, d));

        let mut offset  = 0;
        // if we have gaussian noie, add the derivatives with respect to sigma
        if self.noise {
            for i in 0..n {
                matrix_differentials[[i, i, 0]] = 2.0 * self.sigma;
            }
            offset +=1
        }

        // then loop over the other parameters. If we did not have the noise, then there will be
        // fewer params to loop over
        for i in 0..n {
            for j in 0..(i + 1) {
                let gradients = self
                    .kernel
                    .calc(X.row(i).view(), X.row(j).view(), true).grad.unwrap();
                for var_index in 0..(d - offset) {
                    matrix_differentials[[i, j, var_index + offset]] = gradients[var_index];
                    matrix_differentials[[j, i, var_index + offset]] = gradients[var_index];
                }
            }
        }

        let yt_k_inv = self.K_inv.dot(&y);

        let mut gradient: Array1<f64> = Array1::zeros(d);
        // See Rassmusen and Williams p. 114, equation 5.9
        // lhs : left-hand side
        // rhs : right-hand side
        for i in 0..d {
            let k_grad = matrix_differentials.index_axis(Axis(2), i);

            let lhs = yt_k_inv.t().dot(&k_grad).dot(&yt_k_inv);

            let mut trace = 0.0;
            for j in 0..n {
                trace += self
                    .K_inv
                    .index_axis(Axis(0), j)
                    .dot(&k_grad.index_axis(Axis(0), j));
            }


            gradient[[i]] = 0.5 * (lhs - trace);
        }

        Dual::from((log_likelihood,gradient))

    }
}


