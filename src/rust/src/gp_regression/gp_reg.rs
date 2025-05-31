use crate::predict::PredictionOutput;
use crate::kernel::kernel::Kernel;
use crate::dataset::{DataManager, UnitStandardizedDataset};
use crate::dual_number::{Dual, DualArr};
use crate::constraint::constraints::{Constraints};

use std::sync::Arc;

use ndarray::{Array2, Axis, Array1, Array3, ArrayView2};
use ndarray_linalg::solve::{Inverse, Determinant};

use extendr_api::prelude::*;

use std::f64::consts::PI;
use crate::coin_optimizer::CoinOptimizer;
use crate::kernel::utils::{build_kernel_tree, parse_kernel_recursive, recusive_select_kernel};

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
    constraints : Constraints
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
            noise,
            n_params,
            K_inv : Array2::zeros((n,n)),
            K : Array2::zeros((n,n)),
            sigma : 1.0,
            constraints : Constraints::new(kernel.rec_constraint()),
            kernel, 
        }
    }
}

#[extendr]
impl GPRegression {
    pub fn get_n_params(&self) -> usize {
       self.n_params
    }

    pub fn recommend_constraints(&self) -> Constraints {
        self.constraints.clone()
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
    pub fn predict(&self,  prediction_points :ArrayView2<f64>, sub_kernel : Option<String>) -> PredictionOutput {
        let mut prediction_points = prediction_points.to_owned();
        prediction_points = self.dataset.scale_predictors(prediction_points.view());

        let X = self.dataset.get_X();

        let mut K_star = Array2::zeros((prediction_points.nrows(), self.dataset.shape().0));

        let kernel = match sub_kernel {
            Some(search_str) => {recusive_select_kernel(self.kernel.as_ref(), search_str.as_str())}
            None => {self.kernel.as_ref()}
        };

        K_star.axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(i, mut row)| {
                for (j, element) in row.iter_mut().enumerate() {
                    *element = kernel.calc(prediction_points.row(i).view(), X.row(j), false).x;
                }
            });

        let preds = K_star.dot(&self.K_inv).dot(&self.response);

        let var : Vec<f64> = prediction_points
            .axis_iter(Axis(0))
            //.into_par_iter()
            .enumerate()
            .map(|(i, x)| {
                let x = x.view();
                let rhs = self.kernel.calc(x, x, false).x;
                let lhs = K_star.row(i).t().dot(&self.K_inv).dot(&K_star.row(i));
                rhs - lhs
            })
            .collect();

        let f_var = Array1::from(var);
        let pred_var = f_var.clone() + self.sigma.powi(2);

        PredictionOutput::from((preds, Some(f_var), Some(pred_var)))
    }


    #[allow(non_snake_case)]
    pub fn log_like(&self, gradient : bool)  -> Dual {

        if self.stale {
            panic!("Tried to get log likelihood on non-updated model!");
        }

        let y = self.response.view();
        let X = self.dataset.get_X();
        let n = self.dataset.shape().0;

        let lhs = -0.5 * y.t().dot(&self.K_inv).dot(&y);

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
        // if we have gaussian noise, add the derivatives with respect to sigma
        if self.noise {
            for i in 0..n {
                matrix_differentials[[i, i, 0]] = 2.0 * self.sigma;
            }
            offset +=1
        }

        // Then loop over the other parameters. If we did not have the noise, then there will be
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
        // lhs: left-hand side
        // rhs: right-hand side
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

    pub fn display_kernel(&self) -> String {
        build_kernel_tree(self.kernel.as_ref(), "", "A", true)
    }

    pub fn optimize(&mut self, max_iter : u32, use_constraints : bool) {
        let optimizer = CoinOptimizer::new(self, use_constraints);
        optimizer.run(max_iter as usize);
    }



    pub fn r_new(x: ArrayView2<f64>, y: &[f64], kernel_specification: List, noiseless: bool) -> Self {
        let y = Array1::from(y.to_owned());
        let x = x.to_owned();

        let kernel = parse_kernel_recursive(kernel_specification);
        let dataset = Arc::new(UnitStandardizedDataset::new(x,y.clone()));

        Self::new(kernel, dataset, y, noiseless)
    }

}


extendr_module! {
    mod gp_reg;
    impl GPRegression;
}