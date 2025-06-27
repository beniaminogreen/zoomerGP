use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use crate::constraint::constraints::Constraints;
use crate::dual_number::Dual;
use crate::latent_gp_regression::latent_gp_reg::LatentGPR;
use crate::lyapunov::{lyap_newton_shulz_backward, lyap_newton_shulz_fwd};
use crate::model::Model;
use crate::predict::PredictionOutput;

use std::ops::Neg;
use extendr_api::{extendr, extendr_module};
use ndarray_linalg::Inverse;
use rand_distr::num_traits::real::Real;
use crate::kernel::utils::recusive_select_kernel;

const NUM_FORWARD_ITER : usize = 5;
const NUM_BACKWARD_ITER : usize = 5;

const NUGGET : f64 = 0.001;

fn outer_product(a: ArrayView1<f64>, b: ArrayView1<f64>) -> Array2<f64> {
    // Reshape a to (m, 1) and b to (1, n), then multiply (broadcasting)
    a.to_owned().insert_axis(Axis(1)) * b.to_owned().insert_axis(Axis(0))
}

#[extendr]
impl Model for LatentGPR{
    fn predict(&self, prediction_points: ArrayView2<f64>, sub_kernel: Option<String>) -> PredictionOutput {
        let u : Array1<f64> = self.L.dot(&self.v);

        dbg!(&self.v);
        dbg!(&u);

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


        let k_inv = self.K.inv().unwrap();
        let preds = K_star.dot(&k_inv).dot(&u);

        PredictionOutput::from((preds, None, None))
    }

    fn recommend_constraints(&self) -> Constraints {
        self.constraints.clone()
    }

    fn log_like(&self, gradient: bool) -> Dual {

        // first, do prior

        let log_prior = self.kernel.log_prior(gradient);

        // first calculate log-likelihood component from v
        let v_component : f64  = self.v.iter().map(|x| x.powi(2)).sum::<f64>().neg();

        // then calculate u
        let u : Array1<f64> = self.L.dot(&self.v);

        let likelihood_dual = self.likelihood.log_like(u.as_slice().unwrap(), gradient);

        let log_like = v_component + likelihood_dual.x + &log_prior.x;

        if !gradient {
            return Dual::from(log_like);
        }

        // if we do need to calculate the gradient, we need to calculate partials wrt theta
        let dl_dL : Array2<f64> = outer_product(likelihood_dual.grad.as_ref().unwrap().view(), self.v.view());

        let dl_dK = lyap_newton_shulz_backward(self.L.view(), dl_dL.view(), NUM_BACKWARD_ITER);

        let X = self.dataset.get_X();
        let n = X.nrows();
        let d = self.kernel.num_params();
        let mut matrix_differentials: Array3<f64> = Array3::zeros((n, n, d));

        for i in 0..n {
            for j in 0..(i + 1) {
                let gradients = self
                    .kernel
                    .calc(X.row(i).view(), X.row(j).view(), true).grad.unwrap();
                for var_index in 0..d {
                    matrix_differentials[[i, j, var_index]] = gradients[var_index];
                    matrix_differentials[[j, i, var_index]] = gradients[var_index];
                }
            }
        }


        let mut theta_grads = Vec::with_capacity(d);
        let log_prior_grad = log_prior.grad.unwrap();

        for i in 0..d {
            let dK_dtheta : ArrayView2<f64> = matrix_differentials.index_axis(Axis(2),i);

            let result : f64 =  dl_dK.iter().zip(dK_dtheta.iter()).map(|(a,b)| a*b).sum();


            theta_grads.push(result + log_prior_grad[i]);
        }


        // now calculate grads wrt v
        let v_grads = likelihood_dual.grad.unwrap().dot(&self.L) -2.0 * &self.v;
        theta_grads.extend(v_grads.into_iter());

        Dual::from((log_like, Array1::from(theta_grads)))
    }


    fn get_n_params(&self) -> usize { self.n_params }
    fn update(&mut self) {
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
        for i in 0..self.K.nrows() {
            self.K[[i,i]] += NUGGET;
        }

        let (L , _) = lyap_newton_shulz_fwd(self.K.view(), NUM_FORWARD_ITER);
        self.L = L;

        self.stale = false;
    }

    fn set_params(&mut self, params : &[f64]) {
        let (kernel_params, latent_params) = params.split_at(self.kernel.num_params());
        self.kernel.set_params(&kernel_params);
        self.v = Array1::from(latent_params.to_vec());
        self.stale = true;
    }
}

extendr_module! {
    mod latent_model;
    impl LatentGPR;
}