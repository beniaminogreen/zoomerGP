use extendr_api::prelude::*;
use ndarray::Array1;

use crate::constraint::constraints::Constraints;
use crate::gp_regression::gp_reg::GPRegression;


use indicatif::ProgressBar;

// Implements Algorithm 2 from
// https://arxiv.org/pdf/1705.07795
#[allow(non_snake_case)]
pub struct CoinOptimizer<'a> {
    alpha: f64,
    d: usize,
    L: Array1<f64>,
    W: Array1<f64>,
    W_zero: Array1<f64>,
    G: Array1<f64>,
    R: Array1<f64>,
    theta: Array1<f64>,
    model: &'a mut GPRegression,
    constraints : Option<Constraints>
}

impl<'a> CoinOptimizer<'a> {
    pub fn new(model: &'a mut GPRegression, use_constraints : bool) -> Self {
        let d = model.get_n_params();
        let constraints : Option<Constraints> =  if use_constraints {
            Some(model.recommend_constraints())
        } else {
            None
        };

        Self {
            alpha: 100.0,
            d,
            L: Array1::zeros(d),
            G: Array1::zeros(d),
            R: Array1::zeros(d),
            W: Array1::ones(d),
            W_zero: Array1::ones(d),
            theta: Array1::zeros(d),
            model,
            constraints
        }
    }

    pub fn run(mut self, max_iter: usize) {
        let bar = ProgressBar::new(max_iter as u64);
        let mut best_w = self.W.clone();
        let mut best_log_like = f64::NEG_INFINITY;
        let mut iters_since_improvement = 0;

        for iter in 0..max_iter {
            //dbg!((iter,best_log_like));

            let mut transformed_W = self.W.to_owned();
            let mut param_transform_grads = Array1::ones(self.d);

            if let Some(constraints)  = &self.constraints {
                let constrained = constraints.constrain(self.W.as_slice().expect("non-contiguous in memory"));

                transformed_W = constrained.x;
                param_transform_grads = constrained.grad.unwrap();
            }

            self.model.set_params(transformed_W.as_slice().unwrap());
            self.model.update();
            let dual = self.model.log_like(true);
            let log_like = dual.x;
            let grad = dual.grad.unwrap() * param_transform_grads;
            
            if log_like.gt(&best_log_like) {
                best_w = self.W.clone();
                if (best_log_like - log_like).abs().gt(&0.01) {
                    iters_since_improvement = 0;
                    best_log_like = log_like;
                } else {
                    iters_since_improvement  += 1;
                }
            } else {
                iters_since_improvement += 1;
            }

            if iters_since_improvement > 50 {
                break
            }

            for i in 0..self.d {
                self.L[i] = grad[i].abs().max(self.L[i]); // line 6
                self.G[i] += grad[i].abs(); // line 7

                self.R[i] = (self.R[i] + ((self.W[i] - self.W_zero[i]) * grad[i])).max(0.0); // line 8

                self.theta[i] += grad[i]; // Line 9

                let numerator = self.theta[i] * (self.L[i] + self.R[i]);
                let denominator = self.L[i] * ((self.G[i] + self.L[i]).max(self.alpha * self.L[i]));
                //dbg!(numerator / denominator);
                self.W[i] = self.W_zero[i] + (numerator / denominator);
            }

            //dbg!(&self.R);


            bar.inc(1);
        }

        bar.finish();

        let final_params = best_w;
        self.model.set_params(final_params.as_slice().unwrap());
    }
}
