use extendr_api::prelude::*;
use crate::gp_regression::gp_reg::GPRegression;

use std::ops::{AddAssign, Neg};

use rand::rng;
use rand_distr::{Normal, Distribution};

use ndarray::{Array1, Array2, ArrayView2, Axis};
use crate::model::Model;
use crate::predict::PredictionOutput;

// implements Algorithm 6 from
//https://proceedings.mlr.press/v202/sharrock23a/sharrock23a.pdf
#[allow(non_snake_case)]
#[extendr]
struct CoinSVGP{
    d : usize,
    n : usize,
    X : Array2<f64>,
    X_0 : Array2<f64>,
    L : Array2<f64>,
    G : Array2<f64>,
    H : Array2<f64>,
    R : Array2<f64>,
    model : Box<dyn Model>,
}

fn kernel_and_repulsion(lambdas : ArrayView2<f64>) -> (Array2<f64>, Array2<f64>){
    let n = lambdas.nrows();
    let d = lambdas.ncols();

    // first, calculate bandwidth with median rule
    let mut bws : Vec<f64> = Vec::<f64>::with_capacity(lambdas.len());
    for col in lambdas.axis_iter(Axis(1)) {
        let mut col_vec = col.to_vec();
        col_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut squared_distances : Vec<f64> = col_vec.windows(2).map(|x| (x[1]-x[0]).powi(2)).collect();

        squared_distances.sort_by(|a,b| a.partial_cmp(b).unwrap());

        let median_dist = squared_distances[(lambdas.nrows()-1) /2];

        let bw = (0.5*median_dist / (lambdas.ncols() as f64 + 1.0).ln()).sqrt();

        bws.push(bw);
    }

    let mut kernels = Array2::zeros((n,n));
    for i in 0..n {
        kernels[[i,i]] = 1.0;
        for j in 0..i {
            let inside_exp : f64 = lambdas.row(i).iter()
                .zip(lambdas.row(j).iter())
                .map(|(a,b)| (a-b).powi(2))
                .zip(bws.iter())
                .map(|(dist,bw)| dist / bw.powi(2))
                .sum();

            let result = inside_exp.neg().exp();
            kernels[[i,j]] = result;
            kernels[[j,i]] = result;
        }
    }

    let mut repulsion_terms = Array2::zeros((n,d));
    for  i in 0..n {
        for j in 0..i {
            for k in 0..d {
                let grad = kernels[[i,j]]*(lambdas[[i,k]] - lambdas[[j,k]]) / bws[k].powi(2);
                repulsion_terms[[i,k]] += grad;
                repulsion_terms[[j,k]] -= grad;
            }
        }
    }

    (kernels, repulsion_terms)
}

impl CoinSVGP{
    fn new(lambdas: Array2<f64>, model : GPRegression) -> Self {
        let n = lambdas.nrows();
        let d = lambdas.ncols();

        let L = Array2::zeros((n,d));
        Self{
            X_0 : lambdas.clone(),
            X : lambdas,
            G : L.clone(),
            H : L.clone(),
            R : L.clone(),
            L,
            n,
            d,
            model : Box::new(model)
        }
    }

    fn step(&mut self) {
        let (k_mat, repulsion_terms) = kernel_and_repulsion(self.X.view());

        let old_X =  self.X.clone();

        // first, calculate the matrix of log_density_gradients
        let mut grads : Array2<f64> = Array2::zeros((self.n,self.d));

        for i in 0..self.n {
            let constrained_params = self.model.recommend_constraints().constrain(self.X.row(i).as_slice().unwrap());

            self.model.set_params(constrained_params.x.as_slice().unwrap());
            self.model.update();

            // apply Jacobian Correction
            let grad = &self.model.log_like(true).grad.unwrap() + constrained_params.grad.unwrap().ln();

            grads.row_mut(i).assign(&grad);
        }

        // then compute the negative gradients
        for i  in 0..self.n {
            let mut tally = Array1::zeros(self.d);
            for j in 0..self.n  {
                let attraction_term = grads.row(i).to_owned() * k_mat[[i,j]];
                tally.add_assign(&attraction_term);
            }
            tally.add_assign(&repulsion_terms.row(i));

            let gradient = tally / self.n as f64;

            // Internal update loop:

            for j in 0..self.d {
                let abs_grad = gradient[j].abs();
                self.L[[i,j]] = abs_grad.max(self.L[[i,j]]);
                self.G.add_assign(abs_grad);

                let updated_reward : f64 = self.R[[i,j]] + (gradient[j] * (old_X[[i,j]] - self.X_0[[i,j]]));
                self.R[[i,j]] = updated_reward.max(0.0);

                let update_lhs = self.H[[i,j]] / (self.G[[i,j]] + self.L[[i,j]]).max(100.0 * self.L[[i,j]]);
                let update_rhs = 1.0 + self.R[[i,j]]/self.L[[i,j]];

                self.X[[i,j]] = self.X_0[[i,j]] +  update_lhs * update_rhs;
                self.H[[i,j]].add_assign(gradient[j]);
            }
        }
    }
}

#[extendr]
impl CoinSVGP {
    pub fn r_new(x: ArrayView2<f64>, y: &[f64], kernel_specification: List, noise: bool, n_particles : usize) -> Self {
        let model = GPRegression::r_new(x, y, kernel_specification, noise);

        let d = model.get_n_params();

        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rng();

        // Create a 1D array of 10 elements with normal random values
        let data: Vec<f64> = (0..d*n_particles).map(|_| normal.sample(&mut rng)).collect();
        let lambdas = Array2::from_shape_vec((n_particles, d), data).unwrap();


        Self::new(lambdas.to_owned(), model)
    }

    pub fn run(&mut self, iter : usize) { ;
        for i in 0..iter {
            println!("iteration {i}");
            self.step();
        }
    }

    pub fn predict(&mut self,  prediction_points :ArrayView2<f64>, sub_kernel : Option<String>) -> PredictionOutput {
        let mut pred_outputs = Vec::new();
        for i in 0..self.n {
            self.model.set_params(self.X.row(i).as_slice().unwrap());
            self.model.update();
            pred_outputs.push(self.model.predict(prediction_points, sub_kernel.clone()));
        };

        let mut pred_means = pred_outputs[0].response.clone();
        for i in 1..self.n{
            pred_means.add_assign(&pred_outputs[i].response);
        }

        PredictionOutput::from((pred_means/self.n as f64, None, None))
    }
}


extendr_module! {
    mod coin_svgd;
    impl CoinSVGP;
}