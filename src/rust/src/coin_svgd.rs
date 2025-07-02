use extendr_api::prelude::*;
use crate::gp_regression::gp_reg::GPRegression;
use crate::latent_gp_regression::latent_gp_reg::LatentGPR;

use std::ops::AddAssign;

use rand::rng;
use rand_distr::{Normal, Distribution, Uniform};

use ndarray::{Array1, Array2, ArrayView2, Axis};
use crate::model::{ClonableModel, Model, TestModel};
use crate::predict::PredictionOutput;
use std::cmp::Ordering;
use crate::constraint::constraints::Constraints;
use crate::sparse_latent_gp_regression::sparse_latent_gp_reg::SparseLatentGPR;

// implements Algorithm 6 from
//https://proceedings.mlr.press/v202/sharrock23a/sharrock23a.pdf
#[allow(non_snake_case)]
#[extendr]
struct CoinSVGP{
    d : usize,
    n : usize,
    theta: Array2<f64>,
    theta_0: Array2<f64>,
    L : Array2<f64>,
    abs_grad_sum: Array2<f64>,
    grad_sum: Array2<f64>,
    reward: Array2<f64>,
    model : Box<dyn ClonableModel>,
    constraints: Constraints
}

#[allow(non_snake_case)]
pub fn compute_kernel_matrix(X: ArrayView2<f64>) -> (Array2<f64>, Array1<f64>) {
    // first, calculate the bandwidths
    let mut bws: Vec<f64> = Vec::with_capacity(X.ncols());
    for col in X.axis_iter(Axis(1)) {
        let mut col_vec = col.to_vec();
        col_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));

        let mut distances: Vec<f64> = col_vec.windows(2).map(|x| x[1] - x[0]).collect();
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));

        let median = distances[X.nrows() / 2];
        let bw : f64 = median.powi(2)  / (X.nrows() as f64).log10();

        bws.push(median);
    }

    let n = X.nrows();

    // then populate kernel matrix
    let mut out: Array2<f64> = Array2::zeros((n, n));
    for i in 0..n {
        out[[i, i]] = 1.0;
        for j in 0..i {
            let inside_exp: f64 = X
                .row(i)
                .iter()
                .zip(X.row(j).iter())
                .map(|(a, b)| (a - b).powi(2)) // take squared differences
                .zip(bws.iter()) // then zip with the bandwidths
                .map(|(x_sq, bw)| x_sq / bw)
                .sum(); // scale squared differences and sum

            let result = (-1.0 * inside_exp).exp();
            out[[i, j]] = result;
            out[[j, i]] = result;
        }
    }

    (out, Array1::from_vec(bws))
}

/*fn kernel_and_repulsion(lambdas : ArrayView2<f64>) -> (Array2<f64>, Array2<f64>){
    let n = lambdas.nrows();
    let d = lambdas.ncols();
    // first, calculate bandwidth with median rule
    let mut bws : Vec<f64> = Vec::<f64>::with_capacity(lambdas.len());
    for col in lambdas.axis_iter(Axis(1)) {
        let mut col_vec = col.to_vec();
        col_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));
        let mut distances : Vec<f64> = col_vec.windows(2).map(|x| x[1]-x[0]).collect();
        distances.sort_by(|a,b| a.partial_cmp(b).unwrap_or(Ordering::Less));

        let median_dist = distances[distances.len()/2];

        let bw = median_dist / (n as f64).log10();
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
                 .map(|(dist,bw)| dist / bw)
                 .sum();

             let result = inside_exp.neg().exp();
             kernels[[i,j]] = result;
             kernels[[j,i]] = result;
         }
     }

     let mut repulsion_terms = Array2::zeros((n,d));
     for  i in 0..n {
         for j in 0..n {
             for k in 0..d {
                 let grad = 2.0*kernels[[i,j]]*(lambdas[[i,k]] - lambdas[[j,k]]) / bws[k];
                 repulsion_terms[[i,k]] += grad;
             }
         }
     }

     (kernels, repulsion_terms)
 }*/

 impl CoinSVGP{
     fn new(lambdas: Array2<f64>, model : Box<dyn ClonableModel>) -> Self {
         let n = lambdas.nrows();
         let d = lambdas.ncols();

         let constraints = model.recommend_constraints();

         let L = Array2::zeros((n,d));
         Self{
             theta_0: lambdas.clone(),
             theta: lambdas,
             abs_grad_sum: L.clone(),
             grad_sum: L.clone(),
             reward: L.clone(),
             L,
             n,
             d,
             model,
             constraints
         }
     }

     fn step(&mut self) {
         let (k_mat, bws) = compute_kernel_matrix(self.theta.view());
         let old_X =  self.theta.clone();

         // first, calculate the matrix of log_density_gradients
         let mut gradient_array : Array2<f64> = Array2::zeros((self.n,self.d));
         for i in 0..self.n {
             let constrained_params = self.constraints.constrain(self.theta.row(i).as_slice().unwrap());

             self.model.set_params(constrained_params.x.as_slice().unwrap());
             self.model.update();

             // apply Jacobian Correction
             let grad = &self.model.log_like(true).grad.unwrap() + constrained_params.grad.unwrap().ln();

             gradient_array.row_mut(i).assign(&grad);
         }

         for i in 0..self.n {
             let mut tally: Array1<f64> = Array1::zeros(self.d);
             for j in 0..self.n {
                 let attraction_term = k_mat[[i, j]] * &gradient_array.row(j);
                 if i == j {
                     tally = tally + attraction_term;
                 } else {
                     let repulsion_term = 2.0 * k_mat[[i, j]] * (old_X.row(i).to_owned() - old_X.row(j)) / &bws;
                     tally = tally + attraction_term + repulsion_term;
                 }
             }
             let gradient = tally / (self.n as f64);

             // now move to second for loop of algorithm
             for j in 0..self.d {
                 let abs_grad = gradient[j].abs();
                 self.L[[i, j]] = abs_grad.max(self.L[[i, j]]);
                 self.abs_grad_sum[[i, j]].add_assign(abs_grad);
                 let updated_reward = self.reward[[i, j]] + (gradient[j] * (old_X[[i, j]] - self.theta_0[[i, j]]));
                 self.reward[[i, j]] = updated_reward.max(0.0);
                 self.grad_sum[[i, j]].add_assign(gradient[j]);


                 let update_lhs = self.grad_sum[[i,j]] / (self.abs_grad_sum[[i,j]] + self.L[[i,j]]).max(100.0 * self.L[[i,j]]);
                 let update_rhs = 1.0 + (self.reward[[i,j]]/self.L[[i,j]]);
                 self.theta[[i, j]] = self.theta_0[[i,j]] + (update_lhs*update_rhs);
             }
         }

         // then compute the negative gradients
        /* for i  in 0..self.n {
             let mut tally = Array1::zeros(self.d);
             for j in 0..self.n {
                 tally = tally - gradient_array.row(j).to_owned() * k_mat[[i,j]];
                 if i != j {
                     let repulsion_term = 2.0 * k_mat[[i,j]] * (old_X.row(i).to_owned() - old_X.row(j));
                     tally = tally + repulsion_term;
                 }
             }

             let gradient = tally / self.n as f64;
             let abs_grad = gradient.abs();

             // Internal update loop:
             for j in 0..self.d {
                 self.L[[i,j]] = self.L[[i,j]].max(abs_grad[j]);
                 self.abs_grad_sum.add_assign(abs_grad[j]);
                 self.grad_sum[[i,j]].add_assign(gradient[j]);
                 let updated_reward : f64 = self.reward[[i,j]] + (gradient[j] * (old_X[[i,j]] - self.theta_0[[i,j]]));
                 self.reward[[i,j]] = updated_reward.max(0.0);


                 let update_lhs = self.grad_sum[[i,j]] / (self.abs_grad_sum[[i,j]] + self.L[[i,j]]).max(100.0 * self.L[[i,j]]);
                 let update_rhs = 1.0 + (self.reward[[i,j]]/self.L[[i,j]]);

                 self.theta[[i,j]] = self.theta_0[[i,j]] +  update_lhs * update_rhs;
             }
         }*/
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


         Self::new(lambdas.to_owned(), Box::new(model))
     }

     fn r_new_latent(x: ArrayView2<f64>, y: &[f64], kernel_specification: List, n_particles : usize) -> Self {
         let model = LatentGPR::r_new(x, y, kernel_specification);

         let d = model.get_n_params();

         let normal = Normal::new(0.0, 1.0).unwrap();
         let mut rng = rng();

         // Create a 1D array of 10 elements with normal random values
         let data: Vec<f64> = (0..d*n_particles).map(|_| normal.sample(&mut rng)).collect();
         let lambdas = Array2::from_shape_vec((n_particles, d), data).unwrap();


         Self::new(lambdas.to_owned(), Box::new(model))
     }

     fn r_new_sparse_latent(x: ArrayView2<f64>, y: &[f64], x_inducing : ArrayView2<f64>, kernel_specification: List, n_particles : usize) -> Self {
         let model = SparseLatentGPR::r_new(x, y, x_inducing, kernel_specification);

         let d = model.get_n_params();

         let normal = Normal::new(0.0, 1.0).unwrap();
         let mut rng = rng();

         // Create a 1D array of 10 elements with normal random values
         let data: Vec<f64> = (0..d*n_particles).map(|_| normal.sample(&mut rng)).collect();
         let lambdas = Array2::from_shape_vec((n_particles, d), data).unwrap();


         Self::new(lambdas.to_owned(), Box::new(model))
     }


     pub fn run(&mut self, iter : usize) -> Robj { ;
         for i in 0..iter {
             println!("iteration {i}");
             self.step();
         }

         Robj::try_from(self.theta.clone()).unwrap()
     }

     pub fn predict(&mut self,  prediction_points :ArrayView2<f64>, sub_kernel : Option<String>) -> PredictionOutput {
         let mut pred_outputs = Vec::new();
         let constraints = self.model.recommend_constraints();
         for i in 0..self.n {
             let constrained_params = constraints.constrain(self.theta.row(i).as_slice().unwrap());
             self.model.set_params(constrained_params.x.as_slice().unwrap());
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