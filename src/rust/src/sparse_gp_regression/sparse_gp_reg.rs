use crate::kernel::utils::fix_conditioning;
use crate::kernel::utils::{fast_gradient_matrix, recusive_select_kernel};
use crate::kernel::kernel::Kernel;
use crate::dataset::{DataManager, UnitStandardizedDataset};
use crate::dual_number::{Dual, DualArr};
use crate::constraint::constraints::{Constraints};
use crate::constraint::constraint::{Constraint};

use std::sync::Arc;

use ndarray::{Array2, Axis, Array1, Array3, ArrayView2,s};
use ndarray_linalg::solve::{Inverse, Determinant};
use ndarray_linalg::{DeterminantC, Trace};
use extendr_api::prelude::*;

use std::f64::consts::PI;
use crate::coin_optimizer::CoinOptimizer;
use crate::kernel::utils::{parse_kernel_recursive, build_kernel_tree, stable_log_det};
use crate::predict::PredictionOutput;


#[allow(non_snake_case)]
#[extendr]
pub struct SparseGPRegression {
    pub kernel : Box<dyn Kernel>,
    dataset : Arc<dyn DataManager>,
    pub stale : bool,
    pub response: Array1<f64>,
    pub n: usize,
    pub m: usize,
    pub K_nm: Array2<f64>,
    pub K_mm: Array2<f64>,
    X_inducing : Array2<f64>,
    sigma : f64,
    n_params : usize,
    constraints : Constraints
}

#[extendr]
impl SparseGPRegression {

    #[allow(non_snake_case)]
    pub fn new(
        X: ArrayView2<f64>,
        X_inducing: ArrayView2<f64>,
        y: &[f64],
        kernel_specification: List,
    ) -> Self {
        let mut y = Array1::from(y.to_owned());
        //let y_bar = y.iter().sum::<f64>() / (y.len() as f64);
        // y -= y_bar;


        let X = X.to_owned();
        let dataset = Arc::new(UnitStandardizedDataset::new(X,y.clone()));

        let kernel = parse_kernel_recursive(kernel_specification, dataset.as_ref());

        let X_inducing = dataset.scale_predictors(X_inducing);

        let n = y.len();
        let m = X_inducing.nrows();

        let n_params = kernel.num_params() + 1;


        let kernel_constraints = kernel.rec_constraint();
        let mut constraints = vec![Constraint::new_positive_sp()];
        constraints.extend(kernel_constraints);

        Self {
            dataset,
            response : y,
            X_inducing,
            n,
            m,
            n_params,
            kernel,
            stale : true,
            K_nm: Array2::zeros((n, m)),
            K_mm: Array2::zeros((m, m)),
            sigma: 1.0,
            constraints : Constraints::new(constraints)
        }
    }

    fn get_n_params(&self) -> usize {
        self.n_params
    }

    fn set_params(&mut self, params: &[f64]) {
        // dbg!(params);
        let (first, rest) = params.split_at(1);
        self.sigma = first[0];
        self.kernel.set_params(rest);
        self.stale = true;
    }
    pub fn recommend_constraints(&self) -> Constraints {
        self.constraints.clone()
    }

    pub fn display_kernel(&self) -> String {
        build_kernel_tree(self.kernel.as_ref(), "", "A", true)
    }

    #[allow(non_snake_case)]
    fn update(&mut self) {
        let X = self.dataset.get_X();
        self.K_nm
            .axis_iter_mut(Axis(0))
            .into_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for (j, element) in row.iter_mut().enumerate() {
                    *element = self
                        .kernel
                        .calc(X.row(i).view(), self.X_inducing.row(j).view(), false).x;
                }
            });

        self.K_mm
            .axis_iter_mut(Axis(0))
            .into_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for (j, element) in row.iter_mut().enumerate() {
                    *element = self
                        .kernel
                        .calc(self.X_inducing.row(i).view(), self.X_inducing.row(j),false).x;
                }
            });

        let m = self.K_mm.nrows();
        for i in 0..m {
            self.K_mm[[i, i]] += 0.000001;
        }
        self.stale = false;
    }
    #[allow(non_snake_case)]
    fn log_like(&self, gradient : bool) -> Dual {

        let X = self.dataset.get_X();
        let G = self.K_nm.t().dot(&self.K_nm);
        let mut Z = &self.K_mm + (G.clone() / self.sigma.powi(2));


        let m = Z.nrows();
        for i in 0..m {
            Z[[i, i]] += 0.000001;
        }

        let Z_log_det = stable_log_det(Z.view());
        let K_mm_log_det = stable_log_det(self.K_mm.view());
        let Z_inv = Z.inv();
        let K_mm_inv = self.K_mm.inv();

        if Z_inv.is_err() || K_mm_inv.is_err() || Z_log_det.is_err() || K_mm_log_det.is_err() {
            return Dual::from((-99999999.9999, Array1::zeros(self.n_params)));
        }

        let Z_inv = Z_inv.unwrap();
        let K_mm_inv = K_mm_inv.unwrap();
        let Z_log_det = Z_log_det.unwrap();
        let K_mm_log_det = K_mm_log_det.unwrap();

        // Calculate the Approximate Determinant Based on the PP approximation
        let mut approx_log_det = 2.0 * (self.n as f64) * (self.sigma.ln());
        approx_log_det -= K_mm_log_det;
        approx_log_det += Z_log_det;

        // Calculate the data-fit term using the matrix inversion lemma
        let lhs = self.response.dot(&self.response) / self.sigma.powi(2);
        let y_dot_knm = self.response.dot(&self.K_nm);

        let rhs = y_dot_knm.dot(&Z_inv).dot(&y_dot_knm.t()) / self.sigma.powi(4);

        let data_fit_term = lhs - rhs;

        let normalizer = (self.n as f64) * ((2.0 * PI).ln());

        let log_like = -0.5 * (data_fit_term + approx_log_det + normalizer);

        if !gradient {
            return Dual::from((log_like))
        }

        let mut gradient: Array1<f64> = Array1::zeros(self.n_params);

        let yt_kinv_lhs: Array1<f64> = (&self.response / self.sigma.powi(2)).t().to_owned();
        let yt_kinv_rhs = (&self.response / self.sigma.powi(4))
            .t()
            .dot(&self.K_nm)
            .dot(&Z_inv)
            .dot(&self.K_nm.t());
        let yt_k_inv = yt_kinv_lhs - yt_kinv_rhs;

        let sigma_grad_lhs = self.sigma * yt_k_inv.dot(&yt_k_inv.t());
        let sigma_grad_rhs = (2.0 / self.sigma) * (self.n as f64)
            - (2.0 / self.sigma.powi(3))
            * (self.K_nm.t().dot(&self.K_nm).dot(&Z_inv)).trace().unwrap();

        gradient[0] = sigma_grad_lhs - (sigma_grad_rhs / 2.0);

        // Now we have cauculated the derivative with respect to sigma. it's time to do the other partial derivatives

        // Now Create Matrixes of Partial Derivatives

        let K_mm_derivatives = fast_gradient_matrix(
            self.X_inducing.view(),
            self.X_inducing.view(),
            &self.kernel,
        );

        let K_nm_derivatives =
            fast_gradient_matrix(X.view(), self.X_inducing.view(), &self.kernel);

        let k_nm_yt_k_inv = yt_k_inv.dot(&self.K_nm);
        for i in 0..(self.n_params - 1) {
            let component_a: f64 = yt_k_inv
                .dot(&K_nm_derivatives.index_axis(Axis(2), i))
                .dot(&K_mm_inv)
                .dot(&k_nm_yt_k_inv.t());

            let component_b: f64 = k_nm_yt_k_inv
                .dot(&K_mm_inv)
                .dot(&K_mm_derivatives.index_axis(Axis(2), i))
                .dot(&K_mm_inv)
                .dot(&k_nm_yt_k_inv.t());

            // let component_c: f64 = yt_k_inv
            //     .dot(&self.K_nm)
            //     .dot(&K_mm_inv)
            //     .dot(&K_nm_derivatives.index_axis(Axis(2), i).t())
            //     .dot(&yt_k_inv.t()); // component c = component a

            let data_fit_component = (component_a + component_b + component_a) / 2.0;

            let H = self.K_nm.t().dot(&K_nm_derivatives.index_axis(Axis(2), i));

            let H_k_mm_inv = H.dot(&K_mm_inv);
            let quadratic_term = K_mm_inv
                .dot(&K_mm_derivatives.index_axis(Axis(2), i))
                .dot(&K_mm_inv);

            let trace_component_a_lhs = (H_k_mm_inv).trace().unwrap() / self.sigma.powi(2);
            let trace_component_a_rhs =
                (G.dot(&Z_inv).dot(&H_k_mm_inv)).trace().unwrap() / self.sigma.powi(4);

            let trace_component_b_lhs =
                (G.dot(&quadratic_term)).trace().unwrap() / self.sigma.powi(2);
            let trace_component_b_rhs = (G.dot(&Z_inv).dot(&G).dot(&quadratic_term))
                .trace()
                .unwrap()
                / self.sigma.powi(4);

            let trace_component = (trace_component_a_lhs - trace_component_a_rhs)
                + (0.5 * (trace_component_b_lhs - trace_component_b_rhs));

            // dbg!(trace_component/3.0);

            // For some reasons, gradients are 3x as large as they should be.
            gradient[[i + 1]] = (data_fit_component - trace_component) / 3.0;
        }

        Dual::from((log_like, gradient))
    }

    #[allow(non_snake_case)]
    pub fn predict(&self, prediction_points: ArrayView2<f64>,  sub_kernel : Option<String>) -> PredictionOutput {


        let X = self.dataset.get_X();
        let mut prediction_points = prediction_points.to_owned();
        prediction_points = self.dataset.scale_predictors(prediction_points.view());

        let kernel = match sub_kernel {
            Some(search_str) => {recusive_select_kernel(self.kernel.as_ref(), search_str.as_str())}
            None => {self.kernel.as_ref()}
        };

        let mut K_star = Array2::zeros((prediction_points.nrows(), self.m));

        K_star
            .axis_iter_mut(Axis(0))
            .into_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for (j, element) in row.iter_mut().enumerate() {
                    *element = kernel.calc(
                        prediction_points.row(i).view(),
                        self.X_inducing.row(j).view(),
                        false
                    ).x;
                }
            });

        let G = self.K_nm.t().dot(&self.K_nm);

        let intermediate_matrix = self.sigma.powi(2) * &self.K_mm + G;
        let int_inv = intermediate_matrix.inv().unwrap();

        let k_mn_y = self.K_nm.t().dot(&self.response);

        let output: Array1<f64> = K_star.dot(&int_inv).dot(&k_mn_y);

        let k_mm_inv = self.K_mm.inv().unwrap();

        let var: Vec<f64> = prediction_points
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, point)| {
                let kstar = K_star.row(i);
                let term_1 = kernel.calc(point.view(), point.view(),false).x;
                let term_2 = kstar.t().dot(&k_mm_inv).dot(&kstar);
                let term_3 = self.sigma.powi(2) * kstar.t().dot(&int_inv).dot(&kstar);

                term_1 - term_2 + term_3
            })
            .collect();

        let var: Array1<f64> = Array1::from_vec(var);

        PredictionOutput::from((
            output,
            Some(var.clone()),
            Some(var + self.sigma.powi(2))
        ))
    }
}



extendr_module! {
    mod sparse_gp_reg;
    impl SparseGPRegression;
}
