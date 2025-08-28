use crate::dual_number::Dual;
use ndarray::Array1;

use std::ops::Neg;

pub trait Likelihood : Send + Sync {
    fn log_like(&self, params : &[f64], gradient: bool) -> Dual;
    fn clone_box(&self) -> Box<dyn Likelihood>;
}


#[derive(Debug, Clone)]
pub struct BinomialLogit {
    pub y : Vec<bool>,
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}



impl Likelihood for BinomialLogit {
    fn log_like(&self, params : &[f64], gradient: bool) -> Dual {
        let log_like: f64 = self.y.iter().zip(params.iter()).map(|(&y, &x)| {
            if y {
                // log σ(x) = -log(1 + exp(-x))
                -(-x).exp().ln_1p()
            } else {
                // log(1 - σ(x)) = -log(1 + exp(x))
                -(x.exp()).ln_1p()
            }
        }).sum();

        if !gradient {
            return Dual::from(log_like);
        }

        // Gradient wrt params: dℓ/dx = y - σ(x)
        let grad: Vec<f64> = self.y.iter()
            .zip(params.iter())
            .map(|(&y, &x)| {
                let p = sigmoid(x);
                (if y { 1.0 } else { 0.0 }) - p
            })
            .collect();

        Dual::from((log_like, Array1::from(grad)))
    }

    fn clone_box(&self) -> Box<dyn Likelihood> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Likelihood> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}