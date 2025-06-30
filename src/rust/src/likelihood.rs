use crate::dual_number::Dual;

use std::ops::Neg;

pub trait Likelihood {
    fn log_like(&self, params : &[f64], gradient: bool) -> Dual;
}

pub struct BinomialLogit {
    pub y : Vec<bool>,
}

impl Likelihood for BinomialLogit {
    fn log_like(&self, params : &[f64], gradient: bool) -> Dual {
        let probabilities : Vec<f64> = params.iter().map(|x| 1.0 / (1.0 + x.neg().exp())).collect();

        let log_like = self.y.iter().zip(probabilities.iter()).map(|(y, p)| if *y {p.ln() } else { (1.0-p).ln()}).sum();

        if ! gradient {
            return Dual::from(log_like)
        }

        let grad = probabilities.into_iter()
            .zip(self.y.iter())
            .map(|(p, y)| if *y {1.0-p} else {-p} )
            .collect();

        Dual::from((log_like, grad))

    }
}