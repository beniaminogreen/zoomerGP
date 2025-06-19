use crate::dual_number::Dualf64;
use ndarray::Array1;

#[derive(Clone, Debug)]
pub struct PositiveExpConstraint{}

impl PositiveExpConstraint {
    pub fn new() -> Self {
        Self{}
    }

    pub fn constrain(&self, x: f64, gradient : bool) -> Dualf64 {
        if !gradient {
            Dualf64::from(x.exp())
        } else {
            let result = x.exp();
            Dualf64::from((result, result))
        }
    }

    pub fn unconstrain(&self, x: f64, gradient : bool) -> Dualf64 {
        if !gradient {
            Dualf64::from(x.ln())
        } else {
            Dualf64::from((x.ln(), 1.0/x))
        }
    }

}

#[derive(Clone, Debug)]
pub struct PositiveSoftPlusConstraint{}

impl PositiveSoftPlusConstraint {
    pub fn new() -> Self {
        Self{}
    }

    pub fn constrain(&self, x: f64, gradient : bool) -> Dualf64 {
        // if the input is greater than f64, do not apply exp / softmax, as softmax(x) = x
        let result = if x > 20.0 {
            x
        } else {
            (1.0 + x.exp()).ln()
        };

        if !gradient {
            Dualf64::from(result)
        } else {
            if x > 20.0 {
                Dualf64::from((result, 1.0))
            } else {
                let exp_x = x.exp();
                let grad = exp_x / (1.0 + exp_x);
                Dualf64::from((result, grad))
            }
        }
    }

    pub fn unconstrain(&self, _x: f64, _gradient : bool) -> Dualf64 {
        todo!()
    }

}
