use crate::dual_number::Dual;
use ndarray::Array1;

#[derive(Clone, Debug)]
pub struct PositiveExpConstraint{}

impl PositiveExpConstraint {
    pub fn new() -> Self {
        Self{}
    }

    pub fn constrain(&self, x: f64, gradient : bool) -> Dual {
        if !gradient {
            Dual::from(x.exp())
        } else {
            let result = x.exp();
            Dual::from((result, Array1::from(vec![result])))
        }
    }

    pub fn unconstrain(&self, x: f64, gradient : bool) -> Dual {
        if !gradient {
            Dual::from(x.ln())
        } else {
            Dual::from((x.ln(), Array1::from(vec![1.0/x])))
        }
    }

}

#[derive(Clone, Debug)]
pub struct PositiveSoftPlusConstraint{}

impl PositiveSoftPlusConstraint {
    pub fn new() -> Self {
        Self{}
    }

    pub fn constrain(&self, x: f64, gradient : bool) -> Dual {
        if !gradient {
            Dual::from((1.0 + x.exp()).ln())
        } else {
            let exp_x = x.exp();
            let result = (1.0 + exp_x).ln();
            let grad = exp_x / (1.0 + exp_x);

            Dual::from((result, Array1::from(vec![grad])))
        }
    }

    pub fn unconstrain(&self, _x: f64, _gradient : bool) -> Dual {
        todo!()
    }

}
