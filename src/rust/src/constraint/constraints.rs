use extendr_api::prelude::*;
use ndarray::Array1;
use crate::dual_number::DualArr;

use super::constraint::Constraint;

#[derive(Clone, Debug)]
#[extendr]
pub struct Constraints {
    constraints : Vec<Constraint>
}

impl Constraints {
    pub fn new(constraints: Vec<Constraint>) -> Self {
        Self {
            constraints
        }
    }
}

#[extendr]
impl Constraints{
    pub fn constrain(&self, params : &[f64]) -> DualArr {
        let mut out_params = Array1::zeros(params.len());
        let mut out_grads = Array1::zeros(params.len());

        assert!(params.len() == self.constraints.len());

        for (i, param) in params.iter().enumerate() {
            let constrained_param = self.constraints[i].constrain(*param, true);

            out_params[i] = constrained_param.x;
            out_grads[i] = constrained_param.grad.unwrap();
        }

        DualArr::from((out_params, out_grads))
    }

    pub fn len(&self) {
        dbg!(&self.constraints);
        dbg!(self.constraints.len());
    }
}


extendr_module!{
    mod constraints;
    impl Constraints;
}
