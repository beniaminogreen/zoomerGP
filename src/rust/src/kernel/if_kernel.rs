use ndarray::{ArrayView1, Array1};
use super::kernel::Kernel;

use crate::dual_number::Dual;
use crate::constraint::constraint::Constraint;
use std::fmt::Debug;

#[derive(Debug)]
pub struct IfKernel {
    operating_columns: Vec<usize>,
}

impl IfKernel {
    pub fn new(operating_columns: Vec<usize>) -> Self {
        Self {
            operating_columns,
        }
    }
}

impl Kernel for IfKernel {
    fn rec_constraint(&self ) -> Vec<Constraint> {
        vec![]
    }

    fn num_params(&self) -> usize {
        0
    }

    fn set_params(&mut self, params: &[f64]) {

    }

    fn calc(&self, x: ArrayView1<f64>, y: ArrayView1<f64>, gradient: bool) -> Dual {
        let all_eq = self.operating_columns.iter().all(|i| x[*i].eq(&1.0) && y[*i].eq(&1.0));

        let x = if all_eq {
            1.0
        } else {
            0.0
        };

        if !gradient {
            return Dual::from(x)
        }
        
        Dual::from((x,Array1::from(Vec::new())))
    }

    fn children(&self) -> &[Box<dyn Kernel>] {
        &[]
    }

    fn describe(&self) -> String {
        "If Kernel".to_string()
    }
}

