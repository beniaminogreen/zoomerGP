use super::kernel::Kernel;
use ndarray::{ArrayView1, Array1};
use crate::constraint::constraint::Constraint;

use crate::dual_number::Dual;

#[derive(Debug)]
pub struct CompositeKernel {
    kernels: Vec<Box<dyn Kernel>>,
    num_params : usize
}

impl CompositeKernel {
    pub fn new(kernels: Vec<Box<dyn Kernel>>) -> Self {
        let mut num_params = 0;
        for kernel in kernels.iter() {
            num_params += kernel.num_params();
        }

        Self { kernels , num_params}
    }
}

impl Kernel for CompositeKernel {
    fn rec_constraint(&self) -> Vec<Constraint> {
        let mut out = Vec::new();
        for kernel in self.kernels.iter() {
            out.extend(kernel.rec_constraint());
        };

        out
    }

    fn num_params(&self) -> usize {
        self.num_params
    }

    fn set_params(&mut self, params: &[f64]) {
        let mut remainder = params;
        for kernel in &mut self.kernels {
            let (params_for_this, rhs) = remainder.split_at(kernel.num_params());
            remainder = rhs;
            kernel.set_params(params_for_this);
        }
    }

    fn calc(&self, x: ArrayView1<f64>, y: ArrayView1<f64>, gradient : bool) -> Dual {
        if !gradient {
            self.kernels
                .iter()
                .map(|kernel| kernel.calc(x, y, gradient))
                .sum()
        }
        else {
            let mut gradients : Array1<f64> = Array1::zeros(self.num_params());
            let mut i  = 0;
            let mut result = 0.0;
            for kernel in self.kernels.iter() {
                let output = kernel.calc(x,y, true);
                result += output.x;
                for partial in output.grad.unwrap() {
                    gradients[i] = partial;
                    i += 1;
                }
            }
            return Dual::from((result, gradients))
        }
    }

    fn children(&self) -> &[Box<dyn Kernel>] {
        &self.kernels
    }

    fn describe(&self) -> String {
        "Plus (+)".to_string()
    }
}

