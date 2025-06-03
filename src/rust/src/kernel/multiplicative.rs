use super::kernel::Kernel;
use ndarray::{ArrayView1, Array1, stack, Axis, concatenate};
use crate::dual_number::Dual;

use crate::constraint::constraint::Constraint;

#[derive(Debug)]
pub struct MultiplicativeKernel {
    children: [Box<dyn Kernel>; 2],
    n_params: usize
}

impl MultiplicativeKernel {
    pub fn new(left_kernel: Box<dyn Kernel>, right_kernel: Box<dyn Kernel>) -> Self {
        let left_kernel_params = left_kernel.num_params();
        let right_kernel_params = right_kernel.num_params();

        Self {
            n_params: left_kernel_params + right_kernel_params,
            children : [left_kernel, right_kernel]
        }
    }
}

impl Kernel for MultiplicativeKernel {
    fn rec_constraint(&self) -> Vec<Constraint> {
        let mut left_con = self.children[0].rec_constraint();
        let right_con = self.children[1].rec_constraint();

        left_con.extend(right_con);

        left_con
    }

    fn num_params(&self) -> usize {
        self.n_params
    }

    fn set_params(&mut self, params: &[f64]) {
        let (lhs_params, rhs_params) = params.split_at(self.children[0].num_params());
        self.children[0].set_params(lhs_params);
        self.children[1].set_params(rhs_params);
    }

    // Gradient collection here is correct
    fn calc(&self, x: ArrayView1<f64>, y: ArrayView1<f64>, gradient : bool) -> Dual {
        let lhs = self.children[0].calc(x, y, gradient);
        let rhs = self.children[1].calc(x, y, gradient);

        let result = lhs.x*rhs.x;

        if !gradient {
            return Dual::from(result)
        }

        let lhs_grad = rhs.x * lhs.grad.unwrap();
        let rhs_grad = lhs.x * rhs.grad.unwrap();

        let grad = concatenate![Axis(0), lhs_grad , rhs_grad];


        Dual::from((result, grad))

    }

    fn children(&self) -> &[Box<dyn Kernel>] {
        &self.children
    }

    fn describe(&self) -> String {
        "Times (*)".to_string()
    }
}

