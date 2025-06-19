use std::convert::From;
use std::ops::{Add, Mul};
use std::iter::Sum;
use ndarray::Array1;


use extendr_api::prelude::*;

#[extendr]
pub struct Dualf64 {
    pub x : f64,
    pub grad : Option<f64>
}
#[extendr]
impl Dualf64 {
    fn get_x(&self) -> f64 {
        self.x
    }
    fn get_grad(&self) -> Option<f64> {
        self.grad
    }
}


impl Add for Dualf64 {
    type Output = Dualf64;

    fn add(self, other: Dualf64) -> Dualf64 {
        let grad = if self.grad.is_none() || other.grad.is_none() {
            None
        } else {
            Some(self.grad.unwrap() + other.grad.unwrap())
        };

        Self {
            x: self.x + other.x,
            grad
        }
    }
}

impl Mul for Dualf64 {
    type Output = Dualf64;
    fn mul(self, other: Dualf64) -> Dualf64 {
        let grad = if self.grad.is_none() || other.grad.is_none() {
            None
        } else {
            Some(other.x * self.grad.unwrap() + self.x * other.grad.unwrap())
        };

        Self {
            x: self.x * other.x,
            grad
        }
    }
}

impl From<f64> for Dualf64 {
    fn from(x: f64) -> Self {
        Self{
            x,
            grad : None
        }
    }
}

impl From<(f64, f64)> for Dualf64 {
    fn from(x_and_grad: (f64, f64)) -> Self {
        Self{
            x : x_and_grad.0,
            grad : Some(x_and_grad.1)
        }
    }
}


#[extendr]
#[derive(Debug)]
pub struct Dual{
    pub x : f64,
    pub grad : Option<Array1<f64>>
}

#[extendr]
impl Dual {
    fn get_x(&self) -> f64 {
        self.x
    }
    fn get_grad(&self) -> Option<Doubles> {
        if let Some(grad) = self.grad.clone() {
            Some(grad.into_iter().collect())
        } else {
            None
        }
    }
}

impl Add for Dual {
    type Output = Dual;

    fn add(self, other: Dual) -> Dual {
        let grad = if self.grad.is_none() || other.grad.is_none() {
            None
        } else {
            Some(self.grad.unwrap() + other.grad.unwrap())
        };

        Self {
            x: self.x + other.x,
            grad
        }
    }
}

impl Mul for Dual {
    type Output = Dual;
    fn mul(self, other: Dual) -> Dual {
        let grad = if self.grad.is_none() || other.grad.is_none() {
            None
        } else {
            Some(other.x * self.grad.unwrap() + self.x * other.grad.unwrap())
        };

        Self {
            x: self.x * other.x,
            grad
        }
    }
}

impl From<f64> for Dual {
    fn from(x: f64) -> Self {
        Self{
            x,
            grad : None
            }
    }
}

impl From<(f64, Array1<f64>)> for Dual {
    fn from(x_and_grad: (f64, Array1<f64>)) -> Self {
        Self{
            x : x_and_grad.0,
            grad : Some(x_and_grad.1)
            }
    }
}

impl Sum for Dual {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let first = iter.next().expect("Iterator must not be empty");
        iter.fold(first, |acc, p| acc + p)
    }
}

#[extendr]
pub struct DualArr{
    pub x : Array1<f64>,
    pub grad : Option<Array1<f64>>
}

#[extendr]
impl DualArr {
    fn get_x(&self) -> Vec<f64> {
        self.x.to_vec()
    }
    fn get_grad(&self) -> Option<Doubles> {
        if let Some(grad) = self.grad.clone() {
            Some(grad.into_iter().collect())
        } else {
            None
        }
    }
}

impl From<Array1<f64>> for DualArr {
    fn from(x: Array1<f64>) -> Self {
        Self{
            x,
            grad : None
            }
    }
}

impl From<(Array1<f64>,Array1<f64>)> for DualArr {
    fn from(x : (Array1<f64>, Array1<f64>)) -> Self {
        Self{
            x : x.0,
            grad : Some(x.1)
            }
    }
}

extendr_module!{
    mod dual_number;
    impl Dualf64;
    impl Dual;
    impl DualArr;
}