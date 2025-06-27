use crate::dual_number::Dualf64;

use super::positive::{PositiveExpConstraint, PositiveSoftPlusConstraint};

use extendr_api::prelude::*;

#[derive(Clone, Debug)]
#[extendr]
pub enum Constraint  {
    PositiveExp(PositiveExpConstraint),
    PositiveSoftPlus(PositiveSoftPlusConstraint),
    NoConstraint(NoConstraint)
}
#[derive(Clone, Debug)]
struct NoConstraint {
}

impl NoConstraint {
    pub fn constrain(&self, x: f64, gradient : bool) -> Dualf64 {
        if !gradient {
            Dualf64::from(x)
        } else {
            Dualf64::from((x, 1.0))
        }
    }

    pub fn unconstrain(&self, x: f64, gradient : bool) -> Dualf64 {
        self.constrain(x, gradient)
    }}

impl Constraint {
    pub fn new_positive_exp() -> Self{
        Self::PositiveExp(PositiveExpConstraint::new())
    }

    pub fn new_no_constraint() -> Self {
        Self::NoConstraint(NoConstraint {})
    }

    pub fn new_positive_sp() -> Self {
        Self::PositiveSoftPlus(PositiveSoftPlusConstraint::new())
    }

    pub fn constrain(&self, x: f64, gradient : bool) -> Dualf64 {
        match self {
            Self::PositiveExp(inner) => {inner.constrain(x,gradient)},
            Self::PositiveSoftPlus(inner) => {inner.constrain(x,gradient)},
            Self::NoConstraint(inner) => {inner.constrain(x,gradient)},
        }
    }

    pub fn unconstrain(&self, x: f64, gradient : bool) -> Dualf64 {
        match self {
            Self::PositiveExp(inner) => {inner.unconstrain(x,gradient)},
            Self::PositiveSoftPlus(inner) => {inner.unconstrain(x,gradient)},
            Self::NoConstraint(inner) => {inner.unconstrain(x,gradient)},
        }
    }
}


