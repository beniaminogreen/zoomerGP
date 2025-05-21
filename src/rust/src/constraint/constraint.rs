use crate::dual_number::Dual;

use super::positive::{PositiveExpConstraint, PositiveSoftPlusConstraint};

#[derive(Clone, Debug)]
pub enum Constraint  {
    PositiveExp(PositiveExpConstraint),
    PositiveSoftPlus(PositiveSoftPlusConstraint),
    NoConstraint
}

impl Constraint {
    pub fn new_positive_exp() -> Self{
        Self::PositiveExp(PositiveExpConstraint::new())
    }

    pub fn new_no_constraint() -> Self {
        Self::NoConstraint
    }

    pub fn constrain(&self, x: f64, gradient : bool) -> Dual {
        match self {
            Self::PositiveExp(inner) => {inner.constrain(x,gradient)},
            Self::PositiveSoftPlus(inner) => {inner.constrain(x,gradient)},
            Self::NoConstraint => {Dual::from(x)}
        }
    }

    pub fn unconstrain(&self, x: f64, gradient : bool) -> Dual {
        match self {
            Self::PositiveExp(inner) => {inner.unconstrain(x,gradient)},
            Self::PositiveSoftPlus(inner) => {inner.constrain(x,gradient)},
            Self::NoConstraint => {Dual::from(x)}
        }
    }
}
