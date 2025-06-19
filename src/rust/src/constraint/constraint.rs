use crate::dual_number::Dualf64;

use super::positive::{PositiveExpConstraint, PositiveSoftPlusConstraint};

use extendr_api::prelude::*;

#[derive(Clone, Debug)]
#[extendr]
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

    pub fn new_positive_sp() -> Self {
        Self::PositiveSoftPlus(PositiveSoftPlusConstraint::new())
    }

    pub fn constrain(&self, x: f64, gradient : bool) -> Dualf64 {
        match self {
            Self::PositiveExp(inner) => {inner.constrain(x,gradient)},
            Self::PositiveSoftPlus(inner) => {inner.constrain(x,gradient)},
            Self::NoConstraint => {Dualf64::from(x)}
        }
    }

    pub fn unconstrain(&self, x: f64, gradient : bool) -> Dualf64 {
        match self {
            Self::PositiveExp(inner) => {inner.unconstrain(x,gradient)},
            Self::PositiveSoftPlus(inner) => {inner.constrain(x,gradient)},
            Self::NoConstraint => {Dualf64::from(x)}
        }
    }
}


