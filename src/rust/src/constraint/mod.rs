pub mod constraint;
pub mod constraints;
mod positive;

use extendr_api::prelude::*;

extendr_module!{
    mod constraint;
    use constraints;
}