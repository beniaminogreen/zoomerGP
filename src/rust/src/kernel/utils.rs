use extendr_api::prelude::*;

use super::composite::CompositeKernel;
use super::indicator::IndicatorKernel;
use super::linear::LinearKernel;
use super::multiplicative::MultiplicativeKernel;
use super::spectral_mixture::SpectralMixtureKernel;
use super::rbf::ExpQuadKernel;
use super::periodic::PeriodicKernel;
use super::if_kernel::IfKernel;


use super::kernel::Kernel;
use ndarray_linalg::cholesky::{Cholesky, UPLO};


use ndarray::{Array2, Axis, Array3, ArrayView2};
use ndarray_linalg::SVD;
use crate::dataset::DataManager;

//#[allow(non_snake_case)]
//pub fn matrix_inner_product(x: ArrayView2<f64>) -> Array2<f64> {
//    let (_, R) = x.qr().unwrap();
//    R.t().dot(&R)
//}
//
pub fn get_cond_number(x: ArrayView2<f64>) -> Result<f64> {
    let svd = x.svd(false, false).map_err(|e| Error::Other(format!("SVD computation failed: {:?}", e)))?;
    let singular_values = svd.1;
    // The condition number is the ratio of the largest to the smallest singular value
    Ok(singular_values[0] / singular_values[singular_values.len() - 1])
}

pub fn fix_conditioning(x: &mut Array2<f64>) {
    let mut iter = -8;
    let mut cond_number = get_cond_number(x.view());
    while cond_number.unwrap().log10() > 4.0 && iter < -1 {
        let amount_to_add = (10.0_f64).powi(iter);
        for i in 0..x.nrows() {
            x[[i, i]] += amount_to_add;
        }
        iter += 1;
        cond_number = get_cond_number(x.view());
    }
}

pub fn stable_log_det(x: ArrayView2<f64>) -> Result<f64> {
    let chol = x.cholesky(UPLO::Lower).map_err(|e| Error::Other(format!("Singular matrix when computing log det {:?}", e)))?;
    let log_det = 2.0 * chol.diag().map(|x| x.ln()).sum();
    Ok(log_det)
}

//
//#[allow(non_snake_case)]
pub fn fast_gradient_matrix(
    A: ArrayView2<f64>,
    B: ArrayView2<f64>,
    kernel_func: &Box<dyn Kernel>,
) -> Array3<f64> {
    let mut out: Array3<f64> = Array3::zeros((A.nrows(), B.nrows(), kernel_func.num_params()));
    out.axis_iter_mut(Axis(0))
        .into_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for (j, mut column) in row.axis_iter_mut(Axis(0)).enumerate() {
                column.assign(&kernel_func.calc(A.row(i).view(), B.row(j).view(), true).grad.unwrap());
            }
        });

    out
}

pub fn parse_kernel_recursive(kernel_specification: List, dataset : &dyn DataManager) -> Box<dyn Kernel> {
    let kernel_dict = kernel_specification.into_hashmap();

    if let Some(composite_type) = kernel_dict.get("type") {
        let left: List = List::try_from(kernel_dict.get("left").unwrap()).unwrap();
        let right: List = List::try_from(kernel_dict.get("right").unwrap()).unwrap();
        match composite_type.as_str().unwrap() {
            "multiply" => Box::new(MultiplicativeKernel::new(
                parse_kernel_recursive(left, dataset),
                parse_kernel_recursive(right, dataset),
            )),
            "add" => Box::new(CompositeKernel::new(vec![
                parse_kernel_recursive(left, dataset),
                parse_kernel_recursive(right, dataset),
            ])),
            _ => panic!("Supplied an unsupported type of composite kernel"),
        }
    } else {
        let ktype = kernel_dict["kernel"].as_str().unwrap();
        let cols: Vec<usize> = kernel_dict
            .get("cols")
            .unwrap()
            .as_integer_vector()
            .unwrap()
            .into_iter()
            .map(|x| (x - 1) as usize)
            .collect();

        let kwargs = List::try_from(kernel_dict.get("kwargs").unwrap()).unwrap().into_hashmap();

        match ktype {
            "rbf" => Box::new(ExpQuadKernel::new(cols)),
            "linear" => Box::new(LinearKernel::new(cols)),
            "linear_rbf" => Box::new(MultiplicativeKernel::new(
                Box::new(LinearKernel::new(cols.clone())),
                Box::new(ExpQuadKernel::new(cols)),
            )),
            "id" => Box::new(IndicatorKernel::new(cols)),
            "mask" => Box::new(IfKernel::new(cols)),
            "spectral1" => Box::new(SpectralMixtureKernel::new(cols, 1)),
            "spectral2" => Box::new(SpectralMixtureKernel::new(cols, 2)),
            "spectral3" => Box::new(SpectralMixtureKernel::new(cols, 3)),
            "spectral4" => Box::new(SpectralMixtureKernel::new(cols, 4)),
            "spectral5" => Box::new(SpectralMixtureKernel::new(cols, 5)),
            "periodic" => Box::new(PeriodicKernel::new(cols, kwargs, dataset)),
            _ => panic!("Invalid Kernel Type"),
        }
    }
}

fn get_uppercase_letter(index: usize) -> Option<char> {
    if index < 26 {
        Some((b'A' + index as u8) as char)
    } else {
        None // Out of range
    }
}

pub fn build_kernel_tree(kernel: &dyn Kernel, prefix: &str, label : &str, is_last: bool) -> String {
    let branch = if is_last { "└── " } else { "├── " };
    let mut result = format!("{}{} ({}) {}\n", prefix, branch, label, kernel.describe());

    let new_prefix = if is_last {
        format!("{}    ", prefix)
    } else {
        format!("{}│   ", prefix)
    };

    let children = kernel.children();
    for (i, child) in children.iter().enumerate() {
        let last = i == children.len() - 1;
        let mut new_label = String::with_capacity(1 + label.len());
        new_label.push_str(label);
        new_label.push(get_uppercase_letter(i).unwrap());
        result.push_str(&build_kernel_tree(child.as_ref(), &new_prefix, &new_label, last));
    }

    result
}

fn remove_first_n_chars(s: &str, n: usize) -> &str {
    let mut chars = s.chars();
    for _ in 0..n {
        chars.next();
    }
    chars.as_str()
}

pub fn recusive_select_kernel<'a>(kernel : &'a dyn Kernel, search_string : &str) -> &'a dyn Kernel {
    if search_string.is_empty() {
        panic!("Search string is empty");
    }

    if search_string.len() == 1 {
        return kernel;
    }

    let requested_letter = search_string.chars().nth(1).unwrap();
    let suffix = remove_first_n_chars(search_string, 1);
    for (i, child) in kernel.children().iter().enumerate() {
        let letter_label = get_uppercase_letter(i).unwrap();
        if letter_label ==  requested_letter {
            return recusive_select_kernel(child.as_ref(), suffix)
        };
    };

    panic!("No matching child found for '{}'", requested_letter);
}
