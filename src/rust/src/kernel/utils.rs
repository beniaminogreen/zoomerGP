use extendr_api::prelude::*;

use super::composite::CompositeKernel;
use super::indicator::IndicatorKernel;
use super::linear::LinearKernel;
use super::multiplicative::MultiplicativeKernel;
use super::spectral_mixture::SpectralMixtureKernel;
use super::rbf::ExpQuadKernel;

use super::kernel::Kernel;

//#[allow(non_snake_case)]
//pub fn matrix_inner_product(x: ArrayView2<f64>) -> Array2<f64> {
//    let (_, R) = x.qr().unwrap();
//    R.t().dot(&R)
//}
//
//pub fn get_cond_number(x: ArrayView2<f64>) -> f64 {
//    let svd = x.svd(false, false).unwrap();
//    let singular_values = svd.1;
//    // The condition number is the ratio of the largest to the smallest singular value
//    singular_values[0] / singular_values[singular_values.len() - 1]
//}
//
//pub fn fix_conditioning(x: &mut Array2<f64>) {
//    let mut iter = -8;
//    let mut cond_number = get_cond_number(x.view());
//    while cond_number.log10() > 4.0 && iter < -1 {
//        let amount_to_add = (10.0_f64).powi(iter);
//        for i in 0..x.nrows() {
//            x[[i, i]] += amount_to_add;
//        }
//        iter += 1;
//        cond_number = get_cond_number(x.view());
//    }
//}
//
//#[allow(non_snake_case)]
//pub fn fast_gradient_matrix(
//    A: ArrayView2<f64>,
//    B: ArrayView2<f64>,
//    kernel_func: &Box<dyn KernelFunc>,
//) -> Array3<f64> {
//    let mut out: Array3<f64> = Array3::zeros((A.nrows(), B.nrows(), kernel_func.num_params()));
//    out.axis_iter_mut(Axis(0))
//        .into_par_iter()
//        .enumerate()
//        .for_each(|(i, mut row)| {
//            for (j, mut column) in row.axis_iter_mut(Axis(0)).enumerate() {
//                column.assign(&kernel_func.calculate_gradient(A.row(i).view(), B.row(j).view()));
//            }
//        });
//
//    out
//}

pub fn parse_kernel_recursive(kernel_specification: List) -> Box<dyn Kernel> {
    let kernel_dict = kernel_specification.into_hashmap();

    if let Some(composite_type) = kernel_dict.get("type") {
        let left: List = List::try_from(kernel_dict.get("left").unwrap()).unwrap();
        let right: List = List::try_from(kernel_dict.get("right").unwrap()).unwrap();
        match composite_type.as_str().unwrap() {
            "multiply" => Box::new(MultiplicativeKernel::new(
                parse_kernel_recursive(left),
                parse_kernel_recursive(right),
            )),
            "add" => Box::new(CompositeKernel::new(vec![
                parse_kernel_recursive(left),
                parse_kernel_recursive(right),
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

        match ktype {
            "rbf" => Box::new(ExpQuadKernel::new(cols)),
            "linear" => Box::new(LinearKernel::new(cols)),
            "linear_rbf" => Box::new(MultiplicativeKernel::new(
                Box::new(LinearKernel::new(cols.clone())),
                Box::new(ExpQuadKernel::new(cols)),
            )),
            "id" => Box::new(IndicatorKernel::new(cols)),
            "spectral1" => Box::new(SpectralMixtureKernel::new(cols, 1)),
            "spectral2" => Box::new(SpectralMixtureKernel::new(cols, 2)),
            "spectral3" => Box::new(SpectralMixtureKernel::new(cols, 3)),
            "spectral4" => Box::new(SpectralMixtureKernel::new(cols, 4)),
            "spectral5" => Box::new(SpectralMixtureKernel::new(cols, 5)),
            _ => panic!("Invalid Kernel Type"),
        }
    }
}

pub fn print_kernel_tree(kernel: &dyn Kernel, prefix: &str, is_last: bool) {
    let branch = if is_last { "└── " } else { "├── " };
    println!("{}{}{}", prefix, branch, kernel.describe());

    let new_prefix = if is_last {
        format!("{}    ", prefix)
    } else {
        format!("{}│   ", prefix)
    };

    let children = kernel.children();
    for (i, child) in children.iter().enumerate() {
        let last = i == children.len() - 1;
        print_kernel_tree(child.as_ref(), &new_prefix, last);
    }
}
