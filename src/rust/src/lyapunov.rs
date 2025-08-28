use ndarray::{Array2, ArrayView2};

use extendr_api::prelude::*;

use ndarray_linalg::Trace;
use extendr_api::prelude::*;


// forward nyap ns from https://people.cs.umass.edu/~smaji/projects/matrix-sqrt/
// Y converges to A^(1/2) and Z converges to A^(-1/2)
#[allow(non_snake_case)]
pub fn lyap_newton_shulz_fwd(A : ArrayView2<f64>, n_iter : usize) -> (Array2<f64>, Array2<f64>) {
        let d = A.ncols();
        let norm =  A.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let norm = norm * 20.0;

        let mut Y = A.to_owned() / norm;
        let I = Array2::eye(d);
        let mut Z = Array2::eye(d);

        for _ in 0..n_iter {
            let T : Array2<f64> = 0.5 * (3.0 * &I - Z.dot(&Y));
            Y = Y.dot(&T);
            Z = T.dot(&Z);
        }

     (Y * norm.sqrt(), Z * norm.sqrt())
}

pub fn lyap_newton_shulz_backward(
    Z: ArrayView2<f64>,
    DlDZ: ArrayView2<f64>,
    n_iter: usize
) -> Array2<f64> {
    let n = Z.nrows();
    let d = Z.ncols();
    let norm = Z.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    let mut a = Z.to_owned() / norm;
    let I = 3.0 * Array2::eye(d);
    let mut q = DlDZ.to_owned() / norm;

    for _ in 0..n_iter {
        let temp = &I - a.dot(&a);
        q = 0.5 * (q.dot(&temp) - a.t().dot(&(a.t().dot(&q) - q.dot(&a))));
        a = 0.5 * a.dot(&temp);
    }

    0.5 * q
}


#[extendr]
fn r_lyap_fwd(X : ArrayView2<f64>, iter : usize)  -> Robj {
    let (Y,Z) = lyap_newton_shulz_fwd(X.view(), iter);

    Robj::try_from(&Y).into()
}

#[extendr]
fn r_lyap_backward(Z: ArrayView2<f64>,  DlDZ: ArrayView2<f64>, n_iter: usize) -> Robj {
    let x = lyap_newton_shulz_backward(Z, DlDZ, n_iter);

    Robj::try_from(&x).into()
}


extendr_module! {
    mod lyapunov;
    fn r_lyap_fwd;
    fn r_lyap_backward;
}