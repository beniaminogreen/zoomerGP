use ndarray::{ArrayView2, ArrayView1, Array1, Array2, Axis};
use crate::dual_number::DualArr;

#[allow(non_snake_case)]
pub trait DataManager : Send + Sync {
    fn get_X(&self) -> ArrayView2<f64>;
    fn get_x_scale_factor(&self) -> Array1<f64>;
    fn shape(&self)  -> (usize,usize);
    fn scale_predictors(&self, x: ArrayView2<f64>) -> Array2<f64>;
}

pub struct UnitStandardizedDataset {
    mins: Array1<f64>,
    ranges: Array1<f64>,
    scaled_x : Array2<f64>,
}

impl UnitStandardizedDataset {
    pub fn new(x: Array2<f64>, y : Array1<f64>) -> Self {
        let mins_and_maxes: Vec<(f64, f64)> = x
            .axis_iter(Axis(1))
            .map(|column| {
                let (min, max) = column
                    .iter()
                    .cloned()
                    .fold(None, |m: Option<(f64, f64)>, x| {
                        m.map_or(Some((x, x)), |(m1, m2)| Some((m1.min(x), m2.max(x))))
                    })
                    .unwrap();
                (min, max)
            })
            .collect();

        let mins = Array1::from_iter(mins_and_maxes.iter().map(|(min, _)| *min));
        let ranges = Array1::from_iter(mins_and_maxes.iter().map(|(min, max)| max - min));


        let mut scaled_x = Array2::zeros((x.nrows(), x.ncols()));

        for (i, column) in x.axis_iter(Axis(1)).enumerate() {
            for j in 0..column.len() {
                scaled_x[[j, i]] = (column[j] - mins[i]) / ranges[i];
            }
        }

        let y_bar = y.iter().sum::<f64>() / (y.len() as f64);
        let demeaned_y = y - y_bar;

        Self {
            mins,
            ranges,
            scaled_x,
        }
    }
}

impl DataManager for UnitStandardizedDataset {
    fn get_x_scale_factor(&self) -> Array1<f64> {
        1.0/self.ranges.clone()
    }
    fn get_X(&self) -> ArrayView2<f64> {
        self.scaled_x.view()
    }
    fn scale_predictors(&self, x : ArrayView2<f64>) -> Array2<f64> {
        let mut out = Array2::zeros((x.nrows(), x.ncols()));
        
        for (i, column) in x.axis_iter(Axis(1)).enumerate() {
            for j in 0..column.len() {
                out[[j,i]] = (column[j] - self.mins[i]) / self.ranges[i];
            }
        }
        
        out 
    }

    fn shape(&self) -> (usize, usize) {
        (
            self.scaled_x.nrows(),
            self.scaled_x.ncols()
        )
    }
}
