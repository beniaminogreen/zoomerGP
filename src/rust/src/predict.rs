use extendr_api::Doubles;
use extendr_api::prelude::*;
use ndarray::Array1;

#[extendr]
pub struct PredictionOutput {
    response : Array1<f64>,
    f_var : Option<Array1<f64>>,
    pred_var : Option<Array1<f64>>
}

impl From<(Array1<f64>,Option<Array1<f64>>, Option<Array1<f64>>)> for PredictionOutput {
    fn from(x : (Array1<f64>,Option<Array1<f64>>, Option<Array1<f64>>)) -> Self {
        Self{
            response : x.0,
            f_var : x.1,
            pred_var : x.2
        }
    }
}
#[extendr]
impl PredictionOutput {
    fn get_response(&self) -> Vec<f64> {
        self.response.to_vec()
    }

    fn get_f_var(&self) -> Option<Doubles> {
        match self.f_var { 
            None => None,
            Some(ref f_var) => {Some(Doubles::from_values(f_var.to_vec()))}
        }
    }

    fn get_pred_var(&self) -> Option<Doubles> {
        match self.pred_var {
            None => None,
            Some(ref pred_var) => {
                Some(Doubles::from_values(pred_var.to_vec()))
            }
        }
    }
}

extendr_module!{
    mod predict;
    impl PredictionOutput;
}