use datafusion::error::DataFusionError;
use statrs::distribution::{StudentsT, Triangular};

pub trait Factory3F:
    std::fmt::Debug + Send + Sync + Sized + 'static
{
    fn make(p1: f64, p2: f64, p3: f64) -> Result<Self, DataFusionError>;
}

impl Factory3F for StudentsT {
    fn make(p1: f64, p2: f64, p3: f64) -> Result<Self, DataFusionError> {
        StudentsT::new(p1, p2, p3).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

impl Factory3F for Triangular {
    fn make(p1: f64, p2: f64, p3: f64) -> Result<Self, DataFusionError> {
        Triangular::new(p1, p2, p3).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}
