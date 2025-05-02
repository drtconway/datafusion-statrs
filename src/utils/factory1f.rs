use datafusion::error::DataFusionError;
use statrs::distribution::Bernoulli;
use statrs::distribution::ChiSquared;
use statrs::distribution::Dirac;
use statrs::distribution::Exp;
use statrs::distribution::Geometric;

pub trait Factory1F:
    std::fmt::Debug + Send + Sync + Sized + 'static
{
    fn make(p: f64) -> Result<Self, DataFusionError>;
}

impl Factory1F for Bernoulli {
    fn make(p: f64) -> Result<Self, DataFusionError> {
        Bernoulli::new(p).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

impl Factory1F for ChiSquared {
    fn make(p: f64) -> Result<Self, DataFusionError> {
        ChiSquared::new(p).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

impl Factory1F for Dirac {
    fn make(p: f64) -> Result<Self, DataFusionError> {
        Dirac::new(p).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

impl Factory1F for Exp {
    fn make(p: f64) -> Result<Self, DataFusionError> {
        Exp::new(p).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

impl Factory1F for Geometric {
    fn make(p: f64) -> Result<Self, DataFusionError> {
        Geometric::new(p).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}