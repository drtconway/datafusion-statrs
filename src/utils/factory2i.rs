use datafusion::error::DataFusionError;
use statrs::distribution::DiscreteUniform;

pub trait Factory2I:
    std::fmt::Debug + Send + Sync + Sized + 'static
{
    fn make(p1: i64, p2: i64) -> Result<Self, DataFusionError>;
}

impl Factory2I for DiscreteUniform {
    fn make(p1: i64, p2: i64) -> Result<Self, DataFusionError> {
        DiscreteUniform::new(p1, p2).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

