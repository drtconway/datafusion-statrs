use datafusion::error::DataFusionError;
use statrs::distribution::Chi;

pub trait Factory1U:
    std::fmt::Debug + Send + Sync + Sized + 'static
{
    fn make(p: u64) -> Result<Self, DataFusionError>;
}


impl Factory1U for Chi {
    fn make(p: u64) -> Result<Self, DataFusionError> {
        Chi::new(p).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}
