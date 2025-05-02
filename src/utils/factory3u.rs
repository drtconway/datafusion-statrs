use datafusion::error::DataFusionError;
use statrs::distribution::Hypergeometric;

pub trait Factory3U:
    std::fmt::Debug + Send + Sync + Sized + 'static
{
    fn make(p1: u64, p2: u64, p3: u64) -> Result<Self, DataFusionError>;
}

impl Factory3U for Hypergeometric {
    fn make(p1: u64, p2: u64, p3: u64) -> Result<Self, DataFusionError> {
        Hypergeometric::new(p1, p2, p3).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

