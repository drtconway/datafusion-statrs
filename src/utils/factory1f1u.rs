use datafusion::error::DataFusionError;
use statrs::distribution::Binomial;
use statrs::distribution::Discrete;
use statrs::distribution::DiscreteCDF;

pub trait Factory1F1U:
    Discrete<u64, f64> + DiscreteCDF<u64, f64> + std::fmt::Debug + Send + Sync + Sized + 'static
{
    fn make(p1: f64, p2: u64) -> Result<Self, DataFusionError>;
}

impl Factory1F1U for Binomial {
    fn make(p1: f64, p2: u64) -> Result<Self, DataFusionError> {
        Binomial::new(p1, p2).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

