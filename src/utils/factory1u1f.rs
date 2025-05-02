use datafusion::error::DataFusionError;
use statrs::distribution::{Binomial, Erlang};

pub trait Factory1U1F:
    std::fmt::Debug + Send + Sync + Sized + 'static
{
    fn make(p1: u64, p2: f64) -> Result<Self, DataFusionError>;
}

impl Factory1U1F for Binomial {
    fn make(p1: u64, p2: f64) -> Result<Self, DataFusionError> {
        Binomial::new(p2, p1).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

impl Factory1U1F for Erlang {
    fn make(p1: u64, p2: f64) -> Result<Self, DataFusionError> {
        Erlang::new(p1, p2).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}
