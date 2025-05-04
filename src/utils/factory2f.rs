use datafusion::error::DataFusionError;
use statrs::distribution::Beta;
use statrs::distribution::Cauchy;
use statrs::distribution::FisherSnedecor;
use statrs::distribution::Gamma;
use statrs::distribution::Gumbel;
use statrs::distribution::InverseGamma;
use statrs::distribution::Laplace;
use statrs::distribution::LogNormal;
use statrs::distribution::NegativeBinomial;
use statrs::distribution::Normal;
use statrs::distribution::Pareto;

pub trait Factory2F:
    std::fmt::Debug + Send + Sync + Sized + 'static
{
    fn make(p1: f64, p2: f64) -> Result<Self, DataFusionError>;
}

impl Factory2F for Beta {
    fn make(p1: f64, p2: f64) -> Result<Self, DataFusionError> {
        Beta::new(p1, p2).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

impl Factory2F for Cauchy {
    fn make(p1: f64, p2: f64) -> Result<Self, DataFusionError> {
        Cauchy::new(p1, p2).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

impl Factory2F for FisherSnedecor {
    fn make(p1: f64, p2: f64) -> Result<Self, DataFusionError> {
        FisherSnedecor::new(p1, p2).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

impl Factory2F for Gamma {
    fn make(p1: f64, p2: f64) -> Result<Self, DataFusionError> {
        Gamma::new(p1, p2).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

impl Factory2F for Gumbel {
    fn make(p1: f64, p2: f64) -> Result<Self, DataFusionError> {
        Gumbel::new(p1, p2).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

impl Factory2F for InverseGamma {
    fn make(p1: f64, p2: f64) -> Result<Self, DataFusionError> {
        InverseGamma::new(p1, p2).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

impl Factory2F for Laplace {
    fn make(p1: f64, p2: f64) -> Result<Self, DataFusionError> {
        Laplace::new(p1, p2).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

impl Factory2F for LogNormal {
    fn make(p1: f64, p2: f64) -> Result<Self, DataFusionError> {
        LogNormal::new(p1, p2).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

impl Factory2F for NegativeBinomial {
    fn make(p1: f64, p2: f64) -> Result<Self, DataFusionError> {
        NegativeBinomial::new(p1, p2).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

impl Factory2F for Normal {
    fn make(p1: f64, p2: f64) -> Result<Self, DataFusionError> {
        Normal::new(p1, p2).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

impl Factory2F for Pareto {
    fn make(p1: f64, p2: f64) -> Result<Self, DataFusionError> {
        Pareto::new(p1, p2).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}
