use std::marker::PhantomData;

use datafusion::error::DataFusionError;
use statrs::distribution::{Discrete, DiscreteCDF};

use super::factory1f::Factory1F;

pub trait Evaluator1U1F: std::fmt::Debug + Send + Sync + 'static {
    fn eval(x: u64, p: f64) -> Result<Option<f64>, DataFusionError>;
}

#[derive(Debug)]
pub struct PmfEvaluator1U1F<D: Factory1F + Discrete<u64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory1F + Discrete<u64, f64>> PmfEvaluator1U1F<D> {
}

impl<D: Factory1F + Discrete<u64, f64>> Evaluator1U1F for PmfEvaluator1U1F<D> {
    fn eval(x: u64, p: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(p)?;
        Ok(Some(d.pmf(x)))
    }
}

#[derive(Debug)]
pub struct CdfEvaluator1U1F<D: Factory1F + DiscreteCDF<u64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory1F + DiscreteCDF<u64, f64>> CdfEvaluator1U1F<D> {
}

impl<D: Factory1F + DiscreteCDF<u64, f64>> Evaluator1U1F for CdfEvaluator1U1F<D> {
    fn eval(x: u64, p: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(p)?;
        Ok(Some(d.cdf(x)))
    }
}

#[derive(Debug)]
pub struct SfEvaluator1U1F<D: Factory1F + DiscreteCDF<u64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory1F + DiscreteCDF<u64, f64>> SfEvaluator1U1F<D> {
}

impl<D: Factory1F + DiscreteCDF<u64, f64>> Evaluator1U1F for SfEvaluator1U1F<D> {
    fn eval(x: u64, p: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(p)?;
        Ok(Some(d.sf(x)))
    }
}