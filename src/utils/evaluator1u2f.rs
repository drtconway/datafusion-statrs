use std::marker::PhantomData;

use datafusion::error::DataFusionError;
use statrs::distribution::{Discrete, DiscreteCDF};

use super::factory2f::Factory2F;

pub trait Evaluator1U2F: std::fmt::Debug + Send + Sync + 'static {
    fn eval(x: u64, p1: f64, p2: f64) -> Result<Option<f64>, DataFusionError>;
}

#[derive(Debug)]
pub struct PmfEvaluator1U2F<D: Factory2F + Discrete<u64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory2F + Discrete<u64, f64>> PmfEvaluator1U2F<D> {
}

impl<D: Factory2F + Discrete<u64, f64>> Evaluator1U2F for PmfEvaluator1U2F<D> {
    fn eval(x: u64, p1: f64, p2: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(p1, p2)?;
        Ok(Some(d.pmf(x)))
    }
}

#[derive(Debug)]
pub struct CdfEvaluator1U2F<D: Factory2F + DiscreteCDF<u64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory2F + DiscreteCDF<u64, f64>> CdfEvaluator1U2F<D> {
}

impl<D: Factory2F + DiscreteCDF<u64, f64>> Evaluator1U2F for CdfEvaluator1U2F<D> {
    fn eval(x: u64, p1: f64, p2: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(p1, p2)?;
        Ok(Some(d.cdf(x)))
    }
}

#[derive(Debug)]
pub struct SfEvaluator1U2F<D: Factory2F + DiscreteCDF<u64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory2F + DiscreteCDF<u64, f64>> SfEvaluator1U2F<D> {
}

impl<D: Factory2F + DiscreteCDF<u64, f64>> Evaluator1U2F for SfEvaluator1U2F<D> {
    fn eval(x: u64, p1: f64, p2: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(p1, p2)?;
        Ok(Some(d.sf(x)))
    }
}