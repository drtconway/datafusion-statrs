use std::marker::PhantomData;

use datafusion::error::DataFusionError;
use statrs::distribution::{Discrete, DiscreteCDF};

use super::factory3u::Factory3U;

pub trait Evaluator4U: std::fmt::Debug + Send + Sync + 'static {
    fn eval(x: u64, p1: u64, p2: u64, p3: u64) -> Result<Option<f64>, DataFusionError>;
}

#[derive(Debug)]
pub struct PmfEvaluator4U<D: Factory3U + Discrete<u64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory3U + Discrete<u64, f64>> Evaluator4U for PmfEvaluator4U<D> {
    fn eval(x: u64, p1: u64, p2: u64, p3: u64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(p1, p2, p3)?;
        Ok(Some(d.pmf(x)))
    }
}

#[derive(Debug)]
pub struct CdfEvaluator4U<D: Factory3U + DiscreteCDF<u64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory3U + DiscreteCDF<u64, f64>> Evaluator4U for CdfEvaluator4U<D> {
    fn eval(x: u64, p1: u64, p2: u64, p3: u64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(p1, p2, p3)?;
        Ok(Some(d.cdf(x)))
    }
}

#[derive(Debug)]
pub struct SfEvaluator4U<D: Factory3U + DiscreteCDF<u64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory3U + DiscreteCDF<u64, f64>> Evaluator4U for SfEvaluator4U<D> {
    fn eval(x: u64, p1: u64, p2: u64, p3: u64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(p1, p2, p3)?;
        Ok(Some(d.sf(x)))
    }
}