use std::marker::PhantomData;

use datafusion::error::DataFusionError;
use statrs::distribution::{Continuous, ContinuousCDF};

use super::factory1u::Factory1U;

pub trait Evaluator1F1U: std::fmt::Debug + Send + Sync + 'static {
    fn eval(x: f64, n: u64) -> Result<Option<f64>, DataFusionError>;
}

#[derive(Debug)]
pub struct PdfEvaluator1F1U<D: Factory1U + Continuous<f64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory1U + Continuous<f64, f64>> PdfEvaluator1F1U<D> {
}

impl<D: Factory1U + Continuous<f64, f64>> Evaluator1F1U for PdfEvaluator1F1U<D> {
    fn eval(x: f64, n: u64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(n)?;
        Ok(Some(d.pdf(x)))
    }
}

#[derive(Debug)]
pub struct CdfEvaluator1F1U<D: Factory1U + ContinuousCDF<f64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory1U + ContinuousCDF<f64, f64>> CdfEvaluator1F1U<D> {
}

impl<D: Factory1U + ContinuousCDF<f64, f64>> Evaluator1F1U for CdfEvaluator1F1U<D> {
    fn eval(x: f64, n: u64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(n)?;
        Ok(Some(d.cdf(x)))
    }
}

#[derive(Debug)]
pub struct SfEvaluator1F1U<D: Factory1U + ContinuousCDF<f64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory1U + ContinuousCDF<f64, f64>> SfEvaluator1F1U<D> {
}

impl<D: Factory1U + ContinuousCDF<f64, f64>> Evaluator1F1U for SfEvaluator1F1U<D> {
    fn eval(x: f64, n: u64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(n)?;
        Ok(Some(d.sf(x)))
    }
}