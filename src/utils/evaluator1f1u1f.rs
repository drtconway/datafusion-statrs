use std::marker::PhantomData;

use datafusion::error::DataFusionError;
use statrs::distribution::{Continuous, ContinuousCDF};

use super::factory1u1f::Factory1U1F;

pub trait Evaluator1F1U1F: std::fmt::Debug + Send + Sync + 'static {
    fn eval(x: f64, n: u64, p: f64) -> Result<Option<f64>, DataFusionError>;
}

#[derive(Debug)]
pub struct PdfEvaluator1F1U1F<D: Factory1U1F + Continuous<f64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory1U1F + Continuous<f64, f64>> PdfEvaluator1F1U1F<D> {}

impl<D: Factory1U1F + Continuous<f64, f64>> Evaluator1F1U1F for PdfEvaluator1F1U1F<D> {
    fn eval(x: f64, n: u64, p: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(n, p)?;
        Ok(Some(d.pdf(x)))
    }
}

#[derive(Debug)]
pub struct CdfEvaluator1F1U1F<D: Factory1U1F + ContinuousCDF<f64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory1U1F + ContinuousCDF<f64, f64>> CdfEvaluator1F1U1F<D> {}

impl<D: Factory1U1F + ContinuousCDF<f64, f64>> Evaluator1F1U1F for CdfEvaluator1F1U1F<D> {
    fn eval(x: f64, n: u64, p: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(n, p)?;
        Ok(Some(d.cdf(x)))
    }
}

#[derive(Debug)]
pub struct SfEvaluator1F1U1F<D: Factory1U1F + ContinuousCDF<f64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory1U1F + ContinuousCDF<f64, f64>> SfEvaluator1F1U1F<D> {}

impl<D: Factory1U1F + ContinuousCDF<f64, f64>> Evaluator1F1U1F for SfEvaluator1F1U1F<D> {
    fn eval(x: f64, n: u64, p: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(n, p)?;
        Ok(Some(d.sf(x)))
    }
}
