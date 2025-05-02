use std::marker::PhantomData;

use datafusion::error::DataFusionError;
use statrs::distribution::{Continuous, ContinuousCDF};

use super::factory1f::Factory1F;

pub trait Evaluator2F: std::fmt::Debug + Send + Sync + 'static {
    fn eval(x: f64, p: f64) -> Result<Option<f64>, DataFusionError>;
}

#[derive(Debug)]
pub struct PdfEvaluator2F<D: Factory1F + Continuous<f64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory1F + Continuous<f64, f64>> PdfEvaluator2F<D> {
}

impl<D: Factory1F + Continuous<f64, f64>> Evaluator2F for PdfEvaluator2F<D> {
    fn eval(x: f64, p: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(p)?;
        Ok(Some(d.pdf(x)))
    }
}

#[derive(Debug)]
pub struct CdfEvaluator2F<D: Factory1F + ContinuousCDF<f64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory1F + ContinuousCDF<f64, f64>> CdfEvaluator2F<D> {
}

impl<D: Factory1F + ContinuousCDF<f64, f64>> Evaluator2F for CdfEvaluator2F<D> {
    fn eval(x: f64, p: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(p)?;
        Ok(Some(d.cdf(x)))
    }
}

#[derive(Debug)]
pub struct SfEvaluator2F<D: Factory1F + ContinuousCDF<f64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory1F + ContinuousCDF<f64, f64>> SfEvaluator2F<D> {
}

impl<D: Factory1F + ContinuousCDF<f64, f64>> Evaluator2F for SfEvaluator2F<D> {
    fn eval(x: f64, p: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(p)?;
        Ok(Some(d.sf(x)))
    }
}