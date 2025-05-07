use std::marker::PhantomData;

use datafusion::error::DataFusionError;
use statrs::distribution::{Continuous, ContinuousCDF};

use super::factory3f::Factory3F;

pub trait Evaluator4F: std::fmt::Debug + Send + Sync + 'static {
    fn eval(x: f64, p1: f64, p2: f64, p3: f64) -> Result<Option<f64>, DataFusionError>;
}

#[derive(Debug)]
pub struct PdfEvaluator4F<D: Factory3F + Continuous<f64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory3F + Continuous<f64, f64>> Evaluator4F for PdfEvaluator4F<D> {
    fn eval(x: f64, p1: f64, p2: f64, p3: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(p1, p2, p3)?;
        Ok(Some(d.pdf(x)))
    }
}

#[derive(Debug)]
pub struct LnPdfEvaluator4F<D: Factory3F + Continuous<f64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory3F + Continuous<f64, f64>> Evaluator4F for LnPdfEvaluator4F<D> {
    fn eval(x: f64, p1: f64, p2: f64, p3: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(p1, p2, p3)?;
        Ok(Some(d.ln_pdf(x)))
    }
}

#[derive(Debug)]
pub struct CdfEvaluator4F<D: Factory3F + ContinuousCDF<f64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory3F + ContinuousCDF<f64, f64>> Evaluator4F for CdfEvaluator4F<D> {
    fn eval(x: f64, p1: f64, p2: f64, p3: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(p1, p2, p3)?;
        Ok(Some(d.cdf(x)))
    }
}

#[derive(Debug)]
pub struct SfEvaluator4F<D: Factory3F + ContinuousCDF<f64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory3F + ContinuousCDF<f64, f64>> Evaluator4F for SfEvaluator4F<D> {
    fn eval(x: f64, p1: f64, p2: f64, p3: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(p1, p2, p3)?;
        Ok(Some(d.sf(x)))
    }
}