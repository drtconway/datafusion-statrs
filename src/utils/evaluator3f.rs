use std::marker::PhantomData;

use datafusion::error::DataFusionError;
use statrs::distribution::{Continuous, ContinuousCDF};

use super::factory2f::Factory2F;

pub trait Evaluator3F: std::fmt::Debug + Send + Sync + 'static {
    fn eval(x: f64, p1: f64, p2: f64) -> Result<Option<f64>, DataFusionError>;
}

#[derive(Debug)]
pub struct PdfEvaluator3F<D: Factory2F + Continuous<f64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory2F + Continuous<f64, f64>> PdfEvaluator3F<D> {
}

impl<D: Factory2F + Continuous<f64, f64>> Evaluator3F for PdfEvaluator3F<D> {
    fn eval(x: f64, p1: f64, p2: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(p1, p2)?;
        Ok(Some(d.pdf(x)))
    }
}

#[derive(Debug)]
pub struct CdfEvaluator3F<D: Factory2F + ContinuousCDF<f64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory2F + ContinuousCDF<f64, f64>> CdfEvaluator3F<D> {
    pub fn new() -> Self {
        CdfEvaluator3F {
            _phantom: PhantomData,
        }
    }
}

impl<D: Factory2F + ContinuousCDF<f64, f64>> Evaluator3F for CdfEvaluator3F<D> {
    fn eval(x: f64, p1: f64, p2: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(p1, p2)?;
        Ok(Some(d.cdf(x)))
    }
}

#[derive(Debug)]
pub struct SfEvaluator3F<D: Factory2F + ContinuousCDF<f64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory2F + ContinuousCDF<f64, f64>> SfEvaluator3F<D> {
    pub fn new() -> Self {
        SfEvaluator3F {
            _phantom: PhantomData,
        }
    }
}

impl<D: Factory2F + ContinuousCDF<f64, f64>> Evaluator3F for SfEvaluator3F<D> {
    fn eval(x: f64, p1: f64, p2: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(p1, p2)?;
        Ok(Some(d.sf(x)))
    }
}