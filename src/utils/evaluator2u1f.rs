use std::marker::PhantomData;

use datafusion::error::DataFusionError;
use statrs::distribution::{Discrete, DiscreteCDF};

use super::factory1u1f::Factory1U1F;

pub trait Evaluator2U1F: std::fmt::Debug + Send + Sync + 'static {
    fn eval(x: u64, n: u64, p: f64) -> Result<Option<f64>, DataFusionError>;
}

#[derive(Debug)]
pub struct PmfEvaluator2U1F<D: Factory1U1F + Discrete<u64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory1U1F + Discrete<u64, f64>> Evaluator2U1F for PmfEvaluator2U1F<D> {
    fn eval(x: u64, n: u64, p: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(n, p)?;
        Ok(Some(d.pmf(x)))
    }
}

#[derive(Debug)]
pub struct LnPmfEvaluator2U1F<D: Factory1U1F + Discrete<u64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory1U1F + Discrete<u64, f64>> Evaluator2U1F for LnPmfEvaluator2U1F<D> {
    fn eval(x: u64, n: u64, p: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(n, p)?;
        Ok(Some(d.ln_pmf(x)))
    }
}

#[derive(Debug)]
pub struct CdfEvaluator2U1F<D: Factory1U1F + DiscreteCDF<u64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory1U1F + DiscreteCDF<u64, f64>> Evaluator2U1F for CdfEvaluator2U1F<D> {
    fn eval(x: u64, n: u64, p: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(n, p)?;
        Ok(Some(DiscreteCDF::cdf(&d, x)))
    }
}

#[derive(Debug)]
pub struct SfEvaluator2U1F<D: Factory1U1F + DiscreteCDF<u64, f64>> {
    _phantom: PhantomData<D>,
}

impl<D: Factory1U1F + DiscreteCDF<u64, f64>> Evaluator2U1F for SfEvaluator2U1F<D> {
    fn eval(x: u64, n: u64, p: f64) -> Result<Option<f64>, DataFusionError> {
        let d = D::make(n, p)?;
        Ok(Some(DiscreteCDF::sf(&d, x)))
    }
}