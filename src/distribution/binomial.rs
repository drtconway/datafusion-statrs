//! Module containing functions to the Binomial Distribution.
//! 
//! Implemented by [`statrs::distribution::Binomial`].
//! 
//! The [Binomial Distribution](https://en.wikipedia.org/wiki/Binomial_distribution) has two
//! parameters:
//! 
//! n: n ∈ N (natural numbers)  
//! p: 0 ≤ p ≤ 1
//! 
//! Usage:
//! 
//! `binomial_pmf(x, n, p)`  
//! `binomial_ln_pmf(x, n, p)`  
//! `binomial_cdf(x, n, p)`  
//! `binomial_sf(x, n, p)`
//! 
//! with
//! 
//!   `x`: 0 ≤ x ≤ n `UInt64`/`BIGINT UNSIGNED`,  
//!   `n`: 0 ≤ n `UInt64`/`BIGINT UNSIGNED`,  
//!   `p`: [0, 1] `Float64`/`DOUBLE`
//! 
//! Examples
//! ```
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() -> std::io::Result<()> {
//!     let mut ctx = datafusion::prelude::SessionContext::new();
//!     datafusion_statrs::distribution::binomial::register(&mut ctx)?;
//!     ctx.sql("SELECT binomial_cdf(CAST(2 AS BIGINT UNSIGNED), CAST(5 AS BIGINT UNSIGNED), 0.2)").await?
//!        .show().await?;
//!     Ok(())
//! }
//! ```

use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::Binomial;

use crate::utils::discrete2u1f::Discrete2U1F;
use crate::utils::evaluator2u1f::{CdfEvaluator2U1F, LnPmfEvaluator2U1F, PmfEvaluator2U1F, SfEvaluator2U1F};

type Pmf = Discrete2U1F<PmfEvaluator2U1F<Binomial>>;

/// ScalarUDF for the Binomial Distribution PMF
pub fn pmf() -> ScalarUDF {
    ScalarUDF::from(Pmf::new("binomial_pmf"))
}

type LnPmf = Discrete2U1F<LnPmfEvaluator2U1F<Binomial>>;

/// ScalarUDF for the Binomial Distribution PMF
pub fn ln_pmf() -> ScalarUDF {
    ScalarUDF::from(LnPmf::new("binomial_ln_pmf"))
}

type Cdf = Discrete2U1F<CdfEvaluator2U1F<Binomial>>;

/// ScalarUDF for the Binomial Distribution CDF
pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("binomial_cdf"))
}

type Sf = Discrete2U1F<SfEvaluator2U1F<Binomial>>;

/// ScalarUDF for the Binomial Distribution SF
pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("binomial_sf"))
}

/// Register the functions for the Binomial Distribution
pub fn register(registry: &mut dyn FunctionRegistry) -> Result<(), DataFusionError> {
    crate::utils::register::register(registry, vec![pmf(), ln_pmf(), cdf(), sf()])
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use assert_eq_float::assert_eq_float;
    use datafusion::{
        arrow::{
            array::{Float64Array, RecordBatch, UInt64Array},
            datatypes::{DataType, Field, Schema, SchemaRef},
        },
        common::cast::as_float64_array,
        error::DataFusionError,
        prelude::{SessionContext, col},
    };
    use statrs::distribution::BinomialError;

    use super::*;

    fn get_schema() -> SchemaRef {
        SchemaRef::new(Schema::new(vec![
            Field::new("x", DataType::UInt64, true),
            Field::new("n", DataType::UInt64, true),
            Field::new("p", DataType::Float64, true),
        ]))
    }

    fn make_records(rows: Vec<(Option<u64>, Option<u64>, Option<f64>)>) -> RecordBatch {
        let mut xs = Vec::new();
        let mut ns = Vec::new();
        let mut ps = Vec::new();
        for row in rows {
            xs.push(row.0);
            ns.push(row.1);
            ps.push(row.2);
        }

        RecordBatch::try_new(
            get_schema(),
            vec![
                Arc::new(UInt64Array::from(xs)),
                Arc::new(UInt64Array::from(ns)),
                Arc::new(Float64Array::from(ps)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn binomial_pmf_success() {
        let pmf = pmf();

        let recs = make_records(vec![
            (Some(0), Some(3), Some(0.25)),
            (Some(1), Some(3), Some(0.25)),
            (None, Some(3), Some(0.25)),
            (Some(0), None, Some(0.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("n"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.421875);
        assert_eq_float!(res_col.value(1), 0.421875);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn binomial_pmf_failure_1() {
        let pmf = pmf();

        let recs = make_records(vec![(Some(0), Some(3), Some(1.25))]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("n"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await;
        match res {
            Err(DataFusionError::External(e)) => {
                let be = e.downcast::<BinomialError>().unwrap();
                assert_eq!(*be.as_ref(), BinomialError::ProbabilityInvalid);
            }
            _ => {
                println!("unexpected result: {:?}", res);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn binomial_ln_pdf_success() {
        let mut ctx = SessionContext::new();
        register(&mut ctx).unwrap();
        let res = ctx
            .sql("SELECT binomial_ln_pmf(CAST(2 AS BIGINT UNSIGNED), CAST(10 AS BIGINT UNSIGNED), 0.5)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 1);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), -3.1248093158291335);
    }

    #[tokio::test]
    async fn binomial_cdf_success() {
        let pmf = cdf();

        let recs = make_records(vec![
            (Some(0), Some(3), Some(0.25)),
            (Some(1), Some(3), Some(0.25)),
            (None, Some(3), Some(0.25)),
            (Some(0), None, Some(0.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("n"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.421875, 3e-15);
        assert_eq_float!(res_col.value(1), 0.84375);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn binomial_sf_success() {
        let pmf = sf();

        let recs = make_records(vec![
            (Some(0), Some(3), Some(0.25)),
            (Some(1), Some(3), Some(0.25)),
            (None, Some(3), Some(0.25)),
            (Some(0), None, Some(0.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("n"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.578125, 3e-15);
        assert_eq_float!(res_col.value(1), 0.15625, 4e-15);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
