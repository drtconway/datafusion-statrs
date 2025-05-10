//! Module containing functions to the Chi Distribution.
//! 
//! Implemented by [`statrs::distribution::Chi`].
//! 
//! The [Chi Distribution](https://en.wikipedia.org/wiki/Chi_distribution) has one
//! parameter:
//! 
//! k: k ∈ N (natural numbers)
//! 
//! Usage:
//! 
//! `chi_pdf(x, k)`  
//! `chi_cdf(x, k)`  
//! `chi_sf(x, k)`
//! 
//! with
//! 
//!   `x`: [0, +∞) `Float64`/`DOUBLE`,  
//!   `k`: (0, +∞) `UInt64`/`BIGINT UNSIGNED`,
//! 
//! Examples
//! ```
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() -> std::io::Result<()> {
//!     let mut ctx = datafusion::prelude::SessionContext::new();
//!     datafusion_statrs::distribution::chi::register(&mut ctx)?;
//!     ctx.sql("SELECT chi_pdf(1.25, CAST(4 AS BIGINT UNSIGNED))").await?
//!        .show().await?;
//!     Ok(())
//! }
//! ```

use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::Chi;

use crate::utils::continuous1f1u::Continuous1F1U;
use crate::utils::evaluator1f1u::{CdfEvaluator1F1U, LnPdfEvaluator1F1U, PdfEvaluator1F1U, SfEvaluator1F1U};

type Pdf = Continuous1F1U<PdfEvaluator1F1U<Chi>>;

/// ScalarUDF for the Chi Distribution PDF
pub fn pdf() -> ScalarUDF {
    ScalarUDF::from(Pdf::new("chi_pdf"))
}

type LnPdf = Continuous1F1U<LnPdfEvaluator1F1U<Chi>>;

/// ScalarUDF for the Chi Distribution log PDF
pub fn ln_pdf() -> ScalarUDF {
    ScalarUDF::from(LnPdf::new("chi_ln_pdf"))
}

type Cdf = Continuous1F1U<CdfEvaluator1F1U<Chi>>;

/// ScalarUDF for the Chi Distribution CDF
pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("chi_cdf"))
}

type Sf = Continuous1F1U<SfEvaluator1F1U<Chi>>;

/// ScalarUDF for the Chi Distribution SF
pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("chi_sf"))
}

/// Register the functions for the Binomial Distribution
pub fn register(registry: &mut dyn FunctionRegistry) -> Result<(), DataFusionError> {
    crate::utils::register::register(registry, vec![pdf(), ln_pdf(), cdf(), sf()])
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
    use statrs::distribution::ChiError;

    use super::*;

    fn get_schema() -> SchemaRef {
        SchemaRef::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("n", DataType::UInt64, true),
        ]))
    }

    fn make_records(rows: Vec<(Option<f64>, Option<u64>)>) -> RecordBatch {
        let mut xs = Vec::new();
        let mut ns = Vec::new();
        for row in rows {
            xs.push(row.0);
            ns.push(row.1);
        }

        RecordBatch::try_new(
            get_schema(),
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(UInt64Array::from(ns)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn chi_pdf_success() {
        let pmf = pdf();

        let recs = make_records(vec![
            (Some(0.5), Some(3)),
            (Some(1.5), Some(3)),
            (None, Some(3)),
            (Some(0.5), None),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("n")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.17603266338214946);
        assert_eq_float!(res_col.value(1), 0.5828291804965118);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn chi_pdf_failure_1() {
        let pmf = pdf();

        let recs = make_records(vec![(Some(0.5), Some(0))]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("n")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await;
        match res {
            Err(DataFusionError::External(e)) => {
                let be = e.downcast::<ChiError>().unwrap();
                assert_eq!(*be.as_ref(), ChiError::FreedomInvalid);
            }
            _ => {
                println!("unexpected result: {:?}", res);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn chi_ln_pdf_success() {
        let mut ctx = SessionContext::new();
        register(&mut ctx).unwrap();
        let res = ctx
            .sql("SELECT chi_ln_pdf(0.2, CAST(2 AS BIGINT UNSIGNED))")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 1);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), -1.6294379124340999);
    }

    #[tokio::test]
    async fn chi_cdf_success() {
        let pmf = cdf();

        let recs = make_records(vec![
            (Some(0.5), Some(3)),
            (Some(1.5), Some(3)),
            (None, Some(3)),
            (Some(0.5), None),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("n")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.030859595783726688);
        assert_eq_float!(res_col.value(1), 0.4778328104646076);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn chi_sf_success() {
        let pmf = sf();

        let recs = make_records(vec![
            (Some(0.5), Some(3)),
            (Some(1.5), Some(3)),
            (None, Some(3)),
            (Some(0.5), None),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("n")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.9691404042162733);
        assert_eq_float!(res_col.value(1), 0.5221671895353923);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
