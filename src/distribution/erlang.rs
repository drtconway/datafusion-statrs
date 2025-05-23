//! Module containing functions to the Erlang Distribution.
//! 
//! Implemented by [`statrs::distribution::Erlang`].
//! 
//! The [Erlang Distribution](https://en.wikipedia.org/wiki/Erlang_distribution) has two
//! parameters:
//! 
//! k: k ∈ N (natural numbers)  
//! λ: 0 < λ
//! 
//! Usage:
//! 
//! `erlang_pdf(x, k, λ)`  
//! `erlang_ln_pdf(x, k, λ)`  
//! `erlang_cdf(x, k, λ)`  
//! `erlang_sf(x, k, λ)`
//! 
//! with
//! 
//!   `x`: [0, +∞) `Float64`/`DOUBLE`,  
//!   `k`: (-∞, +∞) `UInt64`/`BIGINT UNSIGNED`,  
//!   `λ`: (0, +∞) `Float64`/`DOUBLE`
//! 
//! Examples
//! ```
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() -> std::io::Result<()> {
//!     let mut ctx = datafusion::prelude::SessionContext::new();
//!     datafusion_statrs::distribution::erlang::register(&mut ctx)?;
//!     ctx.sql("SELECT erlang_cdf(1.0, CAST(5 AS BIGINT UNSIGNED), 2.0)").await?
//!        .show().await?;
//!     Ok(())
//! }
//! ```

use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::Erlang;

use crate::utils::continuous1f1u1f::Continuous1F1U1F;
use crate::utils::evaluator1f1u1f::{CdfEvaluator1F1U1F, LnPdfEvaluator1F1U1F, PdfEvaluator1F1U1F, SfEvaluator1F1U1F};

type Pdf = Continuous1F1U1F<PdfEvaluator1F1U1F<Erlang>>;

/// ScalarUDF for the Erlang Distribution PDF
pub fn pdf() -> ScalarUDF {
    ScalarUDF::from(Pdf::new("erlang_pdf"))
}

type LnPdf = Continuous1F1U1F<LnPdfEvaluator1F1U1F<Erlang>>;

/// ScalarUDF for the Erlang Distribution PDF
pub fn ln_pdf() -> ScalarUDF {
    ScalarUDF::from(LnPdf::new("erlang_ln_pdf"))
}

type Cdf = Continuous1F1U1F<CdfEvaluator1F1U1F<Erlang>>;

/// ScalarUDF for the Erlang Distribution CDF
pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("erlang_cdf"))
}

type Sf = Continuous1F1U1F<SfEvaluator1F1U1F<Erlang>>;

/// ScalarUDF for the Erlang Distribution SF
pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("erlang_sf"))
}

/// Register the functions for the Erlang Distribution
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
    use statrs::distribution::GammaError;

    use super::*;

    fn get_schema() -> SchemaRef {
        SchemaRef::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("s", DataType::UInt64, true),
            Field::new("r", DataType::Float64, true),
        ]))
    }

    fn make_records(rows: Vec<(Option<f64>, Option<u64>, Option<f64>)>) -> RecordBatch {
        let mut xs = Vec::new();
        let mut ss = Vec::new();
        let mut rs = Vec::new();
        for row in rows {
            xs.push(row.0);
            ss.push(row.1);
            rs.push(row.2);
        }

        RecordBatch::try_new(
            get_schema(),
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(UInt64Array::from(ss)),
                Arc::new(Float64Array::from(rs)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn erlang_pdf_success() {
        let pdf = pdf();

        let recs = make_records(vec![
            (Some(1.0), Some(3), Some(0.25)),
            (Some(2.), Some(3), Some(0.25)),
            (None, Some(3), Some(0.25)),
            (Some(1.0), None, Some(0.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("s"), col("r")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.006084381117745331);
        assert_eq_float!(res_col.value(1), 0.018954083116019732);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn erlang_pdf_failure_1() {
        let pdf = pdf();

        let recs = make_records(vec![(Some(1.0), Some(0), Some(-1.25))]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("s"), col("r")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await;
        match res {
            Err(DataFusionError::External(e)) => {
                let be = e.downcast::<GammaError>().unwrap();
                assert_eq!(*be.as_ref(), GammaError::ShapeInvalid);
            }
            _ => {
                println!("unexpected result: {:?}", res);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn erlang_ln_pdf_success() {
        let mut ctx = SessionContext::new();
        register(&mut ctx).unwrap();
        let res = ctx
            .sql("SELECT erlang_ln_pdf(0.2, CAST(1 AS BIGINT UNSIGNED), 2.0)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 1);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.29314718055994526);
    }

    #[tokio::test]
    async fn erlang_cdf_success() {
        let pdf = cdf();

        let recs = make_records(vec![
            (Some(1.0), Some(3), Some(0.25)),
            (Some(2.), Some(3), Some(0.25)),
            (None, Some(3), Some(0.25)),
            (Some(1.0), None, Some(0.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("s"), col("r")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.0021614966897625073);
        assert_eq_float!(res_col.value(1), 0.014387677966970639);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn erlang_sf_success() {
        let pdf = sf();

        let recs = make_records(vec![
            (Some(1.0), Some(3), Some(0.25)),
            (Some(2.), Some(3), Some(0.25)),
            (None, Some(3), Some(0.25)),
            (Some(1.0), None, Some(0.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("s"), col("r")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.9978385033102375);
        assert_eq_float!(res_col.value(1), 0.9856123220330294);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
