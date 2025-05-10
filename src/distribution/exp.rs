//! Module containing functions to the Exponential Distribution.
//! 
//! Implemented by [`statrs::distribution::Exp`].
//! 
//! The [Exponential Distribution](https://en.wikipedia.org/wiki/Exponential_distribution) has one
//! parameter:
//! 
//! λ: 0 < λ
//! 
//! Usage:
//! 
//! `exp_pdf(x, λ)`  
//! `exp_ln_pdf(x, λ)`  
//! `exp_cdf(x, λ)`  
//! `exp_sf(x, λ)`
//! 
//! with
//! 
//!   `x`: [0, +∞) `Float64`/`DOUBLE`,  
//!   `λ`: (0, +∞) `Float64`/`DOUBLE`,
//! 
//! Examples
//! ```
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() -> std::io::Result<()> {
//!     let mut ctx = datafusion::prelude::SessionContext::new();
//!     datafusion_statrs::distribution::exp::register(&mut ctx)?;
//!     ctx.sql("SELECT exp_pdf(1.25, 1.0)").await?
//!        .show().await?;
//!     Ok(())
//! }
//! ```

use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::Exp;

use crate::utils::continuous2f::Continuous2F;
use crate::utils::evaluator2f::{CdfEvaluator2F, LnPdfEvaluator2F, PdfEvaluator2F, SfEvaluator2F};

type Pdf = Continuous2F<PdfEvaluator2F<Exp>>;

/// ScalarUDF for the Exponential Distribution PDF
pub fn pdf() -> ScalarUDF {
    ScalarUDF::from(Pdf::new("exp_pdf"))
}

type LnPdf = Continuous2F<LnPdfEvaluator2F<Exp>>;

/// ScalarUDF for the Exponential Distribution log PDF
pub fn ln_pdf() -> ScalarUDF {
    ScalarUDF::from(LnPdf::new("exp_ln_pdf"))
}

type Cdf = Continuous2F<CdfEvaluator2F<Exp>>;

/// ScalarUDF for the Exponential Distribution CDF
pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("exp_cdf"))
}

type Sf = Continuous2F<SfEvaluator2F<Exp>>;

/// ScalarUDF for the Exponential Distribution SF
pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("exp_sf"))
}

/// Register the functions for the Exponential Distribution
pub fn register(registry: &mut dyn FunctionRegistry) -> Result<(), DataFusionError> {
    crate::utils::register::register(registry, vec![pdf(), ln_pdf(), cdf(), sf()])
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use assert_eq_float::assert_eq_float;
    use datafusion::{
        arrow::{
            array::{Float64Array, RecordBatch},
            datatypes::{DataType, Field, Schema, SchemaRef},
        },
        common::cast::as_float64_array,
        error::DataFusionError,
        prelude::{SessionContext, col},
    };
    use statrs::distribution::ExpError;

    use super::*;

    fn get_schema() -> SchemaRef {
        SchemaRef::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("p", DataType::Float64, true),
        ]))
    }

    fn make_records(rows: Vec<(Option<f64>, Option<f64>)>) -> RecordBatch {
        let mut xs = Vec::new();
        let mut ps = Vec::new();
        for row in rows {
            xs.push(row.0);
            ps.push(row.1);
        }

        RecordBatch::try_new(
            get_schema(),
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ps)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn exp_pdf_success() {
        let pdf = pdf();

        let recs = make_records(vec![
            (Some(1.0), Some(0.25)),
            (Some(2.0), Some(0.25)),
            (None, Some(0.25)),
            (Some(1.0), None),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.19470019576785122);
        assert_eq_float!(res_col.value(1), 0.15163266492815836);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn exp_pdf_failure_1() {
        let pdf = pdf();

        let recs = make_records(vec![(Some(1.0), Some(-1.25))]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await;
        match res {
            Err(DataFusionError::External(e)) => {
                let be = e.downcast::<ExpError>().unwrap();
                assert_eq!(*be.as_ref(), ExpError::RateInvalid);
            }
            _ => {
                println!("unexpected result: {:?}", res);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn exp_ln_pdf_success() {
        let mut ctx = SessionContext::new();
        register(&mut ctx).unwrap();
        let res = ctx
            .sql("SELECT exp_ln_pdf(0.2, 2.0)")
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
    async fn exp_cdf_success() {
        let pdf = cdf();

        let recs = make_records(vec![
            (Some(1.0), Some(0.25)),
            (Some(2.0), Some(0.25)),
            (None, Some(0.25)),
            (Some(1.0), None),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.22119921692859512);
        assert_eq_float!(res_col.value(1), 0.3934693402873666);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn exp_sf_success() {
        let pdf = sf();

        let recs = make_records(vec![
            (Some(1.0), Some(0.25)),
            (Some(2.0), Some(0.25)),
            (None, Some(0.25)),
            (Some(1.0), None),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.7788007830714049);
        assert_eq_float!(res_col.value(1), 0.6065306597126334);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
