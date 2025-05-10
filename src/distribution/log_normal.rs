//! Module containing functions to the log Normal Distribution.
//! 
//! Implemented by [`statrs::distribution::LogNormal`].
//! 
//! The [log Normal Distribution](https://en.wikipedia.org/wiki/Log-normal_distribution) has two
//! parameters:
//! 
//! μ: μ ∈ R (real numbers) 
//! σ: 0 < σ (rate)
//! 
//! Usage:
//! 
//! `log_normal_pdf(x, μ, σ)`  
//! `log_normal_ln_pdf(x, μ, σ)`  
//! `log_normal_cdf(x, μ, σ)`  
//! `log_normal_sf(x, μ, σ)`
//! 
//! with
//! 
//!   `x`: (-∞, +∞) `Float64`/`DOUBLE`,  
//!   `μ`: (-∞, +∞) `Float64`/`DOUBLE`,  
//!   `σ`: (0, +∞) `Float64`/`DOUBLE`
//! 
//! Examples
//! ```
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() -> std::io::Result<()> {
//!     let mut ctx = datafusion::prelude::SessionContext::new();
//!     datafusion_statrs::distribution::log_normal::register(&mut ctx)?;
//!     ctx.sql("SELECT log_normal_pdf(1.1, 9.0, 1.0)").await?
//!        .show().await?;
//!     Ok(())
//! }
//! ```

use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::LogNormal;

use super::super::utils::continuous3f::Continuous3F;
use super::super::utils::evaluator3f::{CdfEvaluator3F, LnPdfEvaluator3F, PdfEvaluator3F, SfEvaluator3F};

type Pdf = Continuous3F<PdfEvaluator3F<LogNormal>>;

/// ScalarUDF for the log-Normal PDF
pub fn pdf() -> ScalarUDF {
    ScalarUDF::from(Pdf::new("log_normal_pdf"))
}

type LnPdf = Continuous3F<LnPdfEvaluator3F<LogNormal>>;

/// ScalarUDF for the log-Normal log PDF
pub fn ln_pdf() -> ScalarUDF {
    ScalarUDF::from(LnPdf::new("log_normal_ln_pdf"))
}

type Cdf = Continuous3F<CdfEvaluator3F<LogNormal>>;

/// ScalarUDF for the log-Normal CDF
pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("log_normal_cdf"))
}

type Sf = Continuous3F<SfEvaluator3F<LogNormal>>;

/// ScalarUDF for the log-Normal SF
pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("log_normal_sf"))
}

/// Register the functions for the log-Normal Distribution
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
    use statrs::distribution::LogNormalError;

    use super::*;

    fn get_schema() -> SchemaRef {
        SchemaRef::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("s", DataType::Float64, true),
            Field::new("r", DataType::Float64, true),
        ]))
    }

    fn make_records(rows: Vec<(Option<f64>, Option<f64>, Option<f64>)>) -> RecordBatch {
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
                Arc::new(Float64Array::from(ss)),
                Arc::new(Float64Array::from(rs)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn log_normal_pdf_success() {
        let pdf = pdf();

        let recs = make_records(vec![
            (Some(1.0), Some(3.0), Some(0.25)),
            (Some(2.), Some(3.0), Some(0.25)),
            (None, Some(3.0), Some(0.25)),
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
        assert_eq_float!(res_col.value(0), 8.585534942652241e-32);
        assert_eq_float!(res_col.value(1), 2.587690993483153e-19);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn log_normal_pdf_failure_1() {
        let pdf = pdf();

        let recs = make_records(vec![(Some(1.0), Some(0.0), Some(-1.25))]);

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
                let be = e.downcast::<LogNormalError>().unwrap();
                assert_eq!(*be.as_ref(), LogNormalError::ScaleInvalid);
            }
            _ => {
                println!("unexpected result: {:?}", res);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn log_normal_ln_pdf_success() {
        let mut ctx = SessionContext::new();
        register(&mut ctx).unwrap();
        let res = ctx
            .sql("SELECT log_normal_ln_pdf(0.2, 8.0, 11.0)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 1);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), -2.0889715009777277);
    }

    #[tokio::test]
    async fn log_normal_cdf_success() {
        let pdf = cdf();

        let recs = make_records(vec![
            (Some(1.0), Some(3.0), Some(0.25)),
            (Some(2.), Some(3.0), Some(0.25)),
            (None, Some(3.0), Some(0.25)),
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
        assert_eq_float!(res_col.value(0), 1.7764821120334314e-33);
        assert_eq_float!(res_col.value(1), 1.3862566635680422e-20);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn log_normal_sf_success() {
        let pdf = sf();

        let recs = make_records(vec![
            (Some(1.0), Some(3.0), Some(0.25)),
            (Some(2.), Some(3.0), Some(0.25)),
            (None, Some(3.0), Some(0.25)),
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
        assert_eq_float!(res_col.value(0), 1.0);
        assert_eq_float!(res_col.value(1), 1.0);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
