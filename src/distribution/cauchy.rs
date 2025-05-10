//! Module containing functions to the Cauchy Distribution.
//! 
//! Implemented by [`statrs::distribution::Cauchy`].
//! 
//! The [Cauchy Distribution](https://en.wikipedia.org/wiki/Cauchy_distribution) has two
//! parameters:
//! 
//! x0: x0 ∈ R (real numbers)  
//! γ: 0 < γ
//! 
//! Usage:
//! 
//! `cauchy_pdf(x, x0, γ)`  
//! `cauchy_ln_pdf(x, x0, γ)`  
//! `cauchy_cdf(x, x0, γ)`  
//! `cauchy_sf(x, x0, γ)`
//! 
//! with
//! 
//!   `x`: (-∞, +∞) `Float64`/`DOUBLE`,  
//!   `x0`: (-∞, +∞) `Float64`/`DOUBLE`,  
//!   `γ`: (0, +∞) `Float64`/`DOUBLE`
//! 
//! Examples
//! ```
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() -> std::io::Result<()> {
//!     let mut ctx = datafusion::prelude::SessionContext::new();
//!     datafusion_statrs::distribution::cauchy::register(&mut ctx)?;
//!     ctx.sql("SELECT cauchy_cdf(-1.0, 2.0, 3.5)").await?
//!        .show().await?;
//!     Ok(())
//! }
//! ```

use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::Cauchy;

use crate::utils::continuous3f::Continuous3F;
use crate::utils::evaluator3f::{CdfEvaluator3F, LnPdfEvaluator3F, PdfEvaluator3F, SfEvaluator3F};

type Pdf = Continuous3F<PdfEvaluator3F<Cauchy>>;

/// ScalarUDF for the Cauchy Distribution PDF
pub fn pdf() -> ScalarUDF {
    ScalarUDF::from(Pdf::new("cauchy_pdf"))
}

type LnPdf = Continuous3F<LnPdfEvaluator3F<Cauchy>>;

/// ScalarUDF for the Cauchy Distribution log PDF
pub fn ln_pdf() -> ScalarUDF {
    ScalarUDF::from(LnPdf::new("cauchy_ln_pdf"))
}

type Cdf = Continuous3F<CdfEvaluator3F<Cauchy>>;

/// ScalarUDF for the Cauchy Distribution CDF
pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("cauchy_cdf"))
}

type Sf = Continuous3F<SfEvaluator3F<Cauchy>>;

/// ScalarUDF for the Cauchy Distribution SF
pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("cauchy_sf"))
}

/// Register the functions for the Cauchy Distribution
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
    use statrs::distribution::CauchyError;

    use super::*;

    fn get_schema() -> SchemaRef {
        SchemaRef::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("x0", DataType::Float64, true),
            Field::new("p", DataType::Float64, true),
        ]))
    }

    fn make_records(rows: Vec<(Option<f64>, Option<f64>, Option<f64>)>) -> RecordBatch {
        let mut xs = Vec::new();
        let mut x0s = Vec::new();
        let mut ps = Vec::new();
        for row in rows {
            xs.push(row.0);
            x0s.push(row.1);
            ps.push(row.2);
        }

        RecordBatch::try_new(
            get_schema(),
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(x0s)),
                Arc::new(Float64Array::from(ps)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn cauchy_pdf_success() {
        let pdf = pdf();

        let recs = make_records(vec![
            (Some(-1.0), Some(0.0), Some(0.5)),
            (Some(1.0), Some(0.0), Some(0.5)),
            (None, Some(3.0), Some(-1.25)),
            (Some(0.0), None, Some(5.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("x0"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.12732395447351627);
        assert_eq_float!(res_col.value(1), 0.12732395447351627);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn cauchy_pdf_failure_1() {
        let pdf = pdf();

        let recs = make_records(vec![(Some(0.0), Some(3.0), Some(-1.25))]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("x0"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await;
        match res {
            Err(DataFusionError::External(e)) => {
                let be = e.downcast::<CauchyError>().unwrap();
                assert_eq!(*be.as_ref(), CauchyError::ScaleInvalid);
            }
            _ => {
                println!("unexpected result: {:?}", res);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn cauchy_ln_pdf_success() {
        let mut ctx = SessionContext::new();
        register(&mut ctx).unwrap();
        let res = ctx
            .sql("SELECT cauchy_ln_pdf(0.2, 8.0, 11.0)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 1);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), -3.9499617925953174);
    }

    #[tokio::test]
    async fn cauchy_cdf_success() {
        let pmf = cdf();

        let recs = make_records(vec![
            (Some(-1.0), Some(0.0), Some(0.5)),
            (Some(1.0), Some(0.0), Some(0.5)),
            (None, Some(3.0), Some(-1.25)),
            (Some(0.0), None, Some(5.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("x0"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.1475836176504332);
        assert_eq_float!(res_col.value(1), 0.8524163823495667);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn beta_sf_success() {
        let pmf = sf();

        let recs = make_records(vec![
            (Some(-1.0), Some(0.0), Some(0.5)),
            (Some(1.0), Some(0.0), Some(0.5)),
            (None, Some(3.0), Some(-1.25)),
            (Some(0.0), None, Some(5.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("x0"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.8524163823495667);
        assert_eq_float!(res_col.value(1), 0.1475836176504332);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
