//! Module containing functions to the Gumbel Distribution.
//! 
//! Implemented by [`statrs::distribution::Gumbel`].
//! 
//! The [Gumbel Distribution](https://en.wikipedia.org/wiki/Gumbel_distribution) has two
//! parameters:
//! 
//! μ: μ ∈ R (real numbers)
//! β: 0 < β
//! 
//! Usage:
//! 
//! `gumbel_pdf(x, μ, β)`  
//! `gumbel_ln_pdf(x, μ, β)`  
//! `gumbel_cdf(x, μ, β)`  
//! `gumbel_sf(x, μ, β)`
//! 
//! with
//! 
//!   `x`: (-∞, +∞) `Float64`/`DOUBLE`,  
//!   `μ`: (-∞, +∞) `Float64`/`DOUBLE`,  
//!   `β`: (0, +∞) `Float64`/`DOUBLE`,
//! 
//! Examples
//! ```
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() -> std::io::Result<()> {
//!     let mut ctx = datafusion::prelude::SessionContext::new();
//!     datafusion_statrs::distribution::gumbel::register(&mut ctx)?;
//!     ctx.sql("SELECT gumbel_pdf(1.25, 1.5, 3.0)").await?
//!        .show().await?;
//!     Ok(())
//! }
//! ```

use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::Gumbel;

use super::super::utils::continuous3f::Continuous3F;
use super::super::utils::evaluator3f::{CdfEvaluator3F, LnPdfEvaluator3F, PdfEvaluator3F, SfEvaluator3F};

type Pdf = Continuous3F<PdfEvaluator3F<Gumbel>>;

/// ScalarUDF for the Gumbel Distribution PDF
pub fn pdf() -> ScalarUDF {
    ScalarUDF::from(Pdf::new("gumbel_pdf"))
}

type LnPdf = Continuous3F<LnPdfEvaluator3F<Gumbel>>;

/// ScalarUDF for the Gumbel Distribution log PDF
pub fn ln_pdf() -> ScalarUDF {
    ScalarUDF::from(LnPdf::new("gumbel_ln_pdf"))
}

type Cdf = Continuous3F<CdfEvaluator3F<Gumbel>>;

/// ScalarUDF for the Gumbel Distribution PDF
pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("gumbel_cdf"))
}

type Sf = Continuous3F<SfEvaluator3F<Gumbel>>;

/// ScalarUDF for the Gumbel Distribution PDF
pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("gumbel_sf"))
}

/// Register the functions for the Gumbel Distribution
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
    use statrs::distribution::GumbelError;

    use super::*;

    fn get_schema() -> SchemaRef {
        SchemaRef::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("m", DataType::Float64, true),
            Field::new("b", DataType::Float64, true),
        ]))
    }

    fn make_records(rows: Vec<(Option<f64>, Option<f64>, Option<f64>)>) -> RecordBatch {
        let mut xs = Vec::new();
        let mut ms = Vec::new();
        let mut bs = Vec::new();
        for row in rows {
            xs.push(row.0);
            ms.push(row.1);
            bs.push(row.2);
        }

        RecordBatch::try_new(
            get_schema(),
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ms)),
                Arc::new(Float64Array::from(bs)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn gumbel_pdf_success() {
        let pdf = pdf();

        let recs = make_records(vec![
            (Some(0.0), Some(3.0), Some(0.25)),
            (Some(1.0), Some(3.0), Some(0.25)),
            (None, Some(3.0), Some(0.25)),
            (Some(0.0), None, Some(0.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("m"), col("b")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.0);
        assert_eq_float!(res_col.value(1), 0.0);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn gumbel_pdf_failure_1() {
        let pdf = pdf();

        let recs = make_records(vec![(Some(0.0), Some(3.0), Some(-1.25))]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("m"), col("b")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await;
        match res {
            Err(DataFusionError::External(e)) => {
                let be = e.downcast::<GumbelError>().unwrap();
                assert_eq!(*be.as_ref(), GumbelError::ScaleInvalid);
            }
            _ => {
                println!("unexpected result: {:?}", res);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn gumbel_ln_pdf_success() {
        let mut ctx = SessionContext::new();
        register(&mut ctx).unwrap();
        let res = ctx
            .sql("SELECT gumbel_ln_pdf(0.2, 8.0, 11.0)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 1);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), -3.7209473796294246);
    }

    #[tokio::test]
    async fn gumbel_cdf_success() {
        let pdf = cdf();

        let recs = make_records(vec![
            (Some(0.0), Some(3.0), Some(0.25)),
            (Some(1.0), Some(3.0), Some(0.25)),
            (None, Some(3.0), Some(0.25)),
            (Some(0.0), None, Some(0.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("m"), col("b")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.0);
        assert_eq_float!(res_col.value(1), 0.0);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn gumbel_sf_success() {
        let pdf = sf();

        let recs = make_records(vec![
            (Some(0.0), Some(3.0), Some(0.25)),
            (Some(1.0), Some(3.0), Some(0.25)),
            (None, Some(3.0), Some(0.25)),
            (Some(0.0), None, Some(0.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("m"), col("b")])).alias("q"),
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
