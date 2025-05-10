//! Module containing functions to the Beta Distribution.
//! 
//! Implemented by [`statrs::distribution::Beta`].
//! 
//! The [Beta Distribution](https://en.wikipedia.org/wiki/Beta_distribution) has two
//! parameters:
//! 
//! α: 0 < α  
//! β: 0 < β
//! 
//! Usage:
//! 
//! `beta_pdf(x, α, β)`  
//! `beta_ln_pdf(x, α, β)`  
//! `beta_cdf(x, α, β)`  
//! `beta_sf(x, α, β)`
//! 
//! with
//! 
//!   `x`: [0, 1] `Float64`/`DOUBLE`,  
//!   `α`: (0, +∞) `Float64`/`DOUBLE`,  
//!   `β`: (0, +∞) `Float64`/`DOUBLE`
//! 
//! Examples
//! ```
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() -> std::io::Result<()> {
//!     let mut ctx = datafusion::prelude::SessionContext::new();
//!     datafusion_statrs::distribution::beta::register(&mut ctx)?;
//!     ctx.sql("SELECT beta_cdf(0.5, 5.0, 8.0)").await?
//!        .show().await?;
//!     Ok(())
//! }
//! ```

use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::Beta;

use crate::utils::continuous3f::Continuous3F;
use crate::utils::evaluator3f::{CdfEvaluator3F, LnPdfEvaluator3F, PdfEvaluator3F, SfEvaluator3F};

type Pdf = Continuous3F<PdfEvaluator3F<Beta>>;

/// ScalarUDF for the Beta Distribution PDF
pub fn pdf() -> ScalarUDF {
    ScalarUDF::from(Pdf::new("beta_pdf"))
}

type LnPdf = Continuous3F<LnPdfEvaluator3F<Beta>>;

/// ScalarUDF for the Beta Distribution log PDF
pub fn ln_pdf() -> ScalarUDF {
    ScalarUDF::from(LnPdf::new("beta_ln_pdf"))
}

type Cdf = Continuous3F<CdfEvaluator3F<Beta>>;

/// ScalarUDF for the Beta Distribution CDF
pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("beta_cdf"))
}

type Sf = Continuous3F<SfEvaluator3F<Beta>>;

/// ScalarUDF for the Beta Distribution SF
pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("beta_sf"))
}

/// Register the functions for the Beta Distribution
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
    use statrs::distribution::BetaError;

    use super::*;

    fn get_schema() -> SchemaRef {
        SchemaRef::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("a", DataType::Float64, true),
            Field::new("b", DataType::Float64, true),
        ]))
    }

    fn make_records(rows: Vec<(Option<f64>, Option<f64>, Option<f64>)>) -> RecordBatch {
        let mut xs = Vec::new();
        let mut as_ = Vec::new();
        let mut bs = Vec::new();
        for row in rows {
            xs.push(row.0);
            as_.push(row.1);
            bs.push(row.2);
        }

        RecordBatch::try_new(
            get_schema(),
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(as_)),
                Arc::new(Float64Array::from(bs)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn beta_pdf_success() {
        let pdf = pdf();

        let recs = make_records(vec![
            (Some(0.25), Some(2.0), Some(8.5)),
            (Some(0.75), Some(2.0), Some(8.5)),
            (None, Some(3.0), Some(1.25)),
            (Some(0.01), None, Some(5.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("a"), col("b")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 2.3336838198042265);
        assert_eq_float!(res_col.value(1), 0.0018482208251953966);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn beta_pdf_failure_1() {
        let pdf = pdf();

        let recs = make_records(vec![(Some(0.0), Some(3.0), Some(-1.25))]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("a"), col("b")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await;
        match res {
            Err(DataFusionError::External(e)) => {
                let be = e.downcast::<BetaError>().unwrap();
                assert_eq!(*be.as_ref(), BetaError::ShapeBInvalid);
            }
            _ => {
                println!("unexpected result: {:?}", res);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn beta_ln_pdf_success() {
        let mut ctx = SessionContext::new();
        register(&mut ctx).unwrap();
        let res = ctx
            .sql("SELECT beta_ln_pdf(0.2, 8.0, 11.0)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 1);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), -0.7316296262886262);
    }

    #[tokio::test]
    async fn beta_cdf_success() {
        let pmf = cdf();

        let recs = make_records(vec![
            (Some(0.25), Some(2.0), Some(8.5)),
            (Some(0.75), Some(2.0), Some(8.5)),
            (None, Some(3.0), Some(1.25)),
            (Some(0.01), None, Some(5.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("a"), col("b")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.7290614760289211);
        assert_eq_float!(res_col.value(1), 0.999943733215332);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn beta_sf_success() {
        let pmf = sf();

        let recs = make_records(vec![
            (Some(0.25), Some(2.0), Some(8.5)),
            (Some(0.75), Some(2.0), Some(8.5)),
            (None, Some(3.0), Some(1.25)),
            (Some(0.01), None, Some(5.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("a"), col("b")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.2709385239710789);
        assert_eq_float!(res_col.value(1), 5.626678466797133e-5);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
