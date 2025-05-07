//! Module containing functions to the Chi-squared Distribution.
//! 
//! Implemented by [`statrs::distribution::ChiSquared`].
//! 
//! The [Chi-squared Distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution) has one
//! parameter:
//! 
//! k: 0 ≤ k
//! 
//! NB the chi-squared is usually parameterised with k ∈ N, but as the underlying implementation is
//! built on the continuous Gamma distribution, the implementation in [`statrs::distribution::ChiSquared`]
//! is generalised to allow k to be continuous.
//! 
//! Usage:
//! 
//! `chi_squared_pdf(x, k)`  
//! `chi_squared_cdf(x, k)`  
//! `chi_squared_sf(x, k)`
//! 
//! with
//! 
//!   `x`: [0, +∞) `Float64`/`DOUBLE`,  
//!   `k`: (0, +∞) `Float64`/`DOUBLE`
//! 
//! Examples
//! ```
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() -> std::io::Result<()> {
//!     let mut ctx = datafusion::prelude::SessionContext::new();
//!     datafusion_statrs::distribution::chi_squared::register(&mut ctx)?;
//!     ctx.sql("SELECT chi_squared_pdf(1.25, 4.0)").await?
//!        .show().await?;
//!     Ok(())
//! }
//! ```

use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::ChiSquared;

use crate::utils::continuous2f::Continuous2F;
use crate::utils::evaluator2f::{CdfEvaluator2F, PdfEvaluator2F, SfEvaluator2F};

type Pdf = Continuous2F<PdfEvaluator2F<ChiSquared>>;

/// ScalarUDF for the Chi-squared Distribution PDF
pub fn pdf() -> ScalarUDF {
    ScalarUDF::from(Pdf::new("chi_squared_pdf"))
}

type Cdf = Continuous2F<CdfEvaluator2F<ChiSquared>>;

/// ScalarUDF for the Chi-squared Distribution CDF
pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("chi_squared_cdf"))
}

type Sf = Continuous2F<SfEvaluator2F<ChiSquared>>;

/// ScalarUDF for the Chi-squared Distribution SF
pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("chi_squared_sf"))
}

/// Register the functions for the Chi-squared Distribution
pub fn register(registry: &mut dyn FunctionRegistry) -> Result<(), DataFusionError> {
    crate::utils::register::register(registry, vec![pdf(), cdf(), sf()])
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::{
        arrow::{
            array::{Float64Array, RecordBatch},
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
            Field::new("n", DataType::Float64, true),
        ]))
    }

    fn make_records(rows: Vec<(Option<f64>, Option<f64>)>) -> RecordBatch {
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
                Arc::new(Float64Array::from(ns)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn chi_squared_pdf_success() {
        let pdf = pdf();

        let recs = make_records(vec![
            (Some(1.0), Some(3.0)),
            (Some(1.0), Some(5.0)),
            (None, Some(3.0)),
            (Some(1.0), None),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("n")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.24197072451914295);
        assert_eq!(res_col.value(1), 0.08065690817304756);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn chi_squared_pdf_failure_1() {
        let pdf = pdf();

        let recs = make_records(vec![(Some(0.1), Some(-1.25))]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("n")])).alias("q"),
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
    async fn chi_squared_cdf_success() {
        let pdf = cdf();

        let recs = make_records(vec![
            (Some(1.0), Some(3.0)),
            (Some(1.0), Some(5.0)),
            (None, Some(3.0)),
            (Some(1.0), None),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("n")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.1987480430987988);
        assert_eq!(res_col.value(1), 0.037434226752703484);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn chi_squared_sf_success() {
        let pdf = sf();

        let recs = make_records(vec![
            (Some(1.0), Some(3.0)),
            (Some(1.0), Some(5.0)),
            (None, Some(3.0)),
            (Some(1.0), None),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("n")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.8012519569012012);
        assert_eq!(res_col.value(1), 0.9625657732472965);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
