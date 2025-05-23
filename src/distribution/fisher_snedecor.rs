//! Module containing functions to the Fisher-Snedecor (aka F) Distribution.
//! 
//! Implemented by [`statrs::distribution::FisherSnedecor`].
//! 
//! The [Fisher-Snedecor Distribution](https://en.wikipedia.org/wiki/F-distribution) has two
//! parameters:
//! 
//! d1: 0 < d1
//! d2: 0 < d2
//! 
//! Usage:
//! 
//! `fisher_snedecor_pdf(x, d1, d2)`  
//! `fisher_snedecor_log_pdf(x, d1, d2)`  
//! `fisher_snedecor_cdf(x, d1, d2)`  
//! `fisher_snedecor_sf(x, d1, d2)`
//! 
//! with
//! 
//!   `x`: [0, +∞) `Float64`/`DOUBLE`,  
//!   `d1`: (0, +∞) `Float64`/`DOUBLE`,  
//!   `d2`: (0, +∞) `Float64`/`DOUBLE`
//! 
//! Examples
//! ```
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() -> std::io::Result<()> {
//!     let mut ctx = datafusion::prelude::SessionContext::new();
//!     datafusion_statrs::distribution::fisher_snedecor::register(&mut ctx)?;
//!     ctx.sql("SELECT fisher_snedecor_cdf(1.0, 2.0, 3.0)").await?
//!        .show().await?;
//!     Ok(())
//! }
//! ```

use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::FisherSnedecor;

use crate::utils::continuous3f::Continuous3F;
use crate::utils::evaluator3f::{CdfEvaluator3F, LnPdfEvaluator3F, PdfEvaluator3F, SfEvaluator3F};

type Pdf = Continuous3F<PdfEvaluator3F<FisherSnedecor>>;

/// ScalarUDF for the Fisher-Snedecor Distribution PDF
pub fn pdf() -> ScalarUDF {
    ScalarUDF::from(Pdf::new("fisher_snedecor_pdf"))
}

type LnPdf = Continuous3F<LnPdfEvaluator3F<FisherSnedecor>>;

/// ScalarUDF for the Fisher-Snedecor Distribution log PDF
pub fn ln_pdf() -> ScalarUDF {
    ScalarUDF::from(LnPdf::new("fisher_snedecor_ln_pdf"))
}

type Cdf = Continuous3F<CdfEvaluator3F<FisherSnedecor>>;

/// ScalarUDF for the Fisher-Snedecor Distribution CDF
pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("fisher_snedecor_cdf"))
}

type Sf = Continuous3F<SfEvaluator3F<FisherSnedecor>>;

/// ScalarUDF for the Fisher-Snedecor Distribution SF
pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("fisher_snedecor_sf"))
}

/// Register the functions for the Fisher-Snedecor Distribution
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
    use statrs::distribution::FisherSnedecorError;

    use super::*;

    fn get_schema() -> SchemaRef {
        SchemaRef::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("p1", DataType::Float64, true),
            Field::new("p2", DataType::Float64, true),
        ]))
    }

    fn make_records(rows: Vec<(Option<f64>, Option<f64>, Option<f64>)>) -> RecordBatch {
        let mut xs = Vec::new();
        let mut p1s = Vec::new();
        let mut p2s = Vec::new();
        for row in rows {
            xs.push(row.0);
            p1s.push(row.1);
            p2s.push(row.2);
        }

        RecordBatch::try_new(
            get_schema(),
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(p1s)),
                Arc::new(Float64Array::from(p2s)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn fisher_snedecor_pdf_success() {
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
                (pdf.call(vec![col("x"), col("p1"), col("p2")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.0);
        assert_eq_float!(res_col.value(1), 0.08642373027968221);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn fisher_snedecor_pdf_failure_1() {
        let pdf = pdf();

        let recs = make_records(vec![(Some(0.0), Some(3.0), Some(-1.25))]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("p1"), col("p2")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await;
        match res {
            Err(DataFusionError::External(e)) => {
                let be = e.downcast::<FisherSnedecorError>().unwrap();
                assert_eq!(*be.as_ref(), FisherSnedecorError::Freedom2Invalid);
            }
            _ => {
                println!("unexpected result: {:?}", res);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn fisher_snedecor_ln_pdf_success() {
        let mut ctx = SessionContext::new();
        register(&mut ctx).unwrap();
        let res = ctx
            .sql("SELECT fisher_snedecor_ln_pdf(0.2, 8.0, 11.0)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 1);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), -1.452483318837048);
    }

    #[tokio::test]
    async fn fisher_snedecor_cdf_success() {
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
                (pdf.call(vec![col("x"), col("p1"), col("p2")])).alias("q"),
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
        assert_eq_float!(res_col.value(1), 0.22377660964255752);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn fisher_snedecor_sf_success() {
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
                (pdf.call(vec![col("x"), col("p1"), col("p2")])).alias("q"),
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
        assert_eq_float!(res_col.value(1), 0.7762233903574425);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
