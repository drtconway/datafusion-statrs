//! Module containing functions to the Student's T Distribution.
//! 
//! Implemented by [`statrs::distribution::StudentsT`].
//! 
//! The [Student's T Distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution) has three
//! parameters:
//! 
//! μ: μ ∈ R (real numbers) 
//! σ: 0 < σ  
//! ν: 0 < ν
//! 
//! Usage:
//! 
//! `students_t_pdf(x, μ, σ, ν)`  
//! `students_t_ln_pdf(x, μ, σ, ν)`  
//! `students_t_cdf(x, μ, σ, ν)`  
//! `students_t_sf(x, μ, σ, ν)`
//! 
//! with
//! 
//!   `x`: (-∞, +∞) `Float64`/`DOUBLE`,  
//!   `μ`: (-∞, +∞) `Float64`/`DOUBLE`,  
//!   `σ`: (0, +∞) `Float64`/`DOUBLE`,  
//!   `ν`: (0, +∞) `Float64`/`DOUBLE`
//! 
//! Examples
//! ```
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() -> std::io::Result<()> {
//!     let mut ctx = datafusion::prelude::SessionContext::new();
//!     datafusion_statrs::distribution::students_t::register(&mut ctx)?;
//!     ctx.sql("SELECT students_t_cdf(1.0, 0.0, 1.0, 2.0)").await?
//!        .show().await?;
//!     Ok(())
//! }
//! ```

use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::StudentsT;

use crate::utils::continuous4f::Continuous4F;
use crate::utils::evaluator4f::{CdfEvaluator4F, LnPdfEvaluator4F, PdfEvaluator4F, SfEvaluator4F};

type Pdf = Continuous4F<PdfEvaluator4F<StudentsT>>;

/// ScalarUDF for the Student's T PDF
pub fn pdf() -> ScalarUDF {
    ScalarUDF::from(Pdf::new("students_t_pdf"))
}

type LnPdf = Continuous4F<LnPdfEvaluator4F<StudentsT>>;

/// ScalarUDF for the Student's T log PDF
pub fn ln_pdf() -> ScalarUDF {
    ScalarUDF::from(LnPdf::new("students_t_ln_pdf"))
}

type Cdf = Continuous4F<CdfEvaluator4F<StudentsT>>;

/// ScalarUDF for the Student's T PDF
pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("students_t_cdf"))
}

type Sf = Continuous4F<SfEvaluator4F<StudentsT>>;

/// ScalarUDF for the Student's T PDF
pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("students_t_sf"))
}

/// Register the functions for the Student's T Distribution
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
    use statrs::distribution::StudentsTError;

    use super::*;

    fn get_schema() -> SchemaRef {
        SchemaRef::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("l", DataType::Float64, true),
            Field::new("s", DataType::Float64, true),
            Field::new("d", DataType::Float64, true),
        ]))
    }

    fn make_records(
        rows: Vec<(Option<f64>, Option<f64>, Option<f64>, Option<f64>)>,
    ) -> RecordBatch {
        let mut xs = Vec::new();
        let mut ls = Vec::new();
        let mut ss = Vec::new();
        let mut ds = Vec::new();
        for row in rows {
            xs.push(row.0);
            ls.push(row.1);
            ss.push(row.2);
            ds.push(row.3);
        }

        RecordBatch::try_new(
            get_schema(),
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ls)),
                Arc::new(Float64Array::from(ss)),
                Arc::new(Float64Array::from(ds)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn students_t_pdf_success() {
        let pdf = pdf();

        let recs = make_records(vec![
            (Some(1.0), Some(3.0), Some(0.25), Some(1.0)),
            (Some(2.), Some(3.0), Some(0.25), Some(1.0)),
            (None, Some(3.0), Some(0.25), Some(1.0)),
            (Some(1.0), None, Some(0.25), Some(1.0)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("l"), col("s"), col("d")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.019588300688233265);
        assert_eq_float!(res_col.value(1), 0.07489644380795071);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn students_t_pdf_failure_1() {
        let pdf = pdf();

        let recs = make_records(vec![(Some(1.0), Some(0.0), Some(1.0), Some(-1.25))]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("l"), col("s"), col("d")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await;
        match res {
            Err(DataFusionError::External(e)) => {
                let be = e.downcast::<StudentsTError>().unwrap();
                assert_eq!(*be.as_ref(), StudentsTError::FreedomInvalid);
            }
            _ => {
                println!("unexpected result: {:?}", res);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn students_t_ln_pdf_success() {
        let mut ctx = SessionContext::new();
        register(&mut ctx).unwrap();
        let res = ctx
            .sql("SELECT students_t_ln_pdf(11.2, 1.0, 2.0, 3.0)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 1);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), -6.232092649113863);
    }

    #[tokio::test]
    async fn students_t_cdf_success() {
        let pdf = cdf();

        let recs = make_records(vec![
            (Some(1.0), Some(3.0), Some(0.25), Some(1.0)),
            (Some(2.), Some(3.0), Some(0.25), Some(1.0)),
            (None, Some(3.0), Some(0.25), Some(1.0)),
            (Some(1.0), None, Some(0.25), Some(1.0)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("l"), col("s"), col("d")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.039583424160565546);
        assert_eq_float!(res_col.value(1), 0.07797913037736928);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn students_t_sf_success() {
        let pdf = sf();

        let recs = make_records(vec![
            (Some(1.0), Some(3.0), Some(0.25), Some(1.0)),
            (Some(2.), Some(3.0), Some(0.25), Some(1.0)),
            (None, Some(3.0), Some(0.25), Some(1.0)),
            (Some(1.0), None, Some(0.25), Some(1.0)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("l"), col("s"), col("d")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq_float!(res_col.value(0), 0.9604165758394344);
        assert_eq_float!(res_col.value(1), 0.9220208696226307);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
