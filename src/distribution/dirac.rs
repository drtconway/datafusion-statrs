//! Module containing functions to the Dirac Distribution.
//! 
//! Implemented by [`statrs::distribution::Dirac`].
//! 
//! The [Dirac Distribution](https://en.wikipedia.org/wiki/Dirac_delta_function) has one
//! parameter:
//! 
//! a: a ∈ R (real numbers)
//! 
//! Usage:
//! 
//! `dirac_cdf(x, a)`  
//! `dirac_sf(x, a)`
//! 
//! with
//! 
//!   `x`: (-∞, +∞) `Float64`/`DOUBLE`,  
//!   `a`: (-∞, +∞) `Float64`/`DOUBLE`
//! 
//! Examples
//! ```
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() -> std::io::Result<()> {
//!     let mut ctx = datafusion::prelude::SessionContext::new();
//!     datafusion_statrs::distribution::dirac::register(&mut ctx)?;
//!     ctx.sql("SELECT dirac_cdf(0.1, 1.2)").await?
//!        .show().await?;
//!     Ok(())
//! }
//! ```

use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::Dirac;

use crate::utils::continuous2f::Continuous2F;
use crate::utils::evaluator2f::{CdfEvaluator2F, SfEvaluator2F};

type Cdf = Continuous2F<CdfEvaluator2F<Dirac>>;

/// ScalarUDF for the Dirac Distribution CDF
pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("dirac_cdf"))
}

type Sf = Continuous2F<SfEvaluator2F<Dirac>>;

/// ScalarUDF for the Dirac Distribution SF
pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("dirac_sf"))
}

/// Register the functions for the Dirac Distribution
pub fn register(registry: &mut dyn FunctionRegistry) -> Result<(), DataFusionError> {
    crate::utils::register::register(registry, vec![cdf(), sf()])
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
        prelude::{SessionContext, col},
    };

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
    async fn dirac_cdf_success() {
        let pmf = cdf();

        let recs = make_records(vec![
            (Some(0.1), Some(0.25)),
            (Some(1.0), Some(0.25)),
            (None, Some(0.25)),
            (Some(1.0), None),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("p")])).alias("q"),
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
        assert_eq!(res_col.value(1), 1.0);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn dirac_sf_success() {
        let pmf = sf();

        let recs = make_records(vec![
            (Some(0.1), Some(0.25)),
            (Some(1.0), Some(0.25)),
            (None, Some(0.25)),
            (Some(1.0), None),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 1.0);
        assert_eq!(res_col.value(1), 0.0);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
