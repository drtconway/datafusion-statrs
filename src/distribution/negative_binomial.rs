use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::NegativeBinomial;

use crate::utils::discrete1u2f::Discrete1U2F;
use crate::utils::evaluator1u2f::{CdfEvaluator1U2F, PmfEvaluator1U2F, SfEvaluator1U2F};

type Pmf = Discrete1U2F<PmfEvaluator1U2F<NegativeBinomial>>;

/// ScalarUDF for the Negative Binomial PMF
pub fn pmf() -> ScalarUDF {
    ScalarUDF::from(Pmf::new("negative_binomial_pmf"))
}

type Cdf = Discrete1U2F<CdfEvaluator1U2F<NegativeBinomial>>;

/// ScalarUDF for the Negative Binomial CDF
pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("negative_binomial_cdf"))
}

type Sf = Discrete1U2F<SfEvaluator1U2F<NegativeBinomial>>;

/// ScalarUDF for the Negative Binomial SF
pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("negative_binomial_sf"))
}

/// Register the functions for the Negative Binomial Distribution
pub fn register(registry: &mut dyn FunctionRegistry) -> Result<(), DataFusionError> {
    crate::utils::register::register(registry, vec![pmf(), cdf(), sf()])
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::{
        arrow::{
            array::{Float64Array, RecordBatch, UInt64Array},
            datatypes::{DataType, Field, Schema, SchemaRef},
        },
        common::cast::as_float64_array,
        error::DataFusionError,
        prelude::{SessionContext, col},
    };
    use statrs::distribution::NegativeBinomialError;

    use super::*;

    fn get_schema() -> SchemaRef {
        SchemaRef::new(Schema::new(vec![
            Field::new("x", DataType::UInt64, true),
            Field::new("r", DataType::Float64, true),
            Field::new("p", DataType::Float64, true),
        ]))
    }

    fn make_records(rows: Vec<(Option<u64>, Option<f64>, Option<f64>)>) -> RecordBatch {
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
                Arc::new(UInt64Array::from(xs)),
                Arc::new(Float64Array::from(ss)),
                Arc::new(Float64Array::from(rs)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn negative_binomial_pmf_success() {
        let pmf = pmf();

        let recs = make_records(vec![
            (Some(1), Some(3.0), Some(0.25)),
            (Some(2), Some(3.0), Some(0.25)),
            (None, Some(3.0), Some(0.25)),
            (Some(1), None, Some(0.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("r"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.035156249999999827);
        assert_eq!(res_col.value(1), 0.05273437499999992);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn negative_binomial_pmf_failure_1() {
        let pmf = pmf();

        let recs = make_records(vec![(Some(1), Some(0.0), Some(1.25))]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("r"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await;
        match res {
            Err(DataFusionError::External(e)) => {
                let be = e.downcast::<NegativeBinomialError>().unwrap();
                assert_eq!(*be.as_ref(), NegativeBinomialError::PInvalid);
            }
            _ => {
                println!("unexpected result: {:?}", res);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn negative_binomial_cdf_success() {
        let pmf = cdf();

        let recs = make_records(vec![
            (Some(1), Some(3.0), Some(0.25)),
            (Some(2), Some(3.0), Some(0.25)),
            (None, Some(3.0), Some(0.25)),
            (Some(1), None, Some(0.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("r"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.050781250000000056);
        assert_eq!(res_col.value(1), 0.10351562499999896);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn negative_binomial_sf_success() {
        let pmf = sf();

        let recs = make_records(vec![
            (Some(1), Some(3.0), Some(0.25)),
            (Some(2), Some(3.0), Some(0.25)),
            (None, Some(3.0), Some(0.25)),
            (Some(1), None, Some(0.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("r"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.94921875);
        assert_eq!(res_col.value(1), 0.896484375000001);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
