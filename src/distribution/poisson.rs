use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::Poisson;

use super::super::utils::discrete1u1f::Discrete1U1F;
use super::super::utils::evaluator1u1f::{CdfEvaluator1U1F, LnPmfEvaluator1U1F, PmfEvaluator1U1F, SfEvaluator1U1F};

type Pmf = Discrete1U1F<PmfEvaluator1U1F<Poisson>>;

/// ScalarUDF for the Poisson PMF
pub fn pmf() -> ScalarUDF {
    ScalarUDF::from(Pmf::new("poisson_pmf"))
}

type LnPmf = Discrete1U1F<LnPmfEvaluator1U1F<Poisson>>;

/// ScalarUDF for the Poisson log PMF
pub fn ln_pmf() -> ScalarUDF {
    ScalarUDF::from(LnPmf::new("poisson_ln_pmf"))
}

type Cdf = Discrete1U1F<CdfEvaluator1U1F<Poisson>>;

/// ScalarUDF for the Poisson PDF
pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("poisson_cdf"))
}

type Sf = Discrete1U1F<SfEvaluator1U1F<Poisson>>;

/// ScalarUDF for the Poisson PDF
pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("poisson_sf"))
}

/// Register the functions for the Poisson Distribution
pub fn register(registry: &mut dyn FunctionRegistry) -> Result<(), DataFusionError> {
    crate::utils::register::register(registry, vec![pmf(), ln_pmf(), cdf(), sf()])
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
    use statrs::distribution::PoissonError;

    use super::*;

    fn get_schema() -> SchemaRef {
        SchemaRef::new(Schema::new(vec![
            Field::new("x", DataType::UInt64, true),
            Field::new("p", DataType::Float64, true),
        ]))
    }

    fn make_records(rows: Vec<(Option<u64>, Option<f64>)>) -> RecordBatch {
        let mut xs = Vec::new();
        let mut ps = Vec::new();
        for row in rows {
            xs.push(row.0);
            ps.push(row.1);
        }

        RecordBatch::try_new(
            get_schema(),
            vec![
                Arc::new(UInt64Array::from(xs)),
                Arc::new(Float64Array::from(ps)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn poisson_pmf_success() {
        let pmf = pmf();

        let recs = make_records(vec![
            (Some(0), Some(0.25)),
            (Some(5), Some(0.25)),
            (None, Some(0.25)),
            (Some(0), None),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![(pmf.call(vec![col("x"), col("p")])).alias("q")])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.7788007830714049);
        assert_eq!(res_col.value(1), 6.337896997651408e-6);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn poisson_pmf_failure_1() {
        let pmf = pmf();

        let recs = make_records(vec![(Some(0), Some(-1.25))]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![(pmf.call(vec![col("x"), col("p")])).alias("q")])
            .unwrap()
            .collect()
            .await;
        match res {
            Err(DataFusionError::External(e)) => {
                let be = e.downcast::<PoissonError>().unwrap();
                assert_eq!(*be.as_ref(), PoissonError::LambdaInvalid);
            }
            _ => {
                println!("unexpected result: {:?}", res);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn poisson_ln_pmf_success() {
        let mut ctx = SessionContext::new();
        register(&mut ctx).unwrap();
        let res = ctx
            .sql("SELECT poisson_ln_pmf(CAST(8 AS BIGINT UNSIGNED), 2.5)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 1);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), -5.77427704775201);
    }

    #[tokio::test]
    async fn poisson_cdf_success() {
        let pmf = cdf();

        let recs = make_records(vec![
            (Some(0), Some(0.25)),
            (Some(5), Some(0.25)),
            (None, Some(0.25)),
            (Some(0), None),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![(pmf.call(vec![col("x"), col("p")])).alias("q")])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.7788007830714048);
        assert_eq!(res_col.value(1), 0.9999997261864366);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn poisson_sf_success() {
        let pmf = sf();

        let recs = make_records(vec![
            (Some(0), Some(0.25)),
            (Some(5), Some(0.25)),
            (None, Some(0.25)),
            (Some(0), None),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![(pmf.call(vec![col("x"), col("p")])).alias("q")])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.2211992169285952);
        assert_eq!(res_col.value(1), 2.738135633828412e-7);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
