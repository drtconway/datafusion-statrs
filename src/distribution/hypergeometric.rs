use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::Hypergeometric;

use crate::utils::discrete4u::Discrete4U;
use crate::utils::evaluator4u::{CdfEvaluator4U, PmfEvaluator4U, SfEvaluator4U};

type Pmf = Discrete4U<PmfEvaluator4U<Hypergeometric>>;

/// ScalarUDF for the Hypergeometric Distribution PMF
pub fn pmf() -> ScalarUDF {
    ScalarUDF::from(Pmf::new("hypergeometric_pmf"))
}

type Cdf = Discrete4U<CdfEvaluator4U<Hypergeometric>>;

/// ScalarUDF for the Hypergeometric Distribution CDF
pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("hypergeometric_cdf"))
}

type Sf = Discrete4U<SfEvaluator4U<Hypergeometric>>;

/// ScalarUDF for the Hypergeometric Distribution SF
pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("hypergeometric_sf"))
}


/// Register the functions for the Hypergeometric Distribution
pub fn register(registry: &mut dyn FunctionRegistry) -> Result<(), DataFusionError> {
    crate::utils::register::register(registry, vec![pmf(), cdf(), sf()])
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::{
        arrow::{
            array::{RecordBatch, UInt64Array},
            datatypes::{DataType, Field, Schema, SchemaRef},
        },
        common::cast::as_float64_array,
        error::DataFusionError,
        prelude::{SessionContext, col},
    };
    use statrs::distribution::HypergeometricError;

    use super::*;

    fn get_schema() -> SchemaRef {
        SchemaRef::new(Schema::new(vec![
            Field::new("x", DataType::UInt64, true),
            Field::new("p", DataType::UInt64, true),
            Field::new("s", DataType::UInt64, true),
            Field::new("d", DataType::UInt64, true),
        ]))
    }

    fn make_records(
        rows: Vec<(Option<u64>, Option<u64>, Option<u64>, Option<u64>)>,
    ) -> RecordBatch {
        let mut xs = Vec::new();
        let mut ps = Vec::new();
        let mut ss = Vec::new();
        let mut ds = Vec::new();
        for row in rows {
            xs.push(row.0);
            ps.push(row.1);
            ss.push(row.2);
            ds.push(row.3);
        }

        RecordBatch::try_new(
            get_schema(),
            vec![
                Arc::new(UInt64Array::from(xs)),
                Arc::new(UInt64Array::from(ps)),
                Arc::new(UInt64Array::from(ss)),
                Arc::new(UInt64Array::from(ds)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn hypergeometric_pmf_success() {
        let pmf = pmf();

        let recs = make_records(vec![
            (Some(0), Some(20), Some(10), Some(15)),
            (Some(5), Some(20), Some(10), Some(15)),
            (None, Some(20), Some(10), Some(15)),
            (Some(5), None, Some(10), Some(15)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("p"), col("s"), col("d")])).alias("q"),
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
        assert_eq!(res_col.value(1), 0.016253869969040248);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn hypergeometric_pmf_failure_1() {
        let pmf = pmf();

        let recs = make_records(vec![(Some(1), Some(0), Some(5), Some(15))]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("p"), col("s"), col("d")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await;
        match res {
            Err(DataFusionError::External(e)) => {
                let be = e.downcast::<HypergeometricError>().unwrap();
                assert_eq!(*be.as_ref(), HypergeometricError::TooManySuccesses);
            }
            _ => {
                println!("unexpected result: {:?}", res);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn hypergeometric_cdf_success() {
        let pmf = cdf();

        let recs = make_records(vec![
            (Some(0), Some(20), Some(10), Some(15)),
            (Some(5), Some(20), Some(10), Some(15)),
            (None, Some(20), Some(10), Some(15)),
            (Some(5), None, Some(10), Some(15)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("p"), col("s"), col("d")])).alias("q"),
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
        assert_eq!(res_col.value(1), 0.01625386996904021);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn hypergeometric_sf_success() {
        let pmf = sf();

        let recs = make_records(vec![
            (Some(0), Some(20), Some(10), Some(15)),
            (Some(5), Some(20), Some(10), Some(15)),
            (None, Some(20), Some(10), Some(15)),
            (Some(5), None, Some(10), Some(15)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("p"), col("s"), col("d")])).alias("q"),
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
        assert_eq!(res_col.value(1), 0.9837461300309583);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
