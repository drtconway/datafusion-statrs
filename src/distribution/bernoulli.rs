use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::Bernoulli;

use super::super::utils::discrete1u1f::Discrete1U1F;
use super::super::utils::evaluator1u1f::{CdfEvaluator1U1F, PmfEvaluator1U1F, SfEvaluator1U1F};

pub type Pmf = Discrete1U1F<PmfEvaluator1U1F<Bernoulli>>;

pub fn pmf() -> ScalarUDF {
    ScalarUDF::from(Pmf::new("bernoulli_pmf"))
}

pub type Cdf = Discrete1U1F<CdfEvaluator1U1F<Bernoulli>>;

pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("bernoulli_cdf"))
}

pub type Sf = Discrete1U1F<SfEvaluator1U1F<Bernoulli>>;

pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("bernoulli_sf"))
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
    use statrs::distribution::BinomialError;

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
    async fn bernoulli_pmf_success() {
        let pmf = pmf();

        let recs = make_records(vec![
            (Some(0), Some(0.25)),
            (Some(1), Some(0.25)),
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
        assert_eq!(res_col.value(0), 0.75);
        assert_eq!(res_col.value(1), 0.25);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn bernoulli_pmf_failure_1() {
        let pmf = pmf();

        let recs = make_records(vec![(Some(0), Some(1.25))]);

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
                let be = e.downcast::<BinomialError>().unwrap();
                assert_eq!(*be.as_ref(), BinomialError::ProbabilityInvalid);
            }
            _ => {
                println!("unexpected result: {:?}", res);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn bernoulli_cdf_success() {
        let pmf = cdf();

        let recs = make_records(vec![
            (Some(0), Some(0.25)),
            (Some(1), Some(0.25)),
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
        assert_eq!(res_col.value(0), 0.7499999999999998); // :cry:
        assert_eq!(res_col.value(1), 1.0);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn bernoulli_sf_success() {
        let pmf = sf();

        let recs = make_records(vec![
            (Some(0), Some(0.25)),
            (Some(1), Some(0.25)),
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
        assert_eq!(res_col.value(0), 0.2500000000000002); // :cry:
        assert_eq!(res_col.value(1), 0.0);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
