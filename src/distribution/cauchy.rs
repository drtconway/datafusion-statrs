use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::Cauchy;

use crate::utils::continuous3f::Continuous3F;
use crate::utils::evaluator3f::{CdfEvaluator3F, PdfEvaluator3F, SfEvaluator3F};

pub type Pdf = Continuous3F<PdfEvaluator3F<Cauchy>>;

pub fn pdf() -> ScalarUDF {
    ScalarUDF::from(Pdf::new("cauchy_pdf"))
}

pub type Cdf = Continuous3F<CdfEvaluator3F<Cauchy>>;

pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("cauchy_cdf"))
}

pub type Sf = Continuous3F<SfEvaluator3F<Cauchy>>;

pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("cauchy_sf"))
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
    use statrs::distribution::CauchyError;

    use super::*;

    fn get_schema() -> SchemaRef {
        SchemaRef::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("x0", DataType::Float64, true),
            Field::new("p", DataType::Float64, true),
        ]))
    }

    fn make_records(rows: Vec<(Option<f64>, Option<f64>, Option<f64>)>) -> RecordBatch {
        let mut xs = Vec::new();
        let mut x0s = Vec::new();
        let mut ps = Vec::new();
        for row in rows {
            xs.push(row.0);
            x0s.push(row.1);
            ps.push(row.2);
        }

        RecordBatch::try_new(
            get_schema(),
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(x0s)),
                Arc::new(Float64Array::from(ps)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn cauchy_pdf_success() {
        let pdf = pdf();

        let recs = make_records(vec![
            (Some(-1.0), Some(0.0), Some(0.5)),
            (Some(1.0), Some(0.0), Some(0.5)),
            (None, Some(3.0), Some(-1.25)),
            (Some(0.0), None, Some(5.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("x0"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.12732395447351627);
        assert_eq!(res_col.value(1), 0.12732395447351627);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn cauchy_pdf_failure_1() {
        let pdf = pdf();

        let recs = make_records(vec![(Some(0.0), Some(3.0), Some(-1.25))]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("x0"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await;
        match res {
            Err(DataFusionError::External(e)) => {
                let be = e.downcast::<CauchyError>().unwrap();
                assert_eq!(*be.as_ref(), CauchyError::ScaleInvalid);
            }
            _ => {
                println!("unexpected result: {:?}", res);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn cauchy_cdf_success() {
        let pmf = cdf();

        let recs = make_records(vec![
            (Some(-1.0), Some(0.0), Some(0.5)),
            (Some(1.0), Some(0.0), Some(0.5)),
            (None, Some(3.0), Some(-1.25)),
            (Some(0.0), None, Some(5.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("x0"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.1475836176504332);
        assert_eq!(res_col.value(1), 0.8524163823495667);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn beta_sf_success() {
        let pmf = sf();

        let recs = make_records(vec![
            (Some(-1.0), Some(0.0), Some(0.5)),
            (Some(1.0), Some(0.0), Some(0.5)),
            (None, Some(3.0), Some(-1.25)),
            (Some(0.0), None, Some(5.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pmf.call(vec![col("x"), col("x0"), col("p")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.8524163823495667);
        assert_eq!(res_col.value(1), 0.1475836176504332);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
