use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::FisherSnedecor;

use crate::utils::continuous3f::Continuous3F;
use crate::utils::evaluator3f::{CdfEvaluator3F, PdfEvaluator3F, SfEvaluator3F};

pub type Pdf = Continuous3F<PdfEvaluator3F<FisherSnedecor>>;

pub fn pdf() -> ScalarUDF {
    ScalarUDF::from(Pdf::new("fisher_snedecor_pdf"))
}

pub type Cdf = Continuous3F<CdfEvaluator3F<FisherSnedecor>>;

pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("fisher_snedecor_cdf"))
}

pub type Sf = Continuous3F<SfEvaluator3F<FisherSnedecor>>;

pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("fisher_snedecor_sf"))
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
        assert_eq!(res_col.value(1), 0.08642373027968221);
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
        assert_eq!(res_col.value(0), 0.0);
        assert_eq!(res_col.value(1), 0.22377660964255752);
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
        assert_eq!(res_col.value(0), 1.0);
        assert_eq!(res_col.value(1), 0.7762233903574425);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
