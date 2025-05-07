use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::Weibull;

use super::super::utils::continuous3f::Continuous3F;
use super::super::utils::evaluator3f::{CdfEvaluator3F, PdfEvaluator3F, SfEvaluator3F};

type Pdf = Continuous3F<PdfEvaluator3F<Weibull>>;

/// ScalarUDF for the Weibull PDF
pub fn pdf() -> ScalarUDF {
    ScalarUDF::from(Pdf::new("weibull_pdf"))
}

type Cdf = Continuous3F<CdfEvaluator3F<Weibull>>;

/// ScalarUDF for the Weibull PDF
pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("weibull_cdf"))
}

type Sf = Continuous3F<SfEvaluator3F<Weibull>>;

/// ScalarUDF for the Weibull PDF
pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("weibull_sf"))
}

/// Register the functions for the Weibull Distribution
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
    use statrs::distribution::WeibullError;

    use super::*;

    fn get_schema() -> SchemaRef {
        SchemaRef::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("lambda", DataType::Float64, true),
            Field::new("k", DataType::Float64, true),
        ]))
    }

    fn make_records(rows: Vec<(Option<f64>, Option<f64>, Option<f64>)>) -> RecordBatch {
        let mut xs = Vec::new();
        let mut ls = Vec::new();
        let mut ks = Vec::new();
        for row in rows {
            xs.push(row.0);
            ls.push(row.1);
            ks.push(row.2);
        }

        RecordBatch::try_new(
            get_schema(),
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ls)),
                Arc::new(Float64Array::from(ks)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn weibull_pdf_success() {
        let pdf = pdf();

        let recs = make_records(vec![
            (Some(0.5), Some(1.0), Some(0.5)),
            (Some(0.75), Some(1.0), Some(0.5)),
            (None, Some(1.0), Some(0.25)),
            (Some(1.0), None, Some(0.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("lambda"), col("k")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.7357588823428847);
        assert_eq!(res_col.value(1), 0.44626032029685964);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn weibull_pdf_failure_1() {
        let pdf = pdf();

        let recs = make_records(vec![(Some(1.0), Some(0.0), Some(1.25))]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("lambda"), col("k")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await;
        match res {
            Err(DataFusionError::External(e)) => {
                let be = e.downcast::<WeibullError>().unwrap();
                assert_eq!(*be.as_ref(), WeibullError::ShapeInvalid);
            }
            _ => {
                println!("unexpected result: {:?}", res);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn weibull_cdf_success() {
        let pdf = cdf();

        let recs = make_records(vec![
            (Some(0.5), Some(1.0), Some(0.5)),
            (Some(0.75), Some(1.0), Some(0.5)),
            (None, Some(1.0), Some(0.25)),
            (Some(1.0), None, Some(0.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("lambda"), col("k")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.6321205588285577);
        assert_eq!(res_col.value(1), 0.7768698398515702);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn weibull_sf_success() {
        let pdf = sf();

        let recs = make_records(vec![
            (Some(0.5), Some(1.0), Some(0.5)),
            (Some(0.75), Some(1.0), Some(0.5)),
            (None, Some(1.0), Some(0.25)),
            (Some(1.0), None, Some(0.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("lambda"), col("k")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.36787944117144233);
        assert_eq!(res_col.value(1), 0.22313016014842982);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
