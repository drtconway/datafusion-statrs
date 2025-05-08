use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::Uniform;

use super::super::utils::continuous3f::Continuous3F;
use super::super::utils::evaluator3f::{CdfEvaluator3F, LnPdfEvaluator3F, PdfEvaluator3F, SfEvaluator3F};

type Pdf = Continuous3F<PdfEvaluator3F<Uniform>>;

/// ScalarUDF for the Uniform PDF
pub fn pdf() -> ScalarUDF {
    ScalarUDF::from(Pdf::new("uniform_pdf"))
}

type LnPdf = Continuous3F<LnPdfEvaluator3F<Uniform>>;

/// ScalarUDF for the Uniform log PDF
pub fn ln_pdf() -> ScalarUDF {
    ScalarUDF::from(LnPdf::new("uniform_ln_pdf"))
}

type Cdf = Continuous3F<CdfEvaluator3F<Uniform>>;

/// ScalarUDF for the Uniform PDF
pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("uniform_cdf"))
}

type Sf = Continuous3F<SfEvaluator3F<Uniform>>;

/// ScalarUDF for the Uniform PDF
pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("uniform_sf"))
}

/// Register the functions for the Uniform Distribution
pub fn register(registry: &mut dyn FunctionRegistry) -> Result<(), DataFusionError> {
    crate::utils::register::register(registry, vec![pdf(), ln_pdf(), cdf(), sf()])
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
    use statrs::distribution::UniformError;

    use super::*;

    fn get_schema() -> SchemaRef {
        SchemaRef::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("min", DataType::Float64, true),
            Field::new("max", DataType::Float64, true),
        ]))
    }

    fn make_records(rows: Vec<(Option<f64>, Option<f64>, Option<f64>)>) -> RecordBatch {
        let mut xs = Vec::new();
        let mut mns = Vec::new();
        let mut mxs = Vec::new();
        for row in rows {
            xs.push(row.0);
            mns.push(row.1);
            mxs.push(row.2);
        }

        RecordBatch::try_new(
            get_schema(),
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(mns)),
                Arc::new(Float64Array::from(mxs)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn uniform_pdf_success() {
        let pdf = pdf();

        let recs = make_records(vec![
            (Some(2.0), Some(1.0), Some(3.25)),
            (Some(2.5), Some(1.0), Some(3.25)),
            (None, Some(1.0), Some(3.25)),
            (Some(1.0), None, Some(3.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("min"), col("max")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.4444444444444444);
        assert_eq!(res_col.value(1), 0.4444444444444444);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn uniform_pdf_failure_1() {
        let pdf = pdf();

        let recs = make_records(vec![(Some(1.0), Some(5.0), Some(1.25))]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("min"), col("max")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await;
        match res {
            Err(DataFusionError::External(e)) => {
                let be = e.downcast::<UniformError>().unwrap();
                assert_eq!(*be.as_ref(), UniformError::MaxNotGreaterThanMin);
            }
            _ => {
                println!("unexpected result: {:?}", res);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn uniform_ln_pdf_success() {
        let mut ctx = SessionContext::new();
        register(&mut ctx).unwrap();
        let res = ctx
            .sql("SELECT uniform_ln_pdf(8.2, 8.0, 11.0)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 1);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), -1.0986122886681098);
    }

    #[tokio::test]
    async fn uniform_cdf_success() {
        let pdf = cdf();

        let recs = make_records(vec![
            (Some(2.0), Some(1.0), Some(3.25)),
            (Some(2.5), Some(1.0), Some(3.25)),
            (None, Some(1.0), Some(3.25)),
            (Some(1.0), None, Some(3.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("min"), col("max")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.4444444444444444);
        assert_eq!(res_col.value(1), 0.6666666666666666);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn uniform_sf_success() {
        let pdf = sf();

        let recs = make_records(vec![
            (Some(2.0), Some(1.0), Some(3.25)),
            (Some(2.5), Some(1.0), Some(3.25)),
            (None, Some(1.0), Some(3.25)),
            (Some(1.0), None, Some(3.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("min"), col("max")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.5555555555555556);
        assert_eq!(res_col.value(1), 0.3333333333333333);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
