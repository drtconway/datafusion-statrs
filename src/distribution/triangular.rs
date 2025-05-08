use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::Triangular;

use crate::utils::continuous4f::Continuous4F;
use crate::utils::evaluator4f::{CdfEvaluator4F, LnPdfEvaluator4F, PdfEvaluator4F, SfEvaluator4F};

type Pdf = Continuous4F<PdfEvaluator4F<Triangular>>;

/// ScalarUDF for the Triangular PDF
pub fn pdf() -> ScalarUDF {
    ScalarUDF::from(Pdf::new("triangular_pdf"))
}

type LnPdf = Continuous4F<LnPdfEvaluator4F<Triangular>>;

/// ScalarUDF for the Triangular log PDF
pub fn ln_pdf() -> ScalarUDF {
    ScalarUDF::from(LnPdf::new("triangular_ln_pdf"))
}

type Cdf = Continuous4F<CdfEvaluator4F<Triangular>>;

/// ScalarUDF for the Triangular PDF
pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("triangular_cdf"))
}

type Sf = Continuous4F<SfEvaluator4F<Triangular>>;

/// ScalarUDF for the Triangular PDF
pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("triangular_sf"))
}

/// Register the functions for the Triangular Distribution
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
    use statrs::distribution::TriangularError;

    use super::*;

    fn get_schema() -> SchemaRef {
        SchemaRef::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("min", DataType::Float64, true),
            Field::new("max", DataType::Float64, true),
            Field::new("mode", DataType::Float64, true),
        ]))
    }

    fn make_records(
        rows: Vec<(Option<f64>, Option<f64>, Option<f64>, Option<f64>)>,
    ) -> RecordBatch {
        let mut xs = Vec::new();
        let mut mns = Vec::new();
        let mut mxs = Vec::new();
        let mut mds = Vec::new();
        for row in rows {
            xs.push(row.0);
            mns.push(row.1);
            mxs.push(row.2);
            mds.push(row.3);
        }

        RecordBatch::try_new(
            get_schema(),
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(mns)),
                Arc::new(Float64Array::from(mxs)),
                Arc::new(Float64Array::from(mds)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn triangular_pdf_success() {
        let pdf = pdf();

        let recs = make_records(vec![
            (Some(5.0), Some(3.0), Some(7.0), Some(4.0)),
            (Some(6.0), Some(3.0), Some(7.0), Some(4.0)),
            (None, Some(3.0), Some(7.0), Some(4.0)),
            (Some(6.0), None, Some(7.0), Some(4.0)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("min"), col("max"), col("mode")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.3333333333333333);
        assert_eq!(res_col.value(1), 0.16666666666666666);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn triangular_pdf_failure_1() {
        let pdf = pdf();

        let recs = make_records(vec![(Some(1.0), Some(0.0), Some(1.0), Some(-1.25))]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("min"), col("max"), col("mode")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await;
        match res {
            Err(DataFusionError::External(e)) => {
                let be = e.downcast::<TriangularError>().unwrap();
                assert_eq!(*be.as_ref(), TriangularError::ModeOutOfRange);
            }
            _ => {
                println!("unexpected result: {:?}", res);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn triangular_ln_pdf_success() {
        let mut ctx = SessionContext::new();
        register(&mut ctx).unwrap();
        let res = ctx
            .sql("SELECT triangular_ln_pdf(3.14, 3.0, 7.0, 6.0)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 1);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), -3.757872325600887);
    }

    #[tokio::test]
    async fn triangular_cdf_success() {
        let pdf = cdf();

        let recs = make_records(vec![
            (Some(5.0), Some(3.0), Some(7.0), Some(4.0)),
            (Some(6.0), Some(3.0), Some(7.0), Some(4.0)),
            (None, Some(3.0), Some(7.0), Some(4.0)),
            (Some(6.0), None, Some(7.0), Some(4.0)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("min"), col("max"), col("mode")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.6666666666666667);
        assert_eq!(res_col.value(1), 0.9166666666666666);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn triangular_sf_success() {
        let pdf = sf();

        let recs = make_records(vec![
            (Some(5.0), Some(3.0), Some(7.0), Some(4.0)),
            (Some(6.0), Some(3.0), Some(7.0), Some(4.0)),
            (None, Some(3.0), Some(7.0), Some(4.0)),
            (Some(6.0), None, Some(7.0), Some(4.0)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("min"), col("max"), col("mode")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.3333333333333333);
        assert_eq!(res_col.value(1), 0.08333333333333333);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
