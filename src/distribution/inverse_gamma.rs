use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;
use statrs::distribution::InverseGamma;

use crate::utils::continuous3f::Continuous3F;
use crate::utils::evaluator3f::{CdfEvaluator3F, PdfEvaluator3F, SfEvaluator3F};

pub type Pdf = Continuous3F<PdfEvaluator3F<InverseGamma>>;

pub fn pdf() -> ScalarUDF {
    ScalarUDF::from(Pdf::new("inverse_inverse_gamma_pdf"))
}

pub type Cdf = Continuous3F<CdfEvaluator3F<InverseGamma>>;

pub fn cdf() -> ScalarUDF {
    ScalarUDF::from(Cdf::new("inverse_inverse_gamma_cdf"))
}

pub type Sf = Continuous3F<SfEvaluator3F<InverseGamma>>;

pub fn sf() -> ScalarUDF {
    ScalarUDF::from(Sf::new("inverse_inverse_gamma_sf"))
}

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
    use statrs::distribution::InverseGammaError;

    use super::*;

    fn get_schema() -> SchemaRef {
        SchemaRef::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("s", DataType::Float64, true),
            Field::new("r", DataType::Float64, true),
        ]))
    }

    fn make_records(rows: Vec<(Option<f64>, Option<f64>, Option<f64>)>) -> RecordBatch {
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
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ss)),
                Arc::new(Float64Array::from(rs)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn inverse_gamma_pdf_success() {
        let pdf = pdf();

        let recs = make_records(vec![
            (Some(1.0), Some(3.0), Some(0.25)),
            (Some(2.), Some(3.0), Some(0.25)),
            (None, Some(3.0), Some(0.25)),
            (Some(1.0), None, Some(0.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("s"), col("r")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.006084381117745331);
        assert_eq!(res_col.value(1), 0.0004309066907151331);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn inverse_gamma_pdf_failure_1() {
        let pdf = pdf();

        let recs = make_records(vec![(Some(1.0), Some(0.0), Some(1.25))]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("s"), col("r")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await;
        match res {
            Err(DataFusionError::External(e)) => {
                let be = e.downcast::<InverseGammaError>().unwrap();
                assert_eq!(*be.as_ref(), InverseGammaError::ShapeInvalid);
            }
            _ => {
                println!("unexpected result: {:?}", res);
                assert!(false);
            }
        }
    }

    #[tokio::test]
    async fn inverse_gamma_cdf_success() {
        let pdf = cdf();

        let recs = make_records(vec![
            (Some(1.0), Some(3.0), Some(0.25)),
            (Some(2.), Some(3.0), Some(0.25)),
            (None, Some(3.0), Some(0.25)),
            (Some(1.0), None, Some(0.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("s"), col("r")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.9978385033102375);
        assert_eq!(res_col.value(1),  0.999703522459112);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }

    #[tokio::test]
    async fn inverse_gamma_sf_success() {
        let pdf = sf();

        let recs = make_records(vec![
            (Some(1.0), Some(3.0), Some(0.25)),
            (Some(2.), Some(3.0), Some(0.25)),
            (None, Some(3.0), Some(0.25)),
            (Some(1.0), None, Some(0.25)),
        ]);

        let ctx = SessionContext::new();
        ctx.register_batch("tbl", recs).unwrap();
        let df = ctx.table("tbl").await.unwrap();
        let res = df
            .select(vec![
                (pdf.call(vec![col("x"), col("s"), col("r")])).alias("q"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].num_columns(), 1);
        assert_eq!(res[0].num_rows(), 4);
        let res_col = as_float64_array(res[0].column(0)).unwrap();
        assert_eq!(res_col.value(0), 0.0021614966897625073);
        assert_eq!(res_col.value(1), 0.00029647754088801934);
        assert!(res_col.value(2).is_nan());
        assert!(res_col.value(3).is_nan());
    }
}
