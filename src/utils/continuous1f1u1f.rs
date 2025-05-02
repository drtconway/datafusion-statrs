use std::{marker::PhantomData, sync::Arc};

use datafusion::{
    arrow::{
        array::{ArrayRef, Float64Array},
        datatypes::DataType,
    },
    common::cast::{as_float64_array, as_uint64_array},
    error::DataFusionError,
    logical_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility},
};

use super::evaluator1f1u1f::Evaluator1F1U1F;

#[derive(Debug)]
pub struct Continuous1F1U1F<E: Evaluator1F1U1F> {
    name: String,
    signature: Signature,
    _phantom: PhantomData<E>,
}

impl<E: Evaluator1F1U1F> Continuous1F1U1F<E> {
    pub fn new(name: &str) -> Self {
        Continuous1F1U1F {
            name: String::from(name),
            signature: Signature::exact(
                vec![DataType::Float64, DataType::UInt64, DataType::Float64],
                Volatility::Immutable,
            ),
            _phantom: PhantomData,
        }
    }
}

impl<E: Evaluator1F1U1F> ScalarUDFImpl for Continuous1F1U1F<E> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> datafusion::error::Result<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue, DataFusionError> {
        let args = ColumnarValue::values_to_arrays(&args.args)?;
        let x_array = as_float64_array(&args[0]).expect("cast failed");
        let n_array = as_uint64_array(&args[1]).expect("cast failed");
        let p_array = as_float64_array(&args[2]).expect("cast failed");

        assert_eq!(x_array.len(), n_array.len());
        assert_eq!(x_array.len(), p_array.len());

        let array: Float64Array = x_array
            .iter()
            .zip(n_array)
            .zip(p_array)
            .map(|((x, n), p)| match (x, n, p) {
                (Some(x), Some(n), Some(p)) => E::eval(x, n, p),
                _ => Ok(Some(f64::NAN)),
            })
            .collect::<Result<Float64Array, DataFusionError>>()?;
        Ok(ColumnarValue::from(Arc::new(array) as ArrayRef))
    }
}
