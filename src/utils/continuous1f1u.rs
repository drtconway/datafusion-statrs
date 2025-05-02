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

use super::evaluator1f1u::Evaluator1F1U;

#[derive(Debug)]
pub struct Continuous1F1U<E: Evaluator1F1U> {
    name: String,
    signature: Signature,
    _phantom: PhantomData<E>,
}

impl<E: Evaluator1F1U> Continuous1F1U<E> {
    pub fn new(name: &str) -> Self {
        Continuous1F1U {
            name: String::from(name),
            signature: Signature::exact(
                vec![DataType::Float64, DataType::UInt64],
                Volatility::Immutable,
            ),
            _phantom: PhantomData,
        }
    }
}

impl<E: Evaluator1F1U> ScalarUDFImpl for Continuous1F1U<E> {
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

        assert_eq!(x_array.len(), n_array.len());

        let array: Float64Array = x_array
            .iter()
            .zip(n_array)
            .map(|(x, n)| match (x, n) {
                (Some(x), Some(n)) => E::eval(x, n),
                _ => Ok(Some(f64::NAN)),
            })
            .collect::<Result<Float64Array, DataFusionError>>()?;
        Ok(ColumnarValue::from(Arc::new(array) as ArrayRef))
    }
}
