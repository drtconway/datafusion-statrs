use std::{marker::PhantomData, sync::Arc};

use datafusion::{
    arrow::{array::{ArrayRef, Float64Array}, datatypes::DataType}, common::cast::as_uint64_array, error::DataFusionError, logical_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility}
};

use super::evaluator4u::Evaluator4U;

#[derive(Debug)]
pub struct Discrete4U<E: Evaluator4U> {
    name: String,
    signature: Signature,
    _phantom: PhantomData<E>
}

impl<E: Evaluator4U> Discrete4U<E> {
    pub fn new(name: &str) -> Self {
        Discrete4U {
            name: String::from(name),
            signature: Signature::uniform(4, vec![DataType::UInt64], Volatility::Immutable),
            _phantom: PhantomData
        }
    }
}

impl<E: Evaluator4U> ScalarUDFImpl for Discrete4U<E> {
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
        let x_array = as_uint64_array(&args[0]).expect("cast failed");
        let p1_array = as_uint64_array(&args[1]).expect("cast failed");
        let p2_array = as_uint64_array(&args[2]).expect("cast failed");
        let p3_array = as_uint64_array(&args[3]).expect("cast failed");

        assert_eq!(x_array.len(), p1_array.len());
        assert_eq!(x_array.len(), p2_array.len());
        assert_eq!(x_array.len(), p3_array.len());

        let array: Float64Array = x_array
            .iter()
            .zip(p1_array)
            .zip(p2_array)
            .zip(p3_array)
            .map(|(((x, p1), p2), p3)| match (x, p1, p2, p3) {
                (Some(x), Some(p1), Some(p2), Some(p3)) => E::eval(x, p1, p2, p3),
                _ => Ok(Some(f64::NAN)),
            })
            .collect::<Result<Float64Array, DataFusionError>>()?;
        Ok(ColumnarValue::from(Arc::new(array) as ArrayRef))
    }
}
