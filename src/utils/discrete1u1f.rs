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

use super::evaluator1u1f::Evaluator1U1F;

#[derive(Debug)]
pub struct Discrete1U1F<E: Evaluator1U1F> {
    name: String,
    signature: Signature,
    _phantom: PhantomData<E>,
}

impl<E: Evaluator1U1F> Discrete1U1F<E> {
    pub fn new(name: &str) -> Self {
        Discrete1U1F {
            name: String::from(name),
            signature: Signature::exact( vec![DataType::UInt64, DataType::Float64], Volatility::Immutable),
            _phantom: PhantomData,
        }
    }
}

impl<E: Evaluator1U1F> ScalarUDFImpl for Discrete1U1F<E> {
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
        let p_array = as_float64_array(&args[1]).expect("cast failed");

        assert_eq!(x_array.len(), p_array.len());

        let array: Float64Array = x_array
            .iter()
            .zip(p_array)
            .map(|(x, p)| match (x, p) {
                (Some(x), Some(p)) => E::eval(x, p),
                _ => Ok(Some(f64::NAN)),
            })
            .collect::<Result<Float64Array, DataFusionError>>()?;
        Ok(ColumnarValue::from(Arc::new(array) as ArrayRef))
    }
}
