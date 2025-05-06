use std::sync::Arc;

use datafusion::{error::DataFusionError, execution::FunctionRegistry, logical_expr::ScalarUDF};
use log::warn;

pub fn register(registry: &mut dyn FunctionRegistry, functions: Vec<ScalarUDF>) -> Result<(), DataFusionError> {
    functions
        .into_iter()
        .map(|f| Arc::new(f))
        .try_for_each(|udf| {
            let existing_udf = registry.register_udf(udf)?;
            if let Some(existing_udf) = existing_udf {
                warn!("Overwrite existing UDF: {}", existing_udf.name());
            }
            Ok(()) as Result<(), DataFusionError>
        })?;
    Ok(())
}