# datafusion-statrs - datafusion bindings for statrs statistical functions

This library provides bindings to provide the statistical distribution functions from [statrs](https://github.com/statrs-dev/statrs) in [datafusion](https://github.com/apache/datafusion).

## Example

```rust
use datafusion::{
    arrow::datatypes::DataType,
    logical_expr::case,
    prelude::{CsvReadOptions, SessionContext, cast, col, lit},
};

#[tokio::main(flavor = "current_thread")]
async fn main() -> std::io::Result<()> {
    let binom_cdf = datafusion_statrs::distribution::binomial::cdf();
    let binom_sf = datafusion_statrs::distribution::binomial::sf();

    let mut opts = CsvReadOptions::default();
    opts.delimiter = b'\t';
    opts.file_extension = ".tsv";

    let ctx = SessionContext::new();
    let df = ctx.read_csv("examples/coins.tsv", opts).await?;

    // Cast the columns to the right types
    df.select(vec![
        cast(col("coin_id"), DataType::UInt64).alias("coin_id"),
        cast(col("tosses"), DataType::UInt64).alias("tosses"),
        cast(col("heads"), DataType::UInt64).alias("heads"),
        cast(col("tails"), DataType::UInt64).alias("tails"),
    ])?

    // Add columns with the min and max of heads and tails.
    .select(vec![
        col("coin_id"),
        col("tosses"),
        col("heads"),
        col("tails"),
        (case(col("heads").lt(col("tails")))
            .when(lit(true), col("heads"))
            .otherwise(col("tails")))?
        .alias("min"),
        (case(col("heads").gt(col("tails")))
            .when(lit(true), col("heads"))
            .otherwise(col("tails")))?
        .alias("max"),
    ])?

    // Now compute the probability of a more extreme outcome as the
    // probability of a lower min and a higher max under the assumption
    // that the coin is fair.
    .select(vec![
        col("coin_id"),
        col("tosses"),
        col("heads"),
        col("tails"),
        col("min"),
        col("max"),
        (binom_cdf.call(vec![col("min"), col("tosses"), lit(0.5)])).alias("lower"),
        (binom_sf.call(vec![col("max"), col("tosses"), lit(0.5)])).alias("upper"),
    ])?

    // Now compute the p-value for the null hypothesis that the coin is fair
    .select(vec![
        col("coin_id"),
        col("tosses"),
        col("heads"),
        col("tails"),
        (col("lower") + col("upper")).alias("p_value"),
    ])?
    
    // Filter for significance
    .filter(col("p_value").lt(lit(0.01)))?
    .show()
    .await?;

    Ok(())
}
```
