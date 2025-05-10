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
    .with_column(
        "min",
        case(col("heads").lt(col("tails")))
            .when(lit(true), col("heads"))
            .otherwise(col("tails"))?,
    )?
    .with_column(
        "max",
        case(col("heads").gt(col("tails")))
            .when(lit(true), col("heads"))
            .otherwise(col("tails"))?,
    )?

    // Now compute the probability of a more extreme outcome as the
    // probability of a lower min and a higher max under the assumption
    // that the coin is fair.
    .with_column("lower", binom_cdf.call(vec![col("min"), col("tosses"), lit(0.5)]))?
    .with_column("upper", binom_sf.call(vec![col("max"), col("tosses"), lit(0.5)]))?

    // Now compute the p-value for the null hypothesis that the coin is fair
    .with_column("p_value", col("lower") + col("upper"))?
    .drop_columns(&["min", "max", "lower", "upper"])?

    // Filter for significance
    .filter(col("p_value").lt(lit(0.01)))?
    .show()
    .await?;

    Ok(())
}
