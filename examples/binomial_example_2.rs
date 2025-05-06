use datafusion::prelude::{CsvReadOptions, SessionContext};

#[tokio::main(flavor = "current_thread")]
async fn main() -> std::io::Result<()> {
    let binom_cdf = datafusion_statrs::distribution::binomial::cdf();
    let binom_sf = datafusion_statrs::distribution::binomial::sf();

    let mut opts = CsvReadOptions::default();
    opts.delimiter = b'\t';
    opts.file_extension = ".tsv";

    let ctx = SessionContext::new();
    ctx.register_udf(binom_cdf);
    ctx.register_udf(binom_sf);
    ctx.register_csv("coins", "examples/coins.tsv", opts)
        .await?;
    ctx.sql("WITH
    coins_1 AS (
        SELECT
            coin_id,
            arrow_cast(tosses, 'UInt64') AS tosses,
            arrow_cast(heads, 'UInt64') AS heads,
            arrow_cast(tails, 'UInt64') AS tails
        FROM coins
        WHERE heads > tails
    ),
    coins_2 AS (
    SELECT
        coin_id,
        tosses,
        heads,
        tails,
        binomial_cdf(tails, tosses, 0.5) AS lower,
        binomial_sf(heads, tosses, 0.5) AS upper
    FROM coins_1
    )
    SELECT coin_id, tosses, heads, tails, (lower + upper) AS p_value FROM coins_2 WHERE lower + upper < 0.01").await?.show().await?;

    Ok(())
}
