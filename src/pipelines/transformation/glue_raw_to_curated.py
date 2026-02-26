"""AWS Glue job placeholder for raw-to-curated transformation."""

from __future__ import annotations

try:
    from awsglue.context import GlueContext
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType
except Exception:  # pragma: no cover
    GlueContext = None
    SparkSession = None
    F = None
    DoubleType = None


def run_raw_to_curated() -> int:
    if not SparkSession or not GlueContext or not F or not DoubleType:
        raise RuntimeError("Glue/PySpark dependencies are not installed")

    spark = SparkSession.builder.getOrCreate()
    _glue_context = GlueContext(spark)

    raw_df = spark.read.json("s3://merchantmind-data-lake/raw/transactions/")

    cleaned_df = (
        raw_df.filter(F.col("transaction_id").isNotNull())
        .filter(F.col("merchant_id").isNotNull())
        .filter(F.col("amount_aud") > 0)
        .withColumn("amount_aud", F.col("amount_aud").cast(DoubleType()))
        .withColumn("timestamp", F.to_timestamp("timestamp"))
        .dropDuplicates(["transaction_id"])
        .withColumn("processed_at", F.current_timestamp())
        .withColumn("data_quality_flag", F.lit("PASSED"))
    )

    cleaned_df.write.mode("overwrite").partitionBy("merchant_category", "state").parquet(
        "s3://merchantmind-data-lake/cleaned/transactions/"
    )

    snowflake_options = {
        "sfURL": "merchantmind.ap-southeast-2.snowflakecomputing.com",
        "sfDatabase": "MERCHANTMIND",
        "sfSchema": "CURATED",
        "sfWarehouse": "COMPUTE_WH",
    }

    cleaned_df.write.format("snowflake").options(**snowflake_options).option(
        "dbtable", "TRANSACTIONS"
    ).mode("append").save()

    return cleaned_df.count()


if __name__ == "__main__":
    count = run_raw_to_curated()
    print(f"Processed {count:,} transactions to curated layer.")
