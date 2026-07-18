import os

import pytest

from src.stats.descriptive import run_map_reduce_analysis, run_sql_analysis


pytestmark = [
    pytest.mark.spark,
    pytest.mark.skipif(
        os.name == "nt",
        reason="Spark inference is supported through the Linux Docker image.",
    ),
]


def test_sql_and_dataframe_filter_agree() -> None:
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("test-sql-analysis")
        .getOrCreate()
    )
    df = spark.createDataFrame(
        [(0, 0), (0, 120), (1, 200), (1, 0), (1, 30)],
        ["label", "pixel406"],
    )

    sql_count, df_count = run_sql_analysis(spark, df)

    assert sql_count == df_count == 3

    assert run_map_reduce_analysis(df) == [(0, 2), (1, 3)]
