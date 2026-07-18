import os

import pytest

from src.stats import descriptive


pytestmark = [
    pytest.mark.spark,
    pytest.mark.skipif(
        os.name == "nt",
        reason="Spark inference is supported through the Linux Docker image.",
    ),
]


def test_run_sql_analysis_matches_dataframe_filter(tmp_path, monkeypatch) -> None:
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("test-sql-analysis")
        .getOrCreate()
    )
    df = spark.createDataFrame(
        [(0, 0), (0, 120), (1, 200), (1, 0), (1, 30)],
        ["label", descriptive.CENTER_PIXEL],
    )
    monkeypatch.setattr(descriptive, "STATS_DIR", str(tmp_path))
    monkeypatch.setattr(descriptive, "ensure_dirs", lambda: None)

    data = descriptive.run_sql_analysis(spark, df)

    assert [row["label"] for row in data] == [0, 1]
    assert [row["n"] for row in data] == [2, 3]
    assert data[0]["avg_center_ink"] == 60.0
    assert (tmp_path / "sql_label_stats.csv").exists()
