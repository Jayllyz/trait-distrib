import os
import sys

from pyspark.sql import SparkSession


def get_spark(app_name: str = "trait-distrib") -> SparkSession:
    os.environ["PYSPARK_PYTHON"] = sys.executable
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
