import os
import sys

from pyspark.sql import SparkSession


os.environ["PYSPARK_PYTHON"] = sys.executable


def main() -> None:
    spark = (
        SparkSession.builder.appName("trait-distrib").master("local[1]").getOrCreate()
    )
    df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "val"])
    df.show()
    spark.stop()


if __name__ == "__main__":
    main()
