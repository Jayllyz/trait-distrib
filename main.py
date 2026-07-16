import os
import sys

import kagglehub
from pyspark.sql import SparkSession


os.environ["PYSPARK_PYTHON"] = sys.executable

COMPETITION = "digit-recognizer"


def fetch_dataset() -> str:
    return kagglehub.competition_download(COMPETITION)


def main() -> None:
    dataset_path = fetch_dataset()

    spark = (
        SparkSession.builder.appName("trait-distrib").master("local[1]").getOrCreate()
    )
    train_df = spark.read.csv(
        os.path.join(dataset_path, "train.csv"), header=True, inferSchema=True
    )
    train_df.printSchema()
    train_df.show(5)
    spark.stop()


if __name__ == "__main__":
    main()
