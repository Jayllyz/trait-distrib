import os
import sys

from pyspark.sql import SparkSession


os.environ["PYSPARK_PYTHON"] = sys.executable

DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "digit-recognizer"
)


def fetch_dataset() -> str:
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    train_csv = os.path.join(DATASET_DIR, "train.csv")
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f"Dataset file not found: {train_csv}")
    return DATASET_DIR


def main() -> None:
    try:
        dataset_path = fetch_dataset()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

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
