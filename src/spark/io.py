import os

from pyspark.sql import DataFrame, SparkSession

from src.config import DATASET_DIR


def fetch_dataset() -> str:
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    train_csv = os.path.join(DATASET_DIR, "train.csv")
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f"Dataset file not found: {train_csv}")
    return DATASET_DIR


def load_csv(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.csv(path, header=True, inferSchema=True)


def load_train(spark: SparkSession) -> DataFrame:
    return load_csv(spark, os.path.join(fetch_dataset(), "train.csv"))
