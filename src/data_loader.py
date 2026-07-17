"""Chargement des données (Séance 3 du cours : formats CSV, JSON, Parquet)."""

import os
import sys

from pyspark.sql import DataFrame, SparkSession

from config import DATASET_DIR


def get_spark(app_name: str = "trait-distrib") -> SparkSession:
    """SparkSession en mode local threadé (pas de broadcast/repartition utiles ici)."""
    os.environ["PYSPARK_PYTHON"] = sys.executable
    return SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()


def fetch_dataset() -> str:
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    train_csv = os.path.join(DATASET_DIR, "train.csv")
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f"Dataset file not found: {train_csv}")
    return DATASET_DIR


def load_csv(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.csv(path, header=True, inferSchema=True)


def load_json(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.json(path)


def load_parquet(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.parquet(path)


def write_parquet(df: DataFrame, path: str) -> None:
    df.write.mode("overwrite").parquet(path)


def load_train(spark: SparkSession) -> DataFrame:
    return load_csv(spark, os.path.join(fetch_dataset(), "train.csv"))


def load_test(spark: SparkSession) -> DataFrame:
    return load_csv(spark, os.path.join(fetch_dataset(), "test.csv"))
