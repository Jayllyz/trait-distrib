from __future__ import annotations

import argparse
import csv
import os
import re
import time
from dataclasses import asdict, dataclass

from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, PCA, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, sum as spark_sum, when

from trait_distrib.spark.session import get_spark
from trait_distrib.config import (
    DATASET_DIR,
    METRICS_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    PREPROCESSING_DIR,
)


PIXEL_RE = re.compile(r"pixel(\d+)")
DEFAULT_EMPTY_PIXEL_THRESHOLD = 0.98
DEFAULT_PCA_COMPONENTS = 50


@dataclass(frozen=True)
class PreprocessingConfig:
    name: str
    normalize: bool = True
    pca_components: int | None = None
    drop_empty_pixels: bool = False
    empty_pixel_threshold: float = DEFAULT_EMPTY_PIXEL_THRESHOLD


def ensure_output_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PREPROCESSING_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)


def get_pixel_columns(df: DataFrame) -> list[str]:
    pixel_columns = [column for column in df.columns if column != "label"]
    pixel_columns.sort(
        key=lambda column: (
            int(PIXEL_RE.search(column).group(1)) if PIXEL_RE.search(column) else 0
        )
    )
    return pixel_columns


def read_digit_csv(spark: SparkSession, csv_path: str, has_label: bool) -> DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    raw_df = spark.read.csv(csv_path, header=True, inferSchema=True)
    if has_label and "label" not in raw_df.columns:
        raise ValueError(f"Missing label column in {csv_path}")

    pixel_columns = [column for column in raw_df.columns if column != "label"]
    ordered_columns = (["label"] if has_label else []) + sorted(
        pixel_columns,
        key=lambda column: (
            int(PIXEL_RE.search(column).group(1)) if PIXEL_RE.search(column) else 0
        ),
    )

    casted_columns = []
    for column in ordered_columns:
        casted_columns.append(col(column).cast("double").alias(column))

    return raw_df.select(*casted_columns)


def read_train_test_frames(spark: SparkSession) -> tuple[DataFrame, DataFrame]:
    train_df = read_digit_csv(spark, os.path.join(DATASET_DIR, "train.csv"), True)
    test_df = read_digit_csv(spark, os.path.join(DATASET_DIR, "test.csv"), False)
    return train_df, test_df


def select_informative_pixels(
    df: DataFrame,
    pixel_columns: list[str],
    empty_pixel_threshold: float,
) -> tuple[list[str], list[str]]:
    total_rows = df.count()
    if total_rows == 0:
        return pixel_columns, []

    zero_counts = df.select(
        *[
            spark_sum(when(col(column) == 0, 1).otherwise(0)).alias(column)
            for column in pixel_columns
        ]
    ).collect()[0]

    selected_columns: list[str] = []
    removed_columns: list[str] = []
    for column in pixel_columns:
        zero_ratio = float(zero_counts[column]) / float(total_rows)
        if zero_ratio < empty_pixel_threshold:
            selected_columns.append(column)
        else:
            removed_columns.append(column)

    return selected_columns, removed_columns


def build_pipeline(
    pixel_columns: list[str],
    normalize: bool,
    pca_components: int | None,
) -> Pipeline:
    stages = []

    assembler_output = (
        "features" if (not normalize and pca_components is None) else "raw_features"
    )
    stages.append(
        VectorAssembler(
            inputCols=pixel_columns,
            outputCol=assembler_output,
            handleInvalid="keep",
        )
    )

    current_features_col = assembler_output
    if normalize:
        scaler_output = "features" if pca_components is None else "scaled_features"
        stages.append(
            MinMaxScaler(
                inputCol=current_features_col,
                outputCol=scaler_output,
            )
        )
        current_features_col = scaler_output

    if pca_components is not None:
        stages.append(
            PCA(
                k=min(pca_components, len(pixel_columns)),
                inputCol=current_features_col,
                outputCol="features",
            )
        )

    return Pipeline(stages=stages)


def prepare_input_frame(df: DataFrame, pixel_columns: list[str]) -> DataFrame:
    selected_columns = (["label"] if "label" in df.columns else []) + pixel_columns
    return df.select(*selected_columns).na.fill(0)


def transform_and_save_frame(
    model,
    df: DataFrame,
    output_path: str,
    has_label: bool,
) -> tuple[int, int]:
    transformed = model.transform(df)
    keep_columns = ["features"]
    if has_label:
        keep_columns.insert(0, "label")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    transformed.select(*keep_columns).write.mode("overwrite").parquet(output_path)
    return transformed.count(), len(keep_columns)


def run_configuration(
    spark: SparkSession,
    train_df: DataFrame,
    test_df: DataFrame,
    config: PreprocessingConfig,
) -> dict[str, object]:
    pixel_columns = get_pixel_columns(train_df)
    kept_pixels = pixel_columns
    removed_pixels: list[str] = []

    if config.drop_empty_pixels:
        kept_pixels, removed_pixels = select_informative_pixels(
            train_df,
            pixel_columns,
            config.empty_pixel_threshold,
        )

    prepared_train = prepare_input_frame(train_df, kept_pixels).cache()
    prepared_test = prepare_input_frame(test_df, kept_pixels).cache()
    prepared_train.count()
    prepared_test.count()

    pipeline = build_pipeline(
        pixel_columns=kept_pixels,
        normalize=config.normalize,
        pca_components=config.pca_components,
    )
    final_feature_dimension = (
        min(config.pca_components, len(kept_pixels))
        if config.pca_components is not None
        else len(kept_pixels)
    )

    start_time = time.perf_counter()
    model = pipeline.fit(prepared_train)
    train_rows, train_columns = transform_and_save_frame(
        model,
        prepared_train,
        os.path.join(PREPROCESSING_DIR, config.name, "train.parquet"),
        has_label=True,
    )
    test_rows, test_columns = transform_and_save_frame(
        model,
        prepared_test,
        os.path.join(PREPROCESSING_DIR, config.name, "test.parquet"),
        has_label=False,
    )
    duration_seconds = time.perf_counter() - start_time

    model_path = os.path.join(MODELS_DIR, config.name)
    model.write().overwrite().save(model_path)

    prepared_train.unpersist()
    prepared_test.unpersist()

    return {
        **asdict(config),
        "selected_pixels": len(kept_pixels),
        "removed_pixels": len(removed_pixels),
        "feature_dimension": final_feature_dimension,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_output_columns": train_columns,
        "test_output_columns": test_columns,
        "duration_seconds": round(duration_seconds, 3),
        "model_path": model_path,
        "train_output_path": os.path.join(
            PREPROCESSING_DIR, config.name, "train.parquet"
        ),
        "test_output_path": os.path.join(
            PREPROCESSING_DIR, config.name, "test.parquet"
        ),
    }


def save_summary(rows: list[dict[str, object]]) -> None:
    summary_path = os.path.join(METRICS_DIR, "preprocessing_summary.csv")
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_preprocessing_workflow() -> None:
    ensure_output_dirs()

    spark = get_spark("trait-distrib-preprocessing")
    spark.sparkContext.setLogLevel("WARN")

    train_df, test_df = read_train_test_frames(spark)
    train_df.cache()
    test_df.cache()
    train_df.count()
    test_df.count()

    configs = [
        PreprocessingConfig(name="raw_pixels", normalize=False),
        PreprocessingConfig(name="normalized_pixels", normalize=True),
        PreprocessingConfig(
            name="normalized_compact_pixels",
            normalize=True,
            drop_empty_pixels=True,
            empty_pixel_threshold=DEFAULT_EMPTY_PIXEL_THRESHOLD,
        ),
        PreprocessingConfig(
            name="normalized_compact_pca",
            normalize=True,
            drop_empty_pixels=True,
            empty_pixel_threshold=DEFAULT_EMPTY_PIXEL_THRESHOLD,
            pca_components=DEFAULT_PCA_COMPONENTS,
        ),
    ]

    summary_rows = []
    for config in configs:
        print(f"[preprocessing] Running {config.name}...")
        summary_rows.append(run_configuration(spark, train_df, test_df, config))

    save_summary(summary_rows)
    train_df.unpersist()
    test_df.unpersist()
    spark.stop()

    print(f"[preprocessing] Artifacts saved in {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Spark preprocessing workflow."
    )
    parser.parse_args()
    run_preprocessing_workflow()


if __name__ == "__main__":
    main()
