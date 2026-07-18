from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Any, cast

import numpy as np
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark import RDD
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql.functions import col

from src.config import (
    BEST_CLASSIFIER_MANIFEST,
    BEST_CLASSIFIER_PREFIX,
    CLASSIFIER_MODELS_DIR,
    METRICS_DIR,
    OUTPUT_DIR,
    PRODUCTION_PREPROCESSING_MODEL_NAME,
)
from src.spark.session import get_spark
from src.ml.preprocessing import (
    DEFAULT_EMPTY_PIXEL_THRESHOLD,
    DEFAULT_PCA_COMPONENTS,
    PreprocessingConfig,
    ensure_output_dirs,
    read_train_test_frames,
    run_configuration,
)


TARGET_PREPROCESSING_CONFIG = PreprocessingConfig(
    name=PRODUCTION_PREPROCESSING_MODEL_NAME,
    normalize=True,
    drop_empty_pixels=True,
    empty_pixel_threshold=DEFAULT_EMPTY_PIXEL_THRESHOLD,
    pca_components=DEFAULT_PCA_COMPONENTS,
)


def ensure_classifier_dirs() -> None:
    os.makedirs(CLASSIFIER_MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)


def load_or_build_preprocessed_train_frame(spark: SparkSession) -> DataFrame:
    train_path = os.path.join(
        OUTPUT_DIR, "preprocessing", TARGET_PREPROCESSING_CONFIG.name, "train.parquet"
    )

    if not os.path.isdir(train_path):
        train_df, test_df = read_train_test_frames(spark)
        run_configuration(spark, train_df, test_df, TARGET_PREPROCESSING_CONFIG)

    return spark.read.parquet(train_path).select(
        col("label").cast("double").alias("label"), col("features")
    )


def get_feature_dimension(df: DataFrame) -> int:
    sample_row = df.select("features").head(1)
    if not sample_row:
        raise ValueError("No feature rows available for training.")
    return int(sample_row[0]["features"].size)


def build_classifiers() -> list[tuple[str, Any]]:
    return [
        (
            "logistic_regression",
            LogisticRegression(
                featuresCol="features",
                labelCol="label",
                predictionCol="prediction",
                probabilityCol="probability",
                rawPredictionCol="rawPrediction",
                family="multinomial",
                maxIter=30,
                regParam=0.08,
                elasticNetParam=0.0,
            ),
        ),
        (
            "decision_tree",
            DecisionTreeClassifier(
                featuresCol="features",
                labelCol="label",
                predictionCol="prediction",
                probabilityCol="probability",
                rawPredictionCol="rawPrediction",
                maxDepth=12,
                minInstancesPerNode=2,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                featuresCol="features",
                labelCol="label",
                predictionCol="prediction",
                probabilityCol="probability",
                rawPredictionCol="rawPrediction",
                numTrees=40,
                maxDepth=12,
                featureSubsetStrategy="sqrt",
                seed=42,
            ),
        ),
    ]


def compute_class_metrics(predictions: DataFrame) -> list[dict[str, Any]]:
    metrics = MulticlassMetrics(
        cast("RDD[Row]", predictions.select("prediction", "label").rdd).map(
            lambda row: (float(row[0]), float(row[1]))
        )
    )

    class_metrics = []
    for label in range(10):
        label_value = float(label)
        class_metrics.append(
            {
                "label": label,
                "precision": float(metrics.precision(label_value)),
                "recall": float(metrics.recall(label_value)),
                "f1": float(metrics.fMeasure(label_value)),
            }
        )
    return class_metrics


def compute_confusion_matrix(predictions: DataFrame) -> np.ndarray:
    metrics = MulticlassMetrics(
        cast("RDD[Row]", predictions.select("prediction", "label").rdd).map(
            lambda row: (float(row[0]), float(row[1]))
        )
    )
    return metrics.confusionMatrix().toArray()


def save_confusion_matrix(matrix: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"] + [str(index) for index in range(matrix.shape[1])])
        for index, row in enumerate(matrix):
            writer.writerow([index] + [int(value) for value in row])


def save_class_metrics(rows: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["label", "precision", "recall", "f1"]
        )
        writer.writeheader()
        writer.writerows(rows)


def save_model_comparison(rows: list[dict[str, Any]]) -> None:
    comparison_path = os.path.join(METRICS_DIR, "model_comparison.csv")
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(comparison_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "model",
        "accuracy",
        "f1",
        "weighted_precision",
        "weighted_recall",
        "train_s",
        "eval_s",
    ]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row[header])))

    def format_row(row: dict[str, Any]) -> str:
        return " | ".join(str(row[header]).ljust(widths[header]) for header in headers)

    print()
    print(format_row({header: header for header in headers}))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        display_row = dict(row)
        for key in ("accuracy", "f1", "weighted_precision", "weighted_recall"):
            display_row[key] = f"{float(display_row[key]):.4f}"
        display_row["train_s"] = f"{float(display_row['train_s']):.2f}"
        display_row["eval_s"] = f"{float(display_row['eval_s']):.2f}"
        print(format_row(display_row))
    print()


def train_and_evaluate_models(
    train_df: DataFrame, val_df: DataFrame, feature_dimension: int
):
    evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    evaluator_weighted_precision = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
    )
    evaluator_weighted_recall = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedRecall"
    )

    results: list[dict[str, Any]] = []
    models: dict[str, Any] = {}

    for model_name, estimator in build_classifiers():
        print(f"[training] Fitting {model_name}...")
        start_train = time.perf_counter()
        try:
            model = estimator.fit(train_df)
            train_s = time.perf_counter() - start_train

            start_eval = time.perf_counter()
            predictions = model.transform(val_df).cache()
            predictions.count()
            eval_s = time.perf_counter() - start_eval

            accuracy = evaluator_accuracy.evaluate(predictions)
            f1 = evaluator_f1.evaluate(predictions)
            weighted_precision = evaluator_weighted_precision.evaluate(predictions)
            weighted_recall = evaluator_weighted_recall.evaluate(predictions)
            class_metrics = compute_class_metrics(predictions)
            confusion_matrix = compute_confusion_matrix(predictions)

            save_class_metrics(
                class_metrics,
                os.path.join(METRICS_DIR, f"class_metrics_{model_name}.csv"),
            )
            save_confusion_matrix(
                confusion_matrix,
                os.path.join(METRICS_DIR, f"confusion_matrix_{model_name}.csv"),
            )

            results.append(
                {
                    "model": model_name,
                    "accuracy": round(float(accuracy), 6),
                    "f1": round(float(f1), 6),
                    "weighted_precision": round(float(weighted_precision), 6),
                    "weighted_recall": round(float(weighted_recall), 6),
                    "train_s": round(float(train_s), 3),
                    "eval_s": round(float(eval_s), 3),
                    "status": "ok",
                }
            )
            models[model_name] = model
            predictions.unpersist()
        except Exception as exc:  # pragma: no cover - keep the workflow resilient
            train_s = time.perf_counter() - start_train
            print(f"[training][error] {model_name}: {exc}")
            results.append(
                {
                    "model": model_name,
                    "accuracy": 0.0,
                    "f1": 0.0,
                    "weighted_precision": 0.0,
                    "weighted_recall": 0.0,
                    "train_s": round(float(train_s), 3),
                    "eval_s": 0.0,
                    "status": f"failed: {exc}",
                }
            )

    return results, models


def pick_best_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful_rows = [row for row in rows if row["status"] == "ok"]
    if not successful_rows:
        failures = "; ".join(str(row["status"]) for row in rows)
        raise RuntimeError(f"No model completed successfully. {failures}")
    return sorted(
        successful_rows,
        key=lambda row: (float(row["f1"]), float(row["accuracy"])),
        reverse=True,
    )[0]


def save_best_model(model: Any, model_name: str) -> str:
    model_path = os.path.join(
        CLASSIFIER_MODELS_DIR, f"{BEST_CLASSIFIER_PREFIX}{model_name}"
    )
    os.makedirs(model_path, exist_ok=True)
    model.write().overwrite().save(model_path)
    with open(BEST_CLASSIFIER_MANIFEST, "w") as manifest:
        manifest.write(model_name)
    return model_path


def run_machine_learning_workflow(sample_fraction: float) -> None:
    ensure_output_dirs()
    ensure_classifier_dirs()

    spark = get_spark("trait-distrib-machine-learning")
    try:
        spark.sparkContext.setLogLevel("WARN")

        prepared_df = load_or_build_preprocessed_train_frame(spark)
        if not 0 < sample_fraction <= 1:
            raise ValueError("sample_fraction must be in the (0, 1] interval")

        working_df = prepared_df.sample(
            withReplacement=False, fraction=sample_fraction, seed=42
        )
        working_df = working_df.cache()
        working_df.count()

        feature_dimension = get_feature_dimension(working_df)
        print(f"[training] Feature dimension: {feature_dimension}")

        train_df, val_df = working_df.randomSplit([0.8, 0.2], seed=42)
        train_df.cache()
        val_df.cache()
        train_df.count()
        val_df.count()

        results, models = train_and_evaluate_models(train_df, val_df, feature_dimension)
        save_model_comparison(results)
        print_table(results)

        best_row = pick_best_model(results)
        best_model_name = str(best_row["model"])
        best_model = models[best_model_name]
        best_model_path = save_best_model(best_model, best_model_name)

        print(f"Meilleur modèle: {best_model_name}")
        print(
            f"Accuracy={float(best_row['accuracy']):.4f} | F1={float(best_row['f1']):.4f} | "
            f"Précision pondérée={float(best_row['weighted_precision']):.4f} | "
            f"Rappel pondéré={float(best_row['weighted_recall']):.4f}"
        )
        print(f"Modèle sauvegardé: {best_model_path}")
        print(
            f"Résultats détaillés: {os.path.join(METRICS_DIR, 'model_comparison.csv')}"
        )
        print(
            f"Matrice de confusion: {os.path.join(METRICS_DIR, f'confusion_matrix_{best_model_name}.csv')}"
        )
        print(
            f"Métriques par classe: {os.path.join(METRICS_DIR, f'class_metrics_{best_model_name}.csv')}"
        )

        train_df.unpersist()
        val_df.unpersist()
        working_df.unpersist()
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Spark ML models.")
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.2,
        help="Fraction of the preprocessed training set used for step 4.",
    )
    args = parser.parse_args()
    run_machine_learning_workflow(sample_fraction=args.sample_fraction)


if __name__ == "__main__":
    main()
