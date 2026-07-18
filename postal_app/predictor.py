"""Prediction contract and demonstration and Spark implementations."""

from functools import lru_cache
import hashlib
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Protocol

import numpy as np

from postal_app.domain import DigitPrediction, PostalAnalysis
from src.config import (
    BEST_CLASSIFIER_MANIFEST,
    BEST_CLASSIFIER_PREFIX,
    CLASSIFIER_MODELS_DIR,
    MODELS_DIR,
    PRODUCTION_PREPROCESSING_MODEL_NAME,
)


logger = logging.getLogger(__name__)


class PredictorConfigurationError(RuntimeError):
    """Raised when the selected prediction backend is unavailable."""


class PredictionError(RuntimeError):
    """Raised when a configured backend cannot complete an inference."""


class Predictor(Protocol):
    """Contract shared by prediction backends."""

    def predict(self, digits: np.ndarray) -> tuple[DigitPrediction, ...]:
        """Predict five uint8 images shaped (5, 28, 28)."""
        ...


class DemoPredictor:
    """Deterministic fake predictor used only to exercise the user journey."""

    def predict(self, digits: np.ndarray) -> tuple[DigitPrediction, ...]:
        _validate_batch(digits)
        predictions: list[DigitPrediction] = []
        for position, digit_image in enumerate(digits):
            digest = hashlib.sha256(
                digit_image.tobytes() + position.to_bytes(1, "big")
            ).digest()
            digit = digest[0] % 10
            confidence = 0.72 + (digest[1] / 255) * 0.27
            predictions.append(DigitPrediction(digit=digit, confidence=confidence))
        return tuple(predictions)


class SparkPredictor:
    """Adapter from five MNIST-like images to persisted Spark ML models."""

    def __init__(
        self,
        spark: Any,
        preprocessing_model: Any,
        classifier_model: Any,
    ) -> None:
        self._spark = spark
        self._preprocessing_model = preprocessing_model
        self._classifier_model = classifier_model

    def predict(self, digits: np.ndarray) -> tuple[DigitPrediction, ...]:
        _validate_batch(digits)
        columns = ["_position", *(f"pixel{index}" for index in range(28 * 28))]
        rows = [
            (position, *(float(value) for value in digit_image.reshape(-1)))
            for position, digit_image in enumerate(digits)
        ]

        try:
            frame = self._spark.createDataFrame(rows, schema=columns)
            prepared = self._preprocessing_model.transform(frame)
            predictions = self._classifier_model.transform(prepared)
            result_rows = (
                predictions.select("_position", "prediction", "probability")
                .orderBy("_position")
                .collect()
            )
        except Exception as error:
            raise PredictionError(
                "Le modèle Spark n'a pas pu analyser les chiffres."
            ) from error

        if len(result_rows) != 5:
            raise PredictionError(
                "Le modèle Spark n'a pas retourné les cinq prédictions attendues."
            )

        converted: list[DigitPrediction] = []
        for row in result_rows:
            digit = int(row["prediction"])
            if not 0 <= digit <= 9:
                raise PredictionError("Le modèle Spark a retourné un chiffre invalide.")
            probability = row["probability"]
            try:
                confidence = float(probability[digit])
                converted.append(DigitPrediction(digit=digit, confidence=confidence))
            except (IndexError, KeyError, TypeError, ValueError) as error:
                raise PredictionError(
                    "Le modèle Spark a retourné une probabilité invalide."
                ) from error
        return tuple(converted)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESSING_MODEL_PATH = (
    Path(MODELS_DIR) / PRODUCTION_PREPROCESSING_MODEL_NAME
)
_SPARK_PREDICTOR_LOCK = Lock()


def get_predictor(mode: str | None = None) -> Predictor:
    """Build the configured backend without silently falling back to demo mode."""

    selected_mode = (mode or os.getenv("PREDICTOR_MODE", "demo")).strip().lower()
    if selected_mode == "demo":
        return DemoPredictor()
    if selected_mode in {"spark", "real"}:
        preprocessing_path = _resolve_model_path(
            os.getenv("SPARK_PREPROCESSING_MODEL_PATH"),
            lambda: DEFAULT_PREPROCESSING_MODEL_PATH,
        )
        classifier_path = _resolve_model_path(
            os.getenv("SPARK_CLASSIFIER_MODEL_PATH"),
            _default_classifier_model_path,
        )
        # lru_cache alone can evaluate the same missing key concurrently. The
        # outer lock guarantees one JVM/model initialization for Streamlit.
        with _SPARK_PREDICTOR_LOCK:
            return _load_spark_predictor(preprocessing_path, classifier_path)
    raise PredictorConfigurationError(
        f"Mode de prédiction inconnu : {selected_mode!r}. Utilisez 'demo' ou 'spark'."
    )


def _resolve_model_path(configured: str | None, default: Callable[[], Path]) -> str:
    path = Path(configured).expanduser() if configured else default()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def _default_classifier_model_path() -> Path:
    try:
        model_name = Path(BEST_CLASSIFIER_MANIFEST).read_text().strip()
    except FileNotFoundError as error:
        raise PredictorConfigurationError(
            "Aucun modèle entraîné : "
            f"manifest introuvable ({BEST_CLASSIFIER_MANIFEST}). "
            "Lancez d'abord l'entraînement (src/ml/training.py)."
        ) from error
    return Path(CLASSIFIER_MODELS_DIR) / f"{BEST_CLASSIFIER_PREFIX}{model_name}"


def _require_existing_model_paths(
    preprocessing_path: str, classifier_path: str
) -> None:
    missing_paths = [
        path
        for path in (preprocessing_path, classifier_path)
        if not Path(path).is_dir()
    ]
    if missing_paths:
        raise PredictorConfigurationError(
            "Artefact(s) du modèle Spark introuvable(s) : " + ", ".join(missing_paths)
        )


def _build_spark_predictor(
    preprocessing_path: str, classifier_path: str
) -> SparkPredictor:
    spark = None
    try:
        from pyspark.ml import PipelineModel
        from pyspark.ml.classification import RandomForestClassificationModel
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.appName("trait-distrib-streamlit")
            .master("local[1]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "1")
            # Git can normalize line endings in Spark's JSON metadata without
            # updating Hadoop's adjacent .crc files. These packaged artifacts
            # are read-only, so checksum sidecars add no value at inference.
            .config(
                "spark.hadoop.fs.file.impl",
                "org.apache.hadoop.fs.RawLocalFileSystem",
            )
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        preprocessing_model = PipelineModel.load(preprocessing_path)
        classifier_model = RandomForestClassificationModel.load(classifier_path)
    except Exception as error:
        logger.exception(
            "Failed to initialize the Spark predictor from preprocessing=%s and classifier=%s",
            preprocessing_path,
            classifier_path,
        )
        if spark is not None:
            spark.stop()
        raise PredictorConfigurationError(
            "Impossible de charger les artefacts du modèle Spark."
        ) from error

    return SparkPredictor(spark, preprocessing_model, classifier_model)


@lru_cache(maxsize=4)
def _load_spark_predictor(
    preprocessing_path: str,
    classifier_path: str,
) -> SparkPredictor:
    _require_existing_model_paths(preprocessing_path, classifier_path)
    return _build_spark_predictor(preprocessing_path, classifier_path)


def analyze_digits(
    digits: np.ndarray, predictor: Predictor, threshold: float = 0.8
) -> PostalAnalysis:
    predictions = predictor.predict(digits)
    return PostalAnalysis(
        predictions=predictions,
        automatic_sort_threshold=threshold,
    )


def _validate_batch(digits: np.ndarray) -> None:
    if digits.shape != (5, 28, 28):
        raise ValueError("the predictor expects a batch shaped (5, 28, 28)")
    if digits.dtype != np.uint8:
        raise ValueError("the predictor expects uint8 pixel values")
