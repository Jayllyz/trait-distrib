import numpy as np
import pytest

import postal_app.predictor as predictor_module
from postal_app.predictor import (
    DemoPredictor,
    PredictionError,
    PredictorConfigurationError,
    SparkPredictor,
    get_predictor,
)


def test_demo_predictor_is_deterministic() -> None:
    batch = np.arange(5 * 28 * 28, dtype=np.uint8).reshape(5, 28, 28)
    predictor = DemoPredictor()

    assert predictor.predict(batch) == predictor.predict(batch.copy())
    assert all(0 <= result.digit <= 9 for result in predictor.predict(batch))
    assert all(0.72 <= result.confidence <= 0.99 for result in predictor.predict(batch))


def test_demo_predictor_rejects_incompatible_input() -> None:
    with pytest.raises(ValueError, match=r"\(5, 28, 28\)"):
        DemoPredictor().predict(np.zeros((1, 28, 28), dtype=np.uint8))


def test_spark_mode_loads_the_configured_backend(monkeypatch, tmp_path) -> None:
    preprocessing_path = tmp_path / "preprocessing"
    classifier_path = tmp_path / "classifier"
    sentinel = object()
    calls = []

    monkeypatch.setenv("SPARK_PREPROCESSING_MODEL_PATH", str(preprocessing_path))
    monkeypatch.setenv("SPARK_CLASSIFIER_MODEL_PATH", str(classifier_path))
    monkeypatch.setattr(
        predictor_module,
        "_load_spark_predictor",
        lambda preprocessing, classifier: (
            calls.append((preprocessing, classifier)) or sentinel
        ),
    )

    assert get_predictor("spark") is sentinel
    assert calls == [(str(preprocessing_path), str(classifier_path))]


def test_spark_loader_rejects_missing_artifacts(tmp_path) -> None:
    with pytest.raises(PredictorConfigurationError, match="introuvable"):
        predictor_module._load_spark_predictor(
            str(tmp_path / "missing-preprocessing"),
            str(tmp_path / "missing-classifier"),
        )


def test_unknown_predictor_mode_is_rejected() -> None:
    with pytest.raises(PredictorConfigurationError, match="inconnu"):
        get_predictor("other")


class _FakeSpark:
    def __init__(self) -> None:
        self.rows = None
        self.schema = None

    def createDataFrame(self, rows, schema):
        self.rows = rows
        self.schema = schema
        return object()


class _PassThroughModel:
    def transform(self, frame):
        return frame


class _ResultFrame:
    def __init__(self, rows) -> None:
        self.rows = rows
        self.selected = None
        self.ordered_by = None

    def select(self, *columns):
        self.selected = columns
        return self

    def orderBy(self, column):
        self.ordered_by = column
        self.rows.sort(key=lambda row: row[column])
        return self

    def collect(self):
        return self.rows


class _ClassifierModel:
    def __init__(self, result_frame) -> None:
        self.result_frame = result_frame

    def transform(self, frame):
        return self.result_frame


def test_spark_predictor_preserves_positions_and_probabilities() -> None:
    spark = _FakeSpark()
    digits = np.zeros((5, 28, 28), dtype=np.uint8)
    digits[:, 0, 0] = np.arange(5, dtype=np.uint8)
    expected_digits = (3, 1, 4, 1, 5)
    rows = []
    for position in reversed(range(5)):
        probabilities = [0.01] * 10
        probabilities[expected_digits[position]] = 0.91 - position * 0.01
        rows.append(
            {
                "_position": position,
                "prediction": float(expected_digits[position]),
                "probability": probabilities,
            }
        )
    result_frame = _ResultFrame(rows)
    predictor = SparkPredictor(
        spark,
        _PassThroughModel(),
        _ClassifierModel(result_frame),
    )

    predictions = predictor.predict(digits)

    assert tuple(prediction.digit for prediction in predictions) == expected_digits
    assert [prediction.confidence for prediction in predictions] == pytest.approx(
        [0.91, 0.90, 0.89, 0.88, 0.87]
    )
    assert result_frame.selected == ("_position", "prediction", "probability")
    assert result_frame.ordered_by == "_position"
    assert spark.schema is not None
    assert spark.rows is not None
    assert spark.schema[0] == "_position"
    assert spark.schema[1] == "pixel0"
    assert spark.schema[-1] == "pixel783"
    assert [row[1] for row in spark.rows] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_spark_predictor_wraps_inference_errors() -> None:
    class _FailingSpark:
        def createDataFrame(self, rows, schema):
            raise RuntimeError("Spark is unavailable")

    predictor = SparkPredictor(_FailingSpark(), object(), object())

    with pytest.raises(PredictionError, match="n'a pas pu"):
        predictor.predict(np.zeros((5, 28, 28), dtype=np.uint8))
