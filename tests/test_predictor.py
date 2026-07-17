import numpy as np
import pytest

from postal_app.predictor import (
    DemoPredictor,
    PredictorConfigurationError,
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


def test_spark_mode_fails_explicitly_instead_of_using_demo() -> None:
    with pytest.raises(PredictorConfigurationError, match="Spark"):
        get_predictor("spark")


def test_unknown_predictor_mode_is_rejected() -> None:
    with pytest.raises(PredictorConfigurationError, match="inconnu"):
        get_predictor("other")
