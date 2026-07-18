import cv2
import numpy as np

from postal_app.predictor import get_predictor


def _synthetic_digit(digit: int) -> np.ndarray:
    image = np.zeros((28, 28), dtype=np.uint8)
    cv2.putText(
        image,
        str(digit),
        (4, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        255,
        2,
        cv2.LINE_AA,
    )
    return image


def test_real_cnn_artifact_produces_five_predictions() -> None:
    batch = np.stack([_synthetic_digit(digit) for digit in (7, 5, 0, 1, 5)])

    predictions = get_predictor("cnn").predict(batch)

    assert len(predictions) == 5
    assert all(0 <= prediction.digit <= 9 for prediction in predictions)
    assert all(0.0 <= prediction.confidence <= 1.0 for prediction in predictions)
