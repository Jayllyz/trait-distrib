"""Prediction contract and explicit demonstration implementation."""

import hashlib
import os
from typing import Protocol

import numpy as np

from postal_app.domain import DigitPrediction, PostalAnalysis


class PredictorConfigurationError(RuntimeError):
    """Raised when the selected prediction backend is unavailable."""


class Predictor(Protocol):
    """Contract that the future Spark model adapter must implement."""

    @property
    def is_demo(self) -> bool: ...

    def predict(self, digits: np.ndarray) -> tuple[DigitPrediction, ...]:
        """Predict five uint8 images shaped (5, 28, 28)."""
        ...


class DemoPredictor:
    """Deterministic fake predictor used only to exercise the user journey."""

    is_demo = True

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


def get_predictor(mode: str | None = None) -> Predictor:
    """Build the configured backend without silently falling back to demo mode."""

    selected_mode = (mode or os.getenv("PREDICTOR_MODE", "demo")).strip().lower()
    if selected_mode == "demo":
        return DemoPredictor()
    if selected_mode in {"spark", "real"}:
        raise PredictorConfigurationError(
            "Le prédicteur Spark n'est pas encore configuré. "
            "Ajoutez l'adaptateur du modèle avant d'utiliser PREDICTOR_MODE=spark."
        )
    raise PredictorConfigurationError(
        f"Mode de prédiction inconnu : {selected_mode!r}. Utilisez 'demo' ou 'spark'."
    )


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
