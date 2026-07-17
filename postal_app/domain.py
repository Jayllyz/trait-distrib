"""Domain objects shared by the UI and prediction implementations."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DigitPrediction:
    """Prediction returned for one handwritten digit."""

    digit: int
    confidence: float

    def __post_init__(self) -> None:
        if not 0 <= self.digit <= 9:
            raise ValueError("digit must be between 0 and 9")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class PostalAnalysis:
    """Business result for a five-digit French postal code."""

    predictions: tuple[DigitPrediction, ...]
    automatic_sort_threshold: float = 0.8

    def __post_init__(self) -> None:
        if len(self.predictions) != 5:
            raise ValueError("a postal analysis requires exactly five predictions")
        if not 0.0 <= self.automatic_sort_threshold <= 1.0:
            raise ValueError("automatic_sort_threshold must be between 0 and 1")

    @property
    def postal_code(self) -> str:
        return "".join(str(prediction.digit) for prediction in self.predictions)

    @property
    def global_confidence(self) -> float:
        return min(prediction.confidence for prediction in self.predictions)

    @property
    def requires_review(self) -> bool:
        return self.global_confidence < self.automatic_sort_threshold

    @property
    def decision(self) -> str:
        if self.requires_review:
            return "Vérification humaine"
        return "Tri automatique"

    @property
    def review_reason(self) -> str | None:
        if not self.requires_review:
            return None
        weakest_index, weakest = min(
            enumerate(self.predictions), key=lambda item: item[1].confidence
        )
        return (
            f"Le chiffre en position {weakest_index + 1} n'atteint que "
            f"{weakest.confidence:.0%} de confiance."
        )
