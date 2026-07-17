import pytest

from postal_app.domain import DigitPrediction, PostalAnalysis


def _predictions(confidences: list[float]) -> tuple[DigitPrediction, ...]:
    return tuple(
        DigitPrediction(digit=index, confidence=confidence)
        for index, confidence in enumerate(confidences)
    )


def test_postal_analysis_uses_weakest_digit_as_global_confidence() -> None:
    analysis = PostalAnalysis(_predictions([0.98, 0.95, 0.99, 0.91, 0.97]))

    assert analysis.postal_code == "01234"
    assert analysis.global_confidence == pytest.approx(0.91)
    assert analysis.decision == "Tri automatique"
    assert analysis.review_reason is None


def test_threshold_is_inclusive_for_automatic_sorting() -> None:
    analysis = PostalAnalysis(_predictions([0.95, 0.8, 0.91, 0.9, 0.88]))

    assert analysis.requires_review is False


def test_low_confidence_requires_review_and_identifies_position() -> None:
    analysis = PostalAnalysis(_predictions([0.95, 0.91, 0.72, 0.9, 0.88]))

    assert analysis.requires_review is True
    assert analysis.decision == "Vérification humaine"
    assert analysis.review_reason == (
        "Le chiffre en position 3 n'atteint que 72% de confiance."
    )


def test_postal_analysis_requires_five_digits() -> None:
    with pytest.raises(ValueError, match="exactly five"):
        PostalAnalysis(_predictions([0.9, 0.9]))
