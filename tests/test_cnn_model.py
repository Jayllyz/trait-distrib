import torch

from postal_app.cnn_model import CNN_FORMAT_VERSION, DigitCNN, build_cnn_artifact


def test_digit_cnn_returns_ten_logits_per_image() -> None:
    model = DigitCNN()

    logits = model(torch.zeros((5, 1, 28, 28), dtype=torch.float32))

    assert logits.shape == (5, 10)


def test_cnn_artifact_is_versioned_and_describes_input() -> None:
    model = DigitCNN()

    artifact = build_cnn_artifact(model.state_dict())

    assert artifact["format_version"] == CNN_FORMAT_VERSION
    assert artifact["input_shape"] == [1, 28, 28]
    assert artifact["classes"] == list(range(10))
