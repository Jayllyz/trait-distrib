"""PyTorch architecture shared by CNN training and inference."""

from typing import Any

from torch import Tensor, nn


CNN_FORMAT_VERSION = 1


class DigitCNN(nn.Module):
    """Compact convolutional network for MNIST-shaped digit images."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.features(images))


def build_cnn_artifact(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Build the versioned, weights-only artifact persisted after training."""

    return {
        "format_version": CNN_FORMAT_VERSION,
        "architecture": "digit_cnn_conv32_conv64_dense128",
        "classes": list(range(10)),
        "input_shape": [1, 28, 28],
        "input_scale": 255.0,
        "state_dict": state_dict,
    }
