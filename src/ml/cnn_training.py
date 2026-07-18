"""Train the PyTorch digit CNN through PySpark TorchDistributor."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import pyarrow.dataset as arrow_dataset
from pyspark.ml.torch.distributor import TorchDistributor
from pyspark.sql import DataFrame
from pyspark.sql.functions import array, col, lit, pmod, xxhash64
import torch
from torch import Tensor, nn
from torch.nn import functional as torch_functional
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import GaussianBlur, RandomAffine, RandomApply

from postal_app.cnn_model import DigitCNN, build_cnn_artifact
from src.config import DATASET_DIR, METRICS_DIR, MODELS_DIR, PROJECT_ROOT
from src.ml.preprocessing import get_pixel_columns, read_digit_csv
from src.spark.session import get_spark


DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MINIMUM_ACCURACY = 0.98
EARLY_STOPPING_PATIENCE = 3
SEED = 42
CNN_MODEL_DIR = Path(MODELS_DIR) / "cnn"
CNN_MODEL_PATH = CNN_MODEL_DIR / "digit_cnn.pt"
CNN_METRICS_PATH = Path(METRICS_DIR) / "cnn_metrics.json"
CNN_CONFUSION_MATRIX_PATH = Path(METRICS_DIR) / "confusion_matrix_cnn.csv"


class DigitDataset(Dataset[tuple[Tensor, Tensor]]):
    """In-memory digit dataset with photo-like training augmentation."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool) -> None:
        self._images = torch.from_numpy(images).reshape(-1, 1, 28, 28)
        self._labels = torch.from_numpy(labels.astype(np.int64, copy=False))
        self._augment = augment
        self._spatial_augmentation = torch.nn.Sequential(
            RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.2),
        )

    def __len__(self) -> int:
        return int(self._labels.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image = self._images[index].to(dtype=torch.float32).div_(255.0)
        if self._augment:
            image = self._spatial_augmentation(image)
            morphology_choice = float(torch.rand(()))
            if morphology_choice < 0.15:
                image = torch_functional.max_pool2d(image, 3, stride=1, padding=1)
            elif morphology_choice < 0.25:
                image = -torch_functional.max_pool2d(-image, 3, stride=1, padding=1)
            if float(torch.rand(())) < 0.2:
                image = (image + torch.randn_like(image) * 0.025).clamp_(0.0, 1.0)
        return image, self._labels[index]


def _load_parquet_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    table = arrow_dataset.dataset(path, format="parquet").to_table()
    frame = table.to_pandas()
    labels = frame.pop("label").to_numpy(dtype=np.int64, copy=True)
    images = frame.to_numpy(dtype=np.uint8, copy=True)
    return images, labels


def _evaluate(
    model: nn.Module, loader: DataLoader[tuple[Tensor, Tensor]], criterion: nn.Module
) -> tuple[float, float, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    example_count = 0
    confusion = np.zeros((10, 10), dtype=np.int64)
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            batch_size = int(labels.shape[0])
            loss_sum += float(loss) * batch_size
            example_count += batch_size
            for actual, predicted in zip(
                labels.tolist(), predictions.tolist(), strict=True
            ):
                confusion[int(actual), int(predicted)] += 1
    accuracy = float(np.trace(confusion) / max(confusion.sum(), 1))
    return loss_sum / max(example_count, 1), accuracy, confusion


def _class_metrics(confusion: np.ndarray) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for label in range(10):
        true_positive = int(confusion[label, label])
        predicted_count = int(confusion[:, label].sum())
        actual_count = int(confusion[label, :].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / actual_count if actual_count else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def train_cnn_worker(
    train_path: str,
    validation_path: str,
    model_path: str,
    metrics_path: str,
    confusion_matrix_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    minimum_accuracy: float,
) -> dict[str, Any]:
    """Training function executed in the TorchDistributor worker process."""

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    train_images, train_labels = _load_parquet_dataset(train_path)
    validation_images, validation_labels = _load_parquet_dataset(validation_path)
    train_loader = DataLoader(
        DigitDataset(train_images, train_labels, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        DigitDataset(validation_images, validation_labels, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = DigitCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_accuracy = -1.0
    best_state: dict[str, Any] | None = None
    best_confusion = np.zeros((10, 10), dtype=np.int64)
    history: list[dict[str, float | int]] = []
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_examples = int(labels.shape[0])
            train_loss_sum += float(loss.detach()) * batch_examples
            train_count += batch_examples

        validation_loss, validation_accuracy, confusion = _evaluate(
            model, validation_loader, criterion
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(epoch_metrics)
        print(
            f"[cnn] epoch={epoch} train_loss={epoch_metrics['train_loss']:.4f} "
            f"validation_loss={validation_loss:.4f} "
            f"validation_accuracy={validation_accuracy:.4f}"
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = copy.deepcopy(model.state_dict())
            best_confusion = confusion.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("CNN training did not produce model weights")

    model_output = Path(model_path)
    metrics_output = Path(metrics_path)
    confusion_output = Path(confusion_matrix_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    confusion_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_cnn_artifact(best_state), model_output)

    metrics: dict[str, Any] = {
        "model": "cnn",
        "train_samples": int(train_labels.size),
        "validation_samples": int(validation_labels.size),
        "best_validation_accuracy": best_accuracy,
        "epochs_completed": len(history),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "history": history,
        "class_metrics": _class_metrics(best_confusion),
    }
    metrics_output.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with confusion_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", *range(10)])
        for label, row in enumerate(best_confusion):
            writer.writerow([label, *(int(value) for value in row)])

    if best_accuracy < minimum_accuracy:
        raise RuntimeError(
            f"CNN validation accuracy {best_accuracy:.4f} is below "
            f"the required {minimum_accuracy:.4f}"
        )
    return metrics


def _prepare_split(frame: DataFrame) -> tuple[DataFrame, DataFrame]:
    pixel_columns = get_pixel_columns(frame)
    split_bucket = pmod(
        xxhash64(col("label"), array(*(col(name) for name in pixel_columns))), lit(5)
    )
    compact_columns = [
        col("label").cast("short").alias("label"),
        *(col(name).cast("short").alias(name) for name in pixel_columns),
    ]
    prepared = frame.withColumn("_split_bucket", split_bucket)
    train_frame = prepared.where(col("_split_bucket") != 0).select(*compact_columns)
    validation_frame = prepared.where(col("_split_bucket") == 0).select(
        *compact_columns
    )
    return train_frame, validation_frame


def run_cnn_training(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    minimum_accuracy: float,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    if not 0.0 <= minimum_accuracy <= 1.0:
        raise ValueError("minimum_accuracy must be between zero and one")

    spark = get_spark("trait-distrib-cnn-training")
    spark.sparkContext.setLogLevel("WARN")
    try:
        source = read_digit_csv(
            spark, str(Path(DATASET_DIR) / "train.csv"), has_label=True
        )
        train_frame, validation_frame = _prepare_split(source)
        with tempfile.TemporaryDirectory(prefix="trait-distrib-cnn-") as temp_dir:
            train_path = str(Path(temp_dir) / "train.parquet")
            validation_path = str(Path(temp_dir) / "validation.parquet")
            train_frame.write.mode("overwrite").parquet(train_path)
            validation_frame.write.mode("overwrite").parquet(validation_path)
            distributor = TorchDistributor(
                num_processes=1, local_mode=True, use_gpu=False
            )
            existing_python_path = os.environ.get("PYTHONPATH")
            os.environ["PYTHONPATH"] = os.pathsep.join(
                part for part in (str(PROJECT_ROOT), existing_python_path) if part
            )
            result = distributor.run(
                train_cnn_worker,
                train_path,
                validation_path,
                str(CNN_MODEL_PATH.resolve()),
                str(CNN_METRICS_PATH.resolve()),
                str(CNN_CONFUSION_MATRIX_PATH.resolve()),
                epochs,
                batch_size,
                learning_rate,
                minimum_accuracy,
            )
    finally:
        spark.stop()
    if not isinstance(result, dict):
        raise RuntimeError("TorchDistributor returned invalid CNN metrics")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the PyTorch CNN through PySpark TorchDistributor."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--minimum-accuracy", type=float, default=DEFAULT_MINIMUM_ACCURACY
    )
    args = parser.parse_args()
    metrics = run_cnn_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        minimum_accuracy=args.minimum_accuracy,
    )
    print(
        f"CNN sauvegardé dans {CNN_MODEL_PATH} avec une accuracy de validation "
        f"de {float(metrics['best_validation_accuracy']):.4f}."
    )


if __name__ == "__main__":
    main()
