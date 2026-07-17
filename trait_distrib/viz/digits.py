import os
import re

import matplotlib.pyplot as plt
import numpy as np

from trait_distrib.config import PLOTS_DIR


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_digit(pixels, title=None, ax=None):
    img = np.array(pixels).reshape(28, 28)
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


def save_sample_images(df, num_samples=10):
    """Prélève un échantillon du DataFrame Spark et sauvegarde les images."""
    ensure_dirs()
    rows = df.sample(fraction=0.001, seed=42).limit(num_samples).collect()
    if not rows:
        print("Avertissement : aucun échantillon prélevé.")
        return
    pixel_cols = [c for c in df.columns if c != "label"]
    pixel_cols.sort(
        key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0
    )
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for idx, row in enumerate(rows):
        if idx >= num_samples:
            break
        label = int(row["label"])
        pixels = [float(row[c]) for c in pixel_cols]
        ax = axes[idx // 5, idx % 5]
        plot_digit(pixels, title=f"Label: {label}", ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sample_images.png"), dpi=150)
    plt.close()
