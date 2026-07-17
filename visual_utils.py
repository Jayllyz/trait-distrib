import os
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import PLOTS_DIR


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def get_sorted_pixel_names_from_dicts(data, key_prefix="pixel"):
    """Retourne la liste des clés correspondant aux pixels (hors label) triées par numéro."""
    # On prend la première ligne pour connaître les colonnes
    if not data:
        return []
    keys = [k for k in data[0].keys() if k.startswith(key_prefix) and k != "label"]
    keys.sort(
        key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0
    )
    return keys


def plot_digit(pixels, title=None, ax=None):
    img = np.array(pixels).reshape(28, 28)
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


def save_frequency_plot(freq_data):
    ensure_dirs()
    if not freq_data:
        print("Avertissement : données de fréquence vides.")
        return
    labels = [d["label"] for d in freq_data]
    counts = [d["count"] for d in freq_data]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=counts, palette="viridis")
    plt.title("Distribution des chiffres dans l'ensemble d'entraînement")
    plt.xlabel("Chiffre")
    plt.ylabel("Nombre d'exemples")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "frequencies.png"), dpi=150)
    plt.close()


def save_global_mean_plot(pixel_stats_data):
    ensure_dirs()
    if not pixel_stats_data:
        print("Avertissement : données pixel_stats vides.")
        return
    # Trier par pixel
    pixel_stats_data.sort(
        key=lambda x: int(re.search(r"\d+", x["pixel"]).group())
        if re.search(r"\d+", x["pixel"])
        else 0
    )
    means = [d["mean"] for d in pixel_stats_data]
    if len(means) != 784:
        print(f"Erreur : nombre de pixels {len(means)} != 784")
        return
    mean_img = np.array(means).reshape(28, 28)
    plt.figure()
    plt.imshow(mean_img, cmap="gray")
    plt.title("Image moyenne globale (tous chiffres confondus)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "global_mean.png"), dpi=150)
    plt.close()


def save_mean_by_label_plot(mean_by_label_data):
    ensure_dirs()
    if not mean_by_label_data:
        print("Avertissement : données mean_by_label vides.")
        return
    pixel_cols = get_sorted_pixel_names_from_dicts(
        mean_by_label_data, key_prefix="pixel"
    )
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        # Trouver la ligne correspondant au label i
        row = next((d for d in mean_by_label_data if d["label"] == i), None)
        if row:
            pixels = [row[c] for c in pixel_cols]
            ax = axes[i // 5, i % 5]
            plot_digit(pixels, title=f"Chiffre {i}", ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "mean_by_label.png"), dpi=150)
    plt.close()


def save_variance_map(variance_data):
    ensure_dirs()
    if not variance_data:
        print("Avertissement : données variance vides.")
        return
    variance_data.sort(
        key=lambda x: int(re.search(r"\d+", x["pixel"]).group())
        if re.search(r"\d+", x["pixel"])
        else 0
    )
    stds = [d["std"] for d in variance_data]
    if len(stds) != 784:
        print(f"Erreur : nombre de pixels {len(stds)} != 784")
        return
    var_img = np.array(stds).reshape(28, 28)
    plt.figure()
    plt.imshow(var_img, cmap="hot", interpolation="nearest")
    plt.colorbar(label="Écart-type des pixels")
    plt.title("Zones à forte variabilité (zones importantes)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "variance_map.png"), dpi=150)
    plt.close()


def save_sample_images(df, num_samples=10):
    """Prend un DataFrame Spark, prélève un échantillon et affiche les images."""
    ensure_dirs()
    sample_df = df.sample(fraction=0.001, seed=42).limit(num_samples)
    if sample_df.count() == 0:
        print("Avertissement : aucun échantillon prélevé.")
        return
    sample_pd = (
        sample_df.toPandas()
    )  # on utilise pandas juste pour l'échantillon (petit)
    pixel_cols = [c for c in sample_pd.columns if c != "label"]
    pixel_cols.sort(
        key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0
    )
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for idx, (_, row) in enumerate(sample_pd.iterrows()):
        if idx >= num_samples:
            break
        label = int(row["label"])
        pixels = row[pixel_cols].values.astype(float)
        ax = axes[idx // 5, idx % 5]
        plot_digit(pixels, title=f"Label: {label}", ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sample_images.png"), dpi=150)
    plt.close()


def save_correlation_matrix(mean_by_label_data):
    ensure_dirs()
    if not mean_by_label_data:
        print("Avertissement : données mean_by_label vides pour corrélation.")
        return
    pixel_cols = get_sorted_pixel_names_from_dicts(
        mean_by_label_data, key_prefix="pixel"
    )
    # Construire un dictionnaire label -> vecteur de pixels
    mean_images = {}
    for d in mean_by_label_data:
        mean_images[d["label"]] = np.array([d[c] for c in pixel_cols])
    if len(mean_images) < 10:
        print(f"Attention : seules {len(mean_images)} classes disponibles.")
    corr_matrix = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if i in mean_images and j in mean_images:
                corr_matrix[i, j] = np.corrcoef(mean_images[i], mean_images[j])[0, 1]
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=range(10),
        yticklabels=range(10),
    )
    plt.title("Matrice de corrélation entre les images moyennes des classes")
    plt.xlabel("Chiffre")
    plt.ylabel("Chiffre")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_matrix.png"), dpi=150)
    plt.close()


def save_class_comparison(mean_by_label_data, pairs=[(1, 7), (3, 8), (4, 9)]):
    ensure_dirs()
    if not mean_by_label_data:
        print("Avertissement : données mean_by_label vides pour comparaison.")
        return
    pixel_cols = get_sorted_pixel_names_from_dicts(
        mean_by_label_data, key_prefix="pixel"
    )
    mean_images = {}
    for d in mean_by_label_data:
        mean_images[d["label"]] = np.array([d[c] for c in pixel_cols])
    fig, axes = plt.subplots(len(pairs), 3, figsize=(9, 3 * len(pairs)))
    if len(pairs) == 1:
        axes = axes.reshape(1, -1)
    for row_idx, (a, b) in enumerate(pairs):
        if a in mean_images and b in mean_images:
            img_a = mean_images[a].reshape(28, 28)
            img_b = mean_images[b].reshape(28, 28)
            diff = np.abs(img_a - img_b)
            axes[row_idx, 0].imshow(img_a, cmap="gray")
            axes[row_idx, 0].set_title(f"Chiffre {a}")
            axes[row_idx, 0].axis("off")
            axes[row_idx, 1].imshow(img_b, cmap="gray")
            axes[row_idx, 1].set_title(f"Chiffre {b}")
            axes[row_idx, 1].axis("off")
            axes[row_idx, 2].imshow(diff, cmap="hot")
            axes[row_idx, 2].set_title(f"Différence |{a}-{b}|")
            axes[row_idx, 2].axis("off")
        else:
            axes[row_idx, 0].text(
                0.5, 0.5, f"Classe {a} manquante", ha="center", va="center"
            )
            axes[row_idx, 1].text(
                0.5, 0.5, f"Classe {b} manquante", ha="center", va="center"
            )
            axes[row_idx, 2].text(0.5, 0.5, "N/A", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_comparisons.png"), dpi=150)
    plt.close()
