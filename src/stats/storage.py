import csv
import os

from src.config import STATS_DIR


def ensure_dirs():
    os.makedirs(STATS_DIR, exist_ok=True)


def load_frequencies():
    data = []
    with open(os.path.join(STATS_DIR, "frequencies.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({"label": int(row["label"]), "count": int(row["count"])})
    return data


def load_pixel_stats():
    data = []
    with open(os.path.join(STATS_DIR, "pixel_stats.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(
                {
                    "pixel": row["pixel"],
                    "mean": float(row["mean"]),
                    "std": float(row["std"]),
                    "min": float(row["min"]),
                    "max": float(row["max"]),
                    "zeros": int(row["zeros"]),
                }
            )
    return data


def load_mean_by_label():
    data = []
    with open(os.path.join(STATS_DIR, "mean_by_label.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = {"label": int(row["label"])}
            for k, v in row.items():
                if k != "label":
                    d[k] = float(v)
            data.append(d)
    return data


def load_variance_map():
    data = []
    with open(os.path.join(STATS_DIR, "variance_map.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({"pixel": row["pixel"], "std": float(row["std"])})
    return data
