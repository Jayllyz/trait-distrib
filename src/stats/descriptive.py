import csv
import os
import re

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, mean, stddev, when
from pyspark.sql.functions import max as spark_max
from pyspark.sql.functions import min as spark_min
from pyspark.sql.functions import sum as spark_sum

from src.config import STATS_DIR
from src.spark.transformations import create_temp_view, filter_rows, run_sql
from src.stats.storage import ensure_dirs

CENTER_PIXEL = "pixel406"  # ligne 14, colonne 14 de l'image 28x28


def get_sorted_pixel_cols(df: DataFrame):
    """Retourne les colonnes pixel triées par numéro (pixel0, pixel1, ...)."""
    cols = [c for c in df.columns if c != "label"]
    cols.sort(
        key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0
    )
    return cols


def run_sql_analysis(spark: SparkSession, df: DataFrame) -> list[dict]:
    """Analyse via Spark SQL (vue temporaire + requêtes) et équivalent DataFrame."""
    create_temp_view(df, "digits")
    label_stats = run_sql(
        spark,
        f"SELECT label, COUNT(*) AS n, ROUND(AVG({CENTER_PIXEL}), 2) AS avg_center_ink "
        "FROM digits GROUP BY label ORDER BY label",
    )
    label_stats.show()

    sql_inked = run_sql(
        spark, f"SELECT COUNT(*) AS n FROM digits WHERE {CENTER_PIXEL} > 0"
    ).collect()[0]["n"]
    df_inked = filter_rows(df, col(CENTER_PIXEL) > 0).count()
    print(f"Images avec le pixel central encré : SQL={sql_inked}, DataFrame={df_inked}")

    data = [
        {
            "label": int(row["label"]),
            "n": int(row["n"]),
            "avg_center_ink": float(row["avg_center_ink"]),
        }
        for row in label_stats.collect()
    ]
    ensure_dirs()
    with open(os.path.join(STATS_DIR, "sql_label_stats.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "n", "avg_center_ink"])
        writer.writeheader()
        writer.writerows(data)
    return data


def compute_and_save_frequencies(df: DataFrame):
    """Calcule la fréquence des labels et sauvegarde en CSV."""
    freq_rows = df.groupBy("label").count().orderBy("label").collect()
    data = [{"label": row["label"], "count": row["count"]} for row in freq_rows]
    ensure_dirs()
    with open(os.path.join(STATS_DIR, "frequencies.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "count"])
        writer.writeheader()
        writer.writerows(data)


def compute_and_save_pixel_stats(df: DataFrame):
    """Calcule moyenne, écart-type, min, max, nombre de zéros par pixel."""
    pixel_cols = get_sorted_pixel_cols(df)
    agg_exprs = []
    for c in pixel_cols:
        agg_exprs.append(mean(col(c)).alias(f"mean_{c}"))
        agg_exprs.append(stddev(col(c)).alias(f"std_{c}"))
        agg_exprs.append(spark_min(col(c)).alias(f"min_{c}"))
        agg_exprs.append(spark_max(col(c)).alias(f"max_{c}"))
        agg_exprs.append(
            spark_sum(when(col(c) == 0, 1).otherwise(0)).alias(f"zero_{c}")
        )
    stats_row = df.select(agg_exprs).collect()[0]
    data = []
    for c in pixel_cols:
        data.append(
            {
                "pixel": c,
                "mean": float(stats_row[f"mean_{c}"]),
                "std": float(stats_row[f"std_{c}"]),
                "min": float(stats_row[f"min_{c}"]),
                "max": float(stats_row[f"max_{c}"]),
                "zeros": int(stats_row[f"zero_{c}"]),
            }
        )
    ensure_dirs()
    with open(os.path.join(STATS_DIR, "pixel_stats.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pixel", "mean", "std", "min", "max", "zeros"]
        )
        writer.writeheader()
        writer.writerows(data)


def compute_and_save_mean_by_label(df: DataFrame):
    """Calcule l'image moyenne pour chaque label."""
    pixel_cols = get_sorted_pixel_cols(df)
    agg_exprs = [mean(col(c)).alias(c) for c in pixel_cols]
    mean_rows = df.groupBy("label").agg(*agg_exprs).orderBy("label").collect()
    data = []
    for row in mean_rows:
        d = {"label": int(row["label"])}
        for c in pixel_cols:
            d[c] = float(row[c])
        data.append(d)
    ensure_dirs()
    with open(os.path.join(STATS_DIR, "mean_by_label.csv"), "w", newline="") as f:
        fieldnames = ["label"] + pixel_cols
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def compute_and_save_variance_map(df: DataFrame):
    """Calcule l'écart-type de chaque pixel (sur tout le dataset)."""
    pixel_cols = get_sorted_pixel_cols(df)
    var_exprs = [stddev(col(c)).alias(c) for c in pixel_cols]
    var_row = df.select(var_exprs).collect()[0]
    data = [{"pixel": c, "std": float(var_row[c])} for c in pixel_cols]
    ensure_dirs()
    with open(os.path.join(STATS_DIR, "variance_map.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pixel", "std"])
        writer.writeheader()
        writer.writerows(data)
