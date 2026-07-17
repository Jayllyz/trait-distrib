import os
import sys

from trait_distrib.config import FORCE_RECOMPUTE_STATS, OUTPUT_DIR, PLOTS_DIR, STATS_DIR
from trait_distrib.spark.io import fetch_dataset, load_train
from trait_distrib.spark.session import get_spark
from trait_distrib.stats import descriptive, storage
from trait_distrib.viz import digits, plots


def main() -> None:
    try:
        fetch_dataset()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    spark = get_spark()
    train_df = load_train(spark).cache()

    stats_files_exist = all(
        os.path.exists(os.path.join(STATS_DIR, f))
        for f in [
            "frequencies.csv",
            "pixel_stats.csv",
            "mean_by_label.csv",
            "variance_map.csv",
        ]
    )

    if not FORCE_RECOMPUTE_STATS and stats_files_exist:
        print("Chargement des statistiques depuis les fichiers sauvegardés...")
        freq_data = storage.load_frequencies()
        pixel_stats_data = storage.load_pixel_stats()
        mean_by_label_data = storage.load_mean_by_label()
        variance_data = storage.load_variance_map()
    else:
        print("Calcul des statistiques avec Spark...")
        descriptive.compute_and_save_frequencies(train_df)
        descriptive.compute_and_save_pixel_stats(train_df)
        descriptive.compute_and_save_mean_by_label(train_df)
        descriptive.compute_and_save_variance_map(train_df)
        # Recharger les données depuis les fichiers
        freq_data = storage.load_frequencies()
        pixel_stats_data = storage.load_pixel_stats()
        mean_by_label_data = storage.load_mean_by_label()
        variance_data = storage.load_variance_map()
        print("Statistiques sauvegardées.")

    # Générer les graphiques
    print("Génération des graphiques...")
    plots.save_frequency_plot(freq_data)
    plots.save_global_mean_plot(pixel_stats_data)
    plots.save_mean_by_label_plot(mean_by_label_data)
    plots.save_variance_map(variance_data)
    digits.save_sample_images(train_df)
    plots.save_correlation_matrix(mean_by_label_data)
    plots.save_class_comparison(mean_by_label_data)

    print(f"Tous les résultats ont été sauvegardés dans {OUTPUT_DIR}")
    spark.stop()


if __name__ == "__main__":
    main()
