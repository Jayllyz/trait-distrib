import os
import sys

import stats_utils
import visual_utils
from src.config import FORCE_RECOMPUTE_STATS, OUTPUT_DIR, PLOTS_DIR, STATS_DIR
from src.data_loader import fetch_dataset, get_spark, load_train


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
        freq_data = stats_utils.load_frequencies()
        pixel_stats_data = stats_utils.load_pixel_stats()
        mean_by_label_data = stats_utils.load_mean_by_label()
        variance_data = stats_utils.load_variance_map()
    else:
        print("Calcul des statistiques avec Spark...")
        stats_utils.compute_and_save_frequencies(train_df)
        stats_utils.compute_and_save_pixel_stats(train_df)
        stats_utils.compute_and_save_mean_by_label(train_df)
        stats_utils.compute_and_save_variance_map(train_df)
        # Recharger les données depuis les fichiers
        freq_data = stats_utils.load_frequencies()
        pixel_stats_data = stats_utils.load_pixel_stats()
        mean_by_label_data = stats_utils.load_mean_by_label()
        variance_data = stats_utils.load_variance_map()
        print("Statistiques sauvegardées.")

    # Générer les graphiques
    print("Génération des graphiques...")
    visual_utils.save_frequency_plot(freq_data)
    visual_utils.save_global_mean_plot(pixel_stats_data)
    visual_utils.save_mean_by_label_plot(mean_by_label_data)
    visual_utils.save_variance_map(variance_data)
    visual_utils.save_sample_images(train_df)
    visual_utils.save_correlation_matrix(mean_by_label_data)
    visual_utils.save_class_comparison(mean_by_label_data)

    print(f"Tous les résultats ont été sauvegardés dans {OUTPUT_DIR}")
    spark.stop()


if __name__ == "__main__":
    main()
