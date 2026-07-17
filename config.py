import os

DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "digit-recognizer"
)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
STATS_DIR = os.path.join(OUTPUT_DIR, "stats")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Si True, recalcule les statistiques même si elles existent déjà
FORCE_RECOMPUTE_STATS = True
