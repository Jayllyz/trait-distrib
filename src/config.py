import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, "digit-recognizer")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
STATS_DIR = os.path.join(OUTPUT_DIR, "stats")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Si True, recalcule les statistiques même si elles existent déjà
FORCE_RECOMPUTE_STATS = False
