import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(PROJECT_ROOT, "digit-recognizer")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
CLASSIFIER_MODELS_DIR = os.path.join(MODELS_DIR, "classifiers")
PREPROCESSING_DIR = os.path.join(OUTPUT_DIR, "preprocessing")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
STATS_DIR = os.path.join(OUTPUT_DIR, "stats")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Si True, recalcule les statistiques même si elles existent déjà
FORCE_RECOMPUTE_STATS = False

PRODUCTION_PREPROCESSING_MODEL_NAME = "normalized_compact_pca"
BEST_CLASSIFIER_PREFIX = "best_"
BEST_CLASSIFIER_MANIFEST = os.path.join(CLASSIFIER_MODELS_DIR, "best_model_name.txt")
