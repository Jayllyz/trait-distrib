# trait-distrib

PySpark project managed with uv.

## Setup

```bash
just install
```

The app downloads the [digit-recognizer](https://www.kaggle.com/competitions/digit-recognizer/data)
dataset via `kagglehub`, which requires Kaggle API credentials. Either place a
`kaggle.json` token in `~/.kaggle/kaggle.json` or set the `KAGGLE_USERNAME` and
`KAGGLE_KEY` environment variables (see
[Kaggle API docs](https://www.kaggle.com/docs/api)).

## Run

```bash
just run
```

## Preprocess

Run the Spark preprocessing workflow for step 3 with feature assembly,
normalization, pixel filtering, and PCA experiments:

```bash
just preprocess
```

The generated artifacts are written under `output/`:

* `output/preprocessing/` for transformed datasets
* `output/models/` for fitted preprocessing pipelines
* `output/metrics/preprocessing_summary.csv` for the configuration comparison

## Train

Run the step 4 machine-learning workflow and display the comparison table in the
terminal:

```bash
just train
```

The workflow evaluates several Spark ML models, saves the comparison metrics,
exports the best model, and writes the confusion matrix plus class-level metrics
under `output/metrics/`.

## Lint, format & typecheck

```bash
just lint       # ruff check --fix
just format     # ruff format
just typecheck  # ty check
just check      # lint + format + typecheck
```

## Test

```bash
just test
```

## Clean

```bash
just clean
```

See `justfile` for all recipes (`just --list`).
