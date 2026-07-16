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
