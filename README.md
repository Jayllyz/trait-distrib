# trait-distrib

Projet PySpark de reconnaissance de chiffres manuscrits, avec une page Streamlit
mobile pour tester la capture d'un code postal.

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

## Page Streamlit

```bash
just streamlit
```

Ouvrir ensuite <http://localhost:8501>. Sur un téléphone, utiliser l'URL HTTPS
déployée afin que le navigateur autorise l'accès à la caméra.

La page accepte une photo prise directement ou un fichier JPG/PNG de 10 Mo au
maximum. Elle recadre l'écriture, sépare les cinq chiffres et produit cinq images
MNIST de 28 × 28 pixels.

Par défaut, `PREDICTOR_MODE=demo` : la segmentation est réelle, mais le code et
les confiances sont simulés et signalés comme tels dans l'interface. Le futur
adaptateur du modèle devra implémenter le contrat `Predictor` de
`postal_app/predictor.py`, puis être sélectionné avec `PREDICTOR_MODE=spark`.

## Déploiement Coolify

Le `Dockerfile` expose le port `8501` et inclut un contrôle de santé sur
`/_stcore/health`.

1. Créer une ressource Coolify depuis ce dépôt avec le build Dockerfile.
2. Définir le port du conteneur sur `8501`.
3. Associer un domaine HTTPS à la ressource.
4. Conserver `PREDICTOR_MODE=demo` tant que l'adaptateur Spark n'est pas ajouté.

Les images sont traitées en mémoire par Streamlit et ne sont pas enregistrées
par l'application.

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
