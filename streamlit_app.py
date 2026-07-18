"""Mobile-first Streamlit page for testing handwritten postal codes."""

import hashlib
import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from postal_app.preprocessing import (
    ImageValidationError,
    SegmentationError,
    load_image,
    segment_postal_code,
    stack_segment_images,
    thicken_segments,
)
from postal_app.predictor import (
    PredictionError,
    PredictorConfigurationError,
    analyze_digits,
    get_predictor,
)

AUTOMATIC_SORT_THRESHOLD = 0.8
CNN_PARAMETER_COUNT = 421_642
CNN_METRICS_PATH = (
    Path(__file__).resolve().parent / "output" / "metrics" / "cnn_metrics.json"
)

st.set_page_config(
    page_title="Lecture de code postal",
    page_icon="✉️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container { max-width: 720px; padding-top: 1.25rem; padding-bottom: 3rem; }
      div[data-testid="stButton"] button,
      div[data-testid="stFileUploader"] button,
      div[data-testid="stCameraInput"] button { min-height: 3rem; }
      .postal-code {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: .75rem;
        color: #0f172a;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: clamp(2.25rem, 12vw, 4.5rem);
        font-weight: 800;
        letter-spacing: .16em;
        line-height: 1.2;
        padding: .65rem .35rem .65rem .55rem;
        text-align: center;
      }
      .demo-badge {
        background: #fff7ed;
        border: 1px solid #fdba74;
        border-radius: .5rem;
        color: #9a3412;
        margin-bottom: 1rem;
        padding: .7rem .85rem;
      }
      @media (max-width: 480px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        h1 { font-size: 1.75rem !important; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _reset_analysis() -> None:
    st.session_state["input_version"] = st.session_state.get("input_version", 0) + 1
    for key in ("analysis", "segments", "analysis_key", "analysis_error"):
        st.session_state.pop(key, None)


@st.cache_data
def _load_cnn_metrics() -> dict:
    try:
        metrics = json.loads(CNN_METRICS_PATH.read_text(encoding="utf-8"))
    except OSError, ValueError, TypeError:
        return {}
    return metrics if isinstance(metrics, dict) else {}


def _render_cnn_characteristics() -> None:
    metrics = _load_cnn_metrics()
    with st.expander("Caractéristiques du CNN", expanded=True):
        st.markdown(
            """
            **Architecture**

            `28×28×1 → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Dense(128) → 10 classes`

            - Entrée en niveaux de gris, normalisée entre 0 et 1.
            - Dropout de 25 % puis 50 % pour limiter le surapprentissage.
            - Entraînement PyTorch sur CPU orchestré par PySpark TorchDistributor.
            - Augmentations : rotations, translations, flou, bruit et épaisseur variable.
            """
        )
        accuracy = metrics.get("best_validation_accuracy")
        train_samples = metrics.get("train_samples")
        validation_samples = metrics.get("validation_samples")
        epochs = metrics.get("epochs_completed")
        first_row = st.columns(2)
        first_row[0].metric(
            "Accuracy de validation",
            f"{float(accuracy):.2%}" if isinstance(accuracy, (int, float)) else "—",
        )
        first_row[1].metric(
            "Paramètres entraînables", f"{CNN_PARAMETER_COUNT:,}".replace(",", " ")
        )
        second_row = st.columns(3)
        second_row[0].metric("Images d’entraînement", train_samples or "—")
        second_row[1].metric("Images de validation", validation_samples or "—")
        second_row[2].metric("Époques", epochs or "—")


def _render_segments(segments: tuple) -> None:
    st.subheader("Chiffres extraits")
    st.caption(
        "Ces cinq images sont les entrées 28 × 28 qui seront envoyées au modèle."
    )
    columns = st.columns(5)
    for index, (column, segment) in enumerate(
        zip(columns, segments, strict=True), start=1
    ):
        with column:
            st.image(segment.image, clamp=True, caption=str(index), width=88)


def _render_result(analysis, segments: tuple) -> None:
    _render_segments(segments)
    st.subheader("Résultat de l’analyse")
    st.markdown(
        f'<div class="postal-code" aria-label="Code postal détecté">'
        f"{analysis.postal_code}</div>",
        unsafe_allow_html=True,
    )

    st.metric("Confiance globale", f"{analysis.global_confidence:.0%}")
    st.caption("La confiance globale correspond au chiffre le moins sûr.")

    for position, prediction in enumerate(analysis.predictions, start=1):
        st.progress(
            prediction.confidence,
            text=(
                f"Position {position} · chiffre {prediction.digit} · "
                f"{prediction.confidence:.0%}"
            ),
        )

    if analysis.requires_review:
        st.warning(f"**Décision : {analysis.decision}**\n\n{analysis.review_reason}")
    else:
        st.success(f"**Décision : {analysis.decision}**")

    st.button(
        "Recommencer avec une nouvelle image",
        type="secondary",
        use_container_width=True,
        on_click=_reset_analysis,
    )


st.title("✉️ Lecture d’un code postal")
st.write(
    "Photographiez un code postal manuscrit de **cinq chiffres** pour tester "
    "la chaîne de traitement."
)

predictor_mode = os.getenv("PREDICTOR_MODE", "demo").strip().lower()
active_predictor_mode = predictor_mode
if predictor_mode == "demo":
    st.markdown(
        '<div class="demo-badge"><strong>Mode démo</strong> — les chiffres et les '
        "confiances affichés sont simulés. La segmentation de la photo est réelle.</div>",
        unsafe_allow_html=True,
    )
elif predictor_mode in {"spark", "real", "cnn"}:
    model_options = {
        "Random Forest — modèle Spark entraîné": "spark",
        "CNN — modèle PyTorch": "cnn",
    }
    model_labels = tuple(model_options)
    default_model_index = 1 if predictor_mode == "cnn" else 0
    selected_model_label = st.radio(
        "Modèle de reconnaissance",
        model_labels,
        index=default_model_index,
        horizontal=True,
    )
    active_predictor_mode = model_options[selected_model_label]
    st.success(f"Mode réel — {selected_model_label}.")
    if active_predictor_mode == "cnn":
        _render_cnn_characteristics()

with st.expander("Conseils pour une photo lisible", expanded=True):
    st.markdown(
        """
        - Écrivez les cinq chiffres horizontalement avec un petit espace entre eux.
        - Utilisez un stylo foncé sur une feuille claire et un éclairage uniforme.
        - Cadrez uniquement le code postal, sans autre texte ni bord de feuille.
        - Évitez le flou, les ombres fortes et les chiffres qui se touchent.
        """
    )

input_version = st.session_state.setdefault("input_version", 0)
input_method = st.radio(
    "Source de l’image",
    ("Prendre une photo", "Importer une image", "Dessiner les chiffres"),
    horizontal=True,
)

image_bytes = None
show_preview = True
if input_method == "Prendre une photo":
    uploaded = st.camera_input(
        "Photographier le code postal",
        key=f"camera_{input_version}",
    )
    if uploaded is not None:
        image_bytes = uploaded.getvalue()
elif input_method == "Importer une image":
    uploaded = st.file_uploader(
        "Choisir une image JPG ou PNG",
        type=("jpg", "jpeg", "png"),
        key=f"upload_{input_version}",
    )
    if uploaded is not None:
        image_bytes = uploaded.getvalue()
else:
    st.caption("Écrivez les cinq chiffres côte à côte, avec un petit espace entre eux.")
    canvas = st_canvas(
        stroke_width=16,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=220,
        width=680,
        drawing_mode="freedraw",
        key=f"canvas_{input_version}",
    )
    show_preview = False
    if canvas.image_data is not None:
        rgba = Image.fromarray(canvas.image_data.astype("uint8"), "RGBA")
        white = Image.new("RGBA", rgba.size, "#FFFFFF")
        drawing = Image.alpha_composite(white, rgba).convert("RGB")
        if np.asarray(drawing.convert("L")).min() < 250:
            buffer = BytesIO()
            drawing.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

if image_bytes is not None:
    current_hash = hashlib.sha256(image_bytes).hexdigest()
    if show_preview:
        st.image(image_bytes, caption="Image à analyser", use_container_width=True)

    thicken_strokes = st.toggle(
        "Épaissir légèrement les traits (expérimental)",
        value=False,
        help=(
            "Applique une dilatation légère aux images 28 × 28 avant de les "
            "envoyer au modèle. Comparez le résultat avec et sans cette option."
        ),
    )
    current_analysis_key = (
        f"{current_hash}:model={active_predictor_mode}:thicken={int(thicken_strokes)}"
    )

    if st.session_state.get("analysis_key") != current_analysis_key:
        for stale_key in ("analysis", "segments", "analysis_error"):
            st.session_state.pop(stale_key, None)

    if st.button(
        "Analyser le code postal",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Détection et séparation des cinq chiffres…"):
                image_rgb = load_image(image_bytes)
                segments = segment_postal_code(image_rgb)
                if thicken_strokes:
                    segments = thicken_segments(segments)
                batch = stack_segment_images(segments)
                predictor = get_predictor(active_predictor_mode)
                analysis = analyze_digits(
                    batch,
                    predictor,
                    threshold=AUTOMATIC_SORT_THRESHOLD,
                )
            st.session_state["analysis"] = analysis
            st.session_state["segments"] = segments
            st.session_state["analysis_key"] = current_analysis_key
            st.session_state.pop("analysis_error", None)
        except (
            ImageValidationError,
            SegmentationError,
            PredictionError,
            PredictorConfigurationError,
        ) as error:
            st.session_state["analysis_error"] = str(error)
            st.session_state["analysis_key"] = current_analysis_key
            st.session_state.pop("analysis", None)
            st.session_state.pop("segments", None)
        except Exception:
            st.session_state["analysis_error"] = (
                "Une erreur inattendue a interrompu l’analyse. Réessayez avec une autre photo."
            )
            st.session_state["analysis_key"] = current_analysis_key
            st.session_state.pop("analysis", None)
            st.session_state.pop("segments", None)

    if (
        st.session_state.get("analysis_error")
        and st.session_state.get("analysis_key") == current_analysis_key
    ):
        st.error(st.session_state["analysis_error"])

    if (
        st.session_state.get("analysis") is not None
        and st.session_state.get("analysis_key") == current_analysis_key
    ):
        _render_result(
            st.session_state["analysis"],
            st.session_state["segments"],
        )
else:
    st.info("Aucune image n’est conservée sur le serveur après la session.")
