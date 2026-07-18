from streamlit.testing.v1 import AppTest


def test_streamlit_page_starts_in_demo_mode(monkeypatch) -> None:
    monkeypatch.setenv("PREDICTOR_MODE", "demo")

    app = AppTest.from_file("streamlit_app.py").run(timeout=15)

    assert not app.exception
    assert app.title[0].value == "✉️ Lecture d’un code postal"
    assert app.radio[0].value == "Prendre une photo"
    assert "Aucune image n’est conservée" in app.info[0].value


def test_streamlit_page_offers_drawing_canvas(monkeypatch) -> None:
    monkeypatch.setenv("PREDICTOR_MODE", "demo")

    app = AppTest.from_file("streamlit_app.py").run(timeout=15)
    assert "Dessiner les chiffres" in app.radio[0].options

    app.radio[0].set_value("Dessiner les chiffres").run(timeout=15)

    assert not app.exception
    assert any("cinq chiffres côte à côte" in caption.value for caption in app.caption)


def test_streamlit_page_announces_real_mode(monkeypatch) -> None:
    monkeypatch.setenv("PREDICTOR_MODE", "spark")

    app = AppTest.from_file("streamlit_app.py").run(timeout=15)

    assert not app.exception
    assert "modèle Spark entraîné" in app.success[0].value
