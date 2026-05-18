"""Tests de los endpoints de la API."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.utils import get_project_root


@pytest.fixture(scope="module")
def client():
    """TestClient que dispara el evento startup (carga el modelo)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_signal():
    """Un latido real del test set para los tests de predicción."""
    csv_path = get_project_root() / "data" / "raw" / "mitbih_test.csv"
    df = pd.read_csv(csv_path, header=None)
    return df.iloc[0, :-1].tolist()

@pytest.mark.requires_model
def test_root(client):
    """GET / responde 200 con la info de la API."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "endpoints" in data

@pytest.mark.requires_model
def test_health(client):
    """GET /health devuelve status ok y modelo cargado."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True

@pytest.mark.requires_model
def test_predict_ok(client, sample_signal):
    """POST /predict con una señal válida devuelve una predicción coherente."""
    response = client.post("/predict", json={"signal": sample_signal})
    assert response.status_code == 200

    data = response.json()
    assert 0 <= data["predicted_class"] <= 4
    assert isinstance(data["class_name"], str)
    assert len(data["probabilities"]) == 5

    # Las probabilidades suman ~1
    total = sum(data["probabilities"].values())
    assert abs(total - 1.0) < 1e-4

@pytest.mark.requires_model
def test_predict_invalid_length(client):
    """POST /predict con señal de longitud incorrecta devuelve 422."""
    bad_signal = [0.0] * 100  # debe ser 187
    response = client.post("/predict", json={"signal": bad_signal})
    assert response.status_code == 422

@pytest.mark.requires_model
def test_predict_missing_field(client):
    """POST /predict sin el campo 'signal' devuelve 422."""
    response = client.post("/predict", json={})
    assert response.status_code == 422