"""API FastAPI para servir predicciones del modelo InceptionTime."""
import logging
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from api.schemas import PredictRequest, PredictResponse, HealthResponse
from src.model import build_model_from_config
from src.utils import get_project_root, load_config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ──────────────────────────────────────────────────────────────────────
# Carga de configuración y modelo (una sola vez al arrancar)
# ──────────────────────────────────────────────────────────────────────
CFG = load_config()
MODEL_PATH = get_project_root() / "models" / "inception_time.pt"
CLASS_NAMES = {int(k): v for k, v in CFG["data"]["class_names"].items()}

# Variables globales que rellenamos al cargar el modelo
_model: torch.nn.Module | None = None
_device: torch.device = torch.device("cpu")  # la API siempre corre en CPU


def load_model() -> torch.nn.Module:
    """Carga el modelo desde el .pt. Llamado al arrancar la API."""
    global _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            f"Descárgalo desde W&B antes de arrancar la API."
        )

    model = build_model_from_config(CFG) # 1. Creo el modelo con pesos aleatorios 
    state_dict = torch.load(MODEL_PATH, map_location=_device) # 2. Cargo los pesos en memoria
    model.load_state_dict(state_dict) # 3. PyTorch coge los tensores del diccionario y los mete en la arquitectura, sustituyendo los pesos aleatorios por los entrenados.
    model.to(_device)
    model.eval()

    logger.info(f"Modelo cargado desde {MODEL_PATH}")
    _model = model
    return model


# ──────────────────────────────────────────────────────────────────────
# App FastAPI
# ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ECG Heartbeat Classifier",
    description="API para clasificar latidos ECG en 5 clases (MIT-BIH).",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event() -> None:
    """Carga el modelo al arrancar la API."""
    load_model()


# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root() -> dict:
    return {
        "name": "ECG Heartbeat Classifier",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "Información de la API",
            "GET /health": "Health-check",
            "POST /predict": "Predicción a partir de una señal ECG de 187 puntos",
            "GET /docs": "Documentación interactiva (Swagger UI)",
        },
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado todavía")

    # 1. Convertir la señal a tensor (1, 1, 187)
    signal = np.array(request.signal, dtype=np.float32)
    tensor = torch.tensor(signal).view(1, 1, -1).to(_device)

    # 2. Inferencia
    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # 3. Construir la respuesta
    predicted = int(probs.argmax())
    return PredictResponse(
        predicted_class=predicted,
        class_name=CLASS_NAMES[predicted],
        probabilities={CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))},
    )