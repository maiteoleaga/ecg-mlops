"""Tests del módulo del modelo (InceptionTime)."""
import torch

from src.model import InceptionTime, build_model_from_config
from src.utils import load_config


def test_inception_time_forward():
    """Forward pass del modelo con un batch dummy."""
    model = InceptionTime(n_classes=5)
    model.eval()

    batch = torch.randn(8, 1, 187) # generar un tensor aleatorio
    with torch.no_grad():
        out = model(batch)

    assert out.shape == (8, 5)
    assert not torch.isnan(out).any() # verificamos si en la salida hay algun NA que puede indicar un bug grave
    assert torch.isfinite(out).all() # verificamos que los valores son finitos


def test_build_model_from_config():
    """El modelo construido desde la config tiene la salida esperada."""
    cfg = load_config()
    model = build_model_from_config(cfg)

    assert isinstance(model, InceptionTime)

    dummy = torch.randn(4, 1, cfg["data"]["sequence_length"])
    model.eval()
    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (4, cfg["data"]["n_classes"])


def test_model_has_trainable_params():
    """El modelo tiene parámetros entrenables y un número razonable."""
    cfg = load_config()
    model = build_model_from_config(cfg)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params > 100_000, "Modelo demasiado pequeño"
    assert n_params < 5_000_000, "Modelo demasiado grande"


def test_output_can_become_probabilities():
    """Los logits pasan a probabilidades válidas con softmax."""
    cfg = load_config()
    model = build_model_from_config(cfg)
    model.eval()

    dummy = torch.randn(3, 1, cfg["data"]["sequence_length"])
    with torch.no_grad():
        logits = model(dummy)
        probs = torch.softmax(logits, dim=1)

    # Cada fila debe sumar 1
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5) # Verificamos que las probabilidades suman 1
    # Probabilidades en [0, 1]
    assert (probs >= 0).all() and (probs <= 1).all()