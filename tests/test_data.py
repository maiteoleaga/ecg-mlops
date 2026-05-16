"""Tests del módulo de datos."""
import numpy as np
import pytest

from src.data import (
    build_dataloaders,
    compute_class_weights,
    load_test,
    load_train,
    make_validation_split,
)
from src.utils import load_config


@pytest.fixture(scope="module") # carga la config una sola vez y la comparte entre todos los tests del módulo
def cfg():
    """Fixture que carga la configuración una sola vez por módulo."""
    return load_config()


def test_load_train(cfg):
    """El train CSV se carga y tiene la forma esperada."""
    X, y = load_train(cfg["data"]["train_csv"])
    assert X.ndim == 2
    assert X.shape[1] == cfg["data"]["sequence_length"]
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert set(np.unique(y)).issubset(set(range(cfg["data"]["n_classes"])))


def test_load_test(cfg):
    """El test CSV se carga correctamente."""
    X, y = load_test(cfg["data"]["test_csv"])
    assert X.shape[0] > 0
    assert X.shape[1] == cfg["data"]["sequence_length"]


def test_make_validation_split(cfg):
    """El split estratificado mantiene proporciones."""
    X, y = load_train(cfg["data"]["train_csv"])
    X_tr, X_val, y_tr, y_val = make_validation_split(
        X, y, val_size=cfg["data"]["val_size"], seed=cfg["seed"]
    )
    assert X_tr.shape[0] + X_val.shape[0] == X.shape[0]
    assert abs(X_val.shape[0] / X.shape[0] - cfg["data"]["val_size"]) < 0.01


def test_compute_class_weights(cfg):
    """Los pesos de clase son positivos y tienen tamaño n_classes."""
    X, y = load_train(cfg["data"]["train_csv"])
    weights = compute_class_weights(y, n_classes=cfg["data"]["n_classes"])
    assert weights.shape == (cfg["data"]["n_classes"],)
    assert (weights > 0).all()


def test_build_dataloaders(cfg):
    """Los DataLoaders devuelven batches con la forma correcta."""
    X, y = load_train(cfg["data"]["train_csv"])
    X_tr, X_val, y_tr, y_val = make_validation_split(
        X, y, val_size=cfg["data"]["val_size"], seed=cfg["seed"]
    )
    train_dl, val_dl = build_dataloaders(X_tr, y_tr, X_val, y_val, batch_size=32)

    xb, yb = next(iter(train_dl))
    assert xb.shape == (32, 1, cfg["data"]["sequence_length"])
    assert yb.shape == (32,)
    assert xb.dtype.is_floating_point