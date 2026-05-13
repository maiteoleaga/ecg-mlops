"""
Preparación del dataset MIT-BIH para entrenamiento.

Funciones para:
- Cargar train y test desde CSV.
- Hacer split estratificado train/validation.
- Calcular pesos de clase para el desbalance.
- Construir TensorDatasets y DataLoaders.

"""
import logging
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from src.utils import load_data, split_features_target, to_tensors

logger = logging.getLogger(__name__)


def load_train(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Carga el CSV de train y devuelve (X, y)."""
    df = load_data(path)
    X, y = split_features_target(df)
    logger.info(f"Train: X={X.shape}, y={y.shape}")
    return X, y


def load_test(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Carga el CSV de test y devuelve (X, y)."""
    df = load_data(path)
    X, y = split_features_target(df)
    logger.info(f"Test: X={X.shape}, y={y.shape}")
    return X, y


def make_validation_split(X_train: np.ndarray, y_train: np.ndarray, val_size: float = 0.15, seed: int = 42,):
    """Hace un split estratificado del train original en train + validation."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=seed,
    )
    logger.info(f"Train tras split: {X_tr.shape}")
    logger.info(f"Validation:       {X_val.shape}")
    return X_tr, X_val, y_tr, y_val


def compute_class_weights(y: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """Calcula pesos por clase (inversamente proporcionales a la frecuencia)."""
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),
        y=y,
    )
    logger.info(f"Class weights: {weights}")
    return weights


def build_dataloaders(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader]:
    """Construye DataLoaders de train y validation a partir de los arrays."""
    X_tr_t, X_val_t, y_tr_t, y_val_t = to_tensors(X_tr, X_val, y_tr, y_val)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    logger.info(f"DataLoaders construidos con batch_size={batch_size}")
    return train_dl, val_dl