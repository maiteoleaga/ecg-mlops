"""""
Funciones auxiliares: carga de configuración, datos y utilidades varias.
"""""

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Devuelve la ruta raíz del proyecto."""
    return Path(__file__).resolve().parents[1]


def load_config(nombre: str = "config.yaml") -> dict:
    """Carga un archivo YAML desde la carpeta config/."""
    raiz_proyecto = get_project_root()
    fichero_leer = raiz_proyecto / "config" / nombre
    logger.info(f"Cargando configuración desde {fichero_leer}")
    with open(fichero_leer) as file:
        output = yaml.safe_load(file)
    logger.info("Configuración cargada correctamente")
    return output


def set_seed(seed: int) -> None:
    """Fija las semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Devuelve cuda si está disponible o cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_data(relative_path: str) -> pd.DataFrame:
    """Carga un CSV desde una ruta relativa a la raíz del proyecto."""
    raiz_proyecto = get_project_root()
    fichero_datos = raiz_proyecto / relative_path
    logger.info(f"Cargando dataset desde {fichero_datos}")
    df = pd.read_csv(fichero_datos, header=None)
    logger.debug(f"Dataset cargado: shape={df.shape}")
    return df


def split_features_target(df: pd.DataFrame):
    """Separa features (columnas 0..N-1) y target (última columna).

    En el dataset MIT-BIH las columnas 0-186 son la señal ECG (187 puntos)
    y la columna 187 es la etiqueta de clase (0-4).
    """
    X = df.iloc[:, :-1].values             # shape (n, 187)
    y = df.iloc[:, -1].values.astype(int)  # shape (n,)
    return X, y


def to_tensors(X_train, X_val, y_train, y_val):
    """Convierte arrays NumPy a tensores PyTorch para clasificación multiclase.

    - X: float32, con un canal extra para Conv1D → shape (n, 1, 187)
    - y: long (entero) → CrossEntropyLoss espera índices de clase enteros
    """
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)

    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    return X_train_t, X_val_t, y_train_t, y_val_t