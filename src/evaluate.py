"""Funciones de evaluación: cálculo de métricas y matriz de confusión."""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def compute_metrics( y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    """Calcula accuracy, F1 (weighted, macro, per-class) y opcionalmente AUC."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_per_class": f1_score(y_true, y_pred, average=None).tolist(),
    }
    if y_proba is not None:
        metrics["auc_weighted"] = float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        )
    return metrics


def log_metrics(metrics: dict, prefix: str = "") -> None:
    """Imprime las métricas vía logger."""
    p = f"[{prefix}] " if prefix else ""
    logger.info(f"{p}Accuracy    : {metrics['accuracy']:.4f}")
    logger.info(f"{p}F1-weighted : {metrics['f1_weighted']:.4f}")
    logger.info(f"{p}F1-macro    : {metrics['f1_macro']:.4f}")
    if "auc_weighted" in metrics:
        logger.info(f"{p}AUC weighted: {metrics['auc_weighted']:.4f}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
    normalize: bool = True,
    title: str = "Matriz de confusión",
) -> None:
    """Dibuja y guarda la matriz de confusión en disco."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(title)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info(f"Matriz de confusión guardada en {output_path}")


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ejecuta el modelo sobre un DataLoader y devuelve (y_true, y_pred, y_proba)."""
    model.eval()
    all_logits, all_targets = [], []
    for xb, yb in dataloader:
        xb = xb.to(device)
        logits = model(xb)
        all_logits.append(logits.cpu())
        all_targets.append(yb)

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0).numpy()
    proba = torch.softmax(logits, dim=1).numpy()
    pred = proba.argmax(axis=1)
    return targets, pred, proba