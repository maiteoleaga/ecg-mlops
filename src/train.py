"""Script de entrenamiento de InceptionTime sobre el dataset MIT-BIH."""
import logging
from pathlib import Path

import argparse
import wandb

import torch
from torch.utils.data import DataLoader

from src.data import build_dataloaders, load_test, load_train, make_validation_split
from src.evaluate import (
    compute_metrics,
    evaluate_model,
    log_metrics,
    plot_confusion_matrix,
)
from src.model import build_model_from_config
from src.utils import get_device, get_project_root, load_config, set_seed

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configura el formato de logs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def init_wandb(cfg: dict, run_name: str | None = None):
    """Inicializa un run de W&B con los hiperparámetros del cfg."""
    wandb_cfg = cfg["wandb"]
    return wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        name=run_name,
        config={
            "seed": cfg["seed"],
            "model": cfg["model"],
            "training": cfg["training"],
            "data": {
                "n_classes": cfg["data"]["n_classes"],
                "val_size": cfg["data"]["val_size"],
            },
        },
    )

def train_one_epoch(model: torch.nn.Module, train_dl: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, device: torch.device) -> float:
    """Entrena el modelo una época y devuelve la pérdida media."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def validate_one_epoch(model: torch.nn.Module, val_dl: DataLoader, loss_fn: torch.nn.Module, device: torch.device) -> float:
    """Calcula la pérdida media en validación."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for xb, yb in val_dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches

def train_one_run(cfg: dict, use_wandb: bool = False, run_name: str | None = None) -> dict:
    """Ejecuta un entrenamiento completo y devuelve métricas finales."""
    set_seed(cfg["seed"])
    device = get_device()
    logger.info(f"Device: {device}")

    # === W&B ===
    run = init_wandb(cfg, run_name=run_name) if use_wandb else None

    # === Datos ===
    X_train, y_train = load_train(cfg["data"]["train_csv"])
    X_test, y_test = load_test(cfg["data"]["test_csv"])

    X_tr, X_val, y_tr, y_val = make_validation_split(
        X_train, y_train, val_size=cfg["data"]["val_size"], seed=cfg["seed"]
    )
    train_dl, val_dl = build_dataloaders(
        X_tr, y_tr, X_val, y_val, batch_size=cfg["training"]["batch_size"]
    )
    _, test_dl = build_dataloaders(
        X_test, y_test, X_test, y_test, batch_size=cfg["training"]["batch_size"]
    )

    # === Modelo ===
    model = build_model_from_config(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Modelo: {type(model).__name__} con {n_params:,} parámetros")
    if run is not None:
        wandb.log({"n_params": n_params})

    # === Optimizador, pérdida y scheduler ===
    lr = cfg["training"]["learning_rate"]
    n_epochs = cfg["training"]["n_epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=n_epochs * len(train_dl)
    )

    # === Bucle de entrenamiento ===
    logger.info(f"Entrenando durante {n_epochs} epochs...")
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_dl, optimizer, loss_fn, device)
        val_loss = validate_one_epoch(model, val_dl, loss_fn, device)
        scheduler.step()
        logger.info(
            f"Epoch {epoch}/{n_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )
        if run is not None:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    # === Evaluación en validación ===
    logger.info("Evaluando en validación...")
    y_true_v, y_pred_v, y_proba_v = evaluate_model(model, val_dl, device)
    val_metrics = compute_metrics(y_true_v, y_pred_v, y_proba_v)
    log_metrics(val_metrics, prefix="val")

    # === Evaluación en test ===
    logger.info("Evaluando en test...")
    y_true_t, y_pred_t, y_proba_t = evaluate_model(model, test_dl, device)
    test_metrics = compute_metrics(y_true_t, y_pred_t, y_proba_t)
    log_metrics(test_metrics, prefix="test")

    if run is not None:
        wandb.log({
            **{f"val/{k}": v for k, v in val_metrics.items() if k != "f1_per_class"},
            **{f"test/{k}": v for k, v in test_metrics.items() if k != "f1_per_class"},
        })

    # === Guardar modelo ===
    root = get_project_root()
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "inception_time.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Modelo guardado en {model_path}")

    # === Matriz de confusión ===
    class_names = [cfg["data"]["class_names"][i] for i in range(cfg["data"]["n_classes"])]
    cm_path = model_dir / "confusion_matrix_test.png"
    plot_confusion_matrix(
        y_true_t, y_pred_t, class_names=class_names,
        output_path=cm_path, normalize=True, title="Confusion matrix (test)",
    )

    # === Subir modelo y matriz como artifact a W&B ===
    if run is not None:
        artifact = wandb.Artifact(
            name="inception_time",
            type="model",
            description=f"InceptionTime trained on MIT-BIH (F1-macro test={test_metrics['f1_macro']:.4f})",
        )
        artifact.add_file(str(model_path))
        artifact.add_file(str(cm_path))
        run.log_artifact(artifact)
        wandb.finish()

    return {"val": val_metrics, "test": test_metrics}


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Entrena InceptionTime sobre MIT-BIH")
    parser.add_argument("--wandb", action="store_true", help="Activar tracking en W&B")
    parser.add_argument("--name", type=str, default=None, help="Nombre del run en W&B")
    args = parser.parse_args()

    cfg = load_config()
    logger.info("=== Empieza el entrenamiento ===")
    results = train_one_run(cfg, use_wandb=args.wandb, run_name=args.name)
    logger.info("=== Entrenamiento finalizado ===")
    logger.info(f"F1-macro test: {results['test']['f1_macro']:.4f}")

if __name__ == "__main__":
    main()