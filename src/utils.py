"""Utility helpers for configuration management, metrics, and reproducibility."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)


LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure root logging with a simple format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration from ``path``."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return ``Path`` object."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def compute_classification_metrics(
    y_true: Iterable[int], y_pred: Iterable[int], y_prob: np.ndarray, num_classes: int
) -> Dict[str, Any]:
    """Compute precision, recall, F1 per class and ROC-AUC."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    metrics = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
    }
    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except ValueError:
        roc_auc = float("nan")
    metrics["roc_auc"] = roc_auc
    return metrics


def generate_classification_report(
    y_true: Iterable[int], y_pred: Iterable[int], target_names: Iterable[str]
) -> str:
    """Return a formatted classification report."""
    return classification_report(y_true, y_pred, target_names=list(target_names), zero_division=0)


def compute_confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def get_device(device_str: str | None = None) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.lower() == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def count_model_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    torch.save(state, path)


def load_checkpoint(path: Path, map_location: str | torch.device | None = None) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def is_autoencoder(mode: str) -> bool:
    return mode.lower() == "autoencoder"


__all__ = [
    "setup_logging",
    "set_seed",
    "load_config",
    "ensure_dir",
    "save_json",
    "compute_classification_metrics",
    "generate_classification_report",
    "compute_confusion_matrix",
    "get_device",
    "count_model_parameters",
    "save_checkpoint",
    "load_checkpoint",
    "is_autoencoder",
]
