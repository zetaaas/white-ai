"""Data loading and preprocessing pipeline for network logs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import preprocess
from .utils import ensure_dir


class NetworkLogsDataset(Dataset[Tuple[torch.Tensor, Optional[torch.Tensor]]]):
    """PyTorch dataset wrapping sequential network features."""

    def __init__(self, sequences: np.ndarray, labels: Optional[np.ndarray] = None):
        self.sequences = torch.from_numpy(sequences).float()
        self.labels = None if labels is None else torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        sequence = self.sequences[idx]
        label = None if self.labels is None else self.labels[idx]
        return sequence, label

    @property
    def num_features(self) -> int:
        return self.sequences.shape[-1]


def read_dataframe(path: Path, file_format: str) -> pd.DataFrame:
    if file_format.lower() == "csv":
        return pd.read_csv(path)
    if file_format.lower() in {"parquet", "pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported format: {file_format}")


def create_sliding_windows(
    features: np.ndarray,
    labels: Optional[np.ndarray],
    window_size: int,
    step: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if features.ndim != 2:
        raise ValueError("Features must be 2D array before windowing")

    num_samples = features.shape[0]
    if num_samples < window_size:
        raise ValueError("Not enough samples to create a single window")

    sequences: List[np.ndarray] = []
    sequence_labels: List[int] = []
    for start in range(0, num_samples - window_size + 1, step):
        end = start + window_size
        sequences.append(features[start:end])
        if labels is not None:
            sequence_labels.append(int(labels[end - 1]))

    sequences_arr = np.stack(sequences, axis=0)
    labels_arr = np.array(sequence_labels, dtype=np.int64) if labels is not None else None
    return sequences_arr, labels_arr


def save_processed_arrays(
    sequences: np.ndarray,
    labels: Optional[np.ndarray],
    path: Path,
) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(path, sequences=sequences, labels=labels if labels is not None else np.array([]))


def load_processed_arrays(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    sequences = data["sequences"]
    labels_arr = data["labels"]
    labels = labels_arr if labels_arr.size > 0 else None
    return sequences, labels


def prepare_dataset(
    df: pd.DataFrame,
    processed_dir: Path,
    split_name: str,
    window_size: int,
    window_step: int,
    artifacts: Optional[preprocess.PreprocessingArtifacts] = None,
    fit: bool = False,
) -> Tuple[NetworkLogsDataset, preprocess.PreprocessingArtifacts]:
    if artifacts is None or fit:
        features, labels, artifacts = preprocess.fit_preprocess(df, processed_dir, fit=True)
    else:
        features, labels = preprocess.transform_with_artifacts(df, artifacts)

    sequences, seq_labels = create_sliding_windows(features, labels, window_size, window_step)
    save_processed_arrays(sequences, seq_labels, processed_dir / f"{split_name}_processed.npz")
    dataset = NetworkLogsDataset(sequences, seq_labels)
    return dataset, artifacts


def load_datasets(
    config: Dict[str, Any]
) -> Tuple[NetworkLogsDataset, NetworkLogsDataset, NetworkLogsDataset, Dict[str, int], Dict[int, str]]:
    data_cfg = config["data"]
    processed_dir = ensure_dir(data_cfg["processed_dir"])
    window_size = int(data_cfg["window_size"])
    window_step = int(data_cfg["window_step"])
    file_format = data_cfg.get("format", "csv")

    train_df = read_dataframe(Path(data_cfg["train_path"]), file_format)
    val_df = read_dataframe(Path(data_cfg["val_path"]), file_format)
    test_df = read_dataframe(Path(data_cfg["test_path"]), file_format)

    train_dataset, artifacts = prepare_dataset(
        train_df,
        processed_dir,
        "train",
        window_size,
        window_step,
        artifacts=None,
        fit=True,
    )

    val_dataset, artifacts = prepare_dataset(
        val_df,
        processed_dir,
        "val",
        window_size,
        window_step,
        artifacts=artifacts,
        fit=False,
    )

    test_dataset, artifacts = prepare_dataset(
        test_df,
        processed_dir,
        "test",
        window_size,
        window_step,
        artifacts=artifacts,
        fit=False,
    )

    label_encoder = artifacts.label_encoder or {}
    label_decoder = artifacts.label_decoder or {}
    return train_dataset, val_dataset, test_dataset, label_encoder, label_decoder


def load_for_inference(
    input_path: Path,
    processed_dir: Path,
    window_size: int,
    window_step: int,
    file_format: str = "csv",
) -> Tuple[np.ndarray, preprocess.PreprocessingArtifacts]:
    df = read_dataframe(input_path, file_format)
    artifacts = preprocess.PreprocessingArtifacts.load(processed_dir)
    features, _ = preprocess.transform_with_artifacts(df, artifacts)
    sequences, _ = create_sliding_windows(features, None, window_size, window_step)
    return sequences, artifacts
