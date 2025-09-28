"""Preprocessing utilities for network log anomaly detection."""
from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TIMESTAMP_COL = "timestamp"
LABEL_COL = "label"


def ip_to_int(ip: str) -> int:
    """Convert IPv4 address to integer."""
    return int(ipaddress.ip_address(ip))


def ip_to_prefix(ip: str, prefix: int = 24) -> str:
    network = ipaddress.ip_network(f"{ip}/{prefix}", strict=False)
    return str(network.network_address)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features for IPs and timestamps."""
    engineered = df.copy()
    engineered[TIMESTAMP_COL] = pd.to_datetime(engineered[TIMESTAMP_COL], utc=True, errors="coerce")
    engineered[TIMESTAMP_COL] = engineered[TIMESTAMP_COL].fillna(method="ffill").fillna(method="bfill")
    if engineered[TIMESTAMP_COL].isna().any():
        engineered[TIMESTAMP_COL] = engineered[TIMESTAMP_COL].fillna(pd.Timestamp.utcnow())
    engineered["timestamp_float"] = engineered[TIMESTAMP_COL].astype("int64") / 1e9
    engineered["hour"] = engineered[TIMESTAMP_COL].dt.hour.fillna(0)
    engineered["hour_sin"] = np.sin(2 * np.pi * engineered["hour"] / 24.0)
    engineered["hour_cos"] = np.cos(2 * np.pi * engineered["hour"] / 24.0)

    for column in ["src_ip", "dst_ip"]:
        engineered[f"{column}_int"] = engineered[column].apply(ip_to_int)
        engineered[f"{column}_prefix"] = engineered[column].apply(ip_to_prefix)

    engineered = engineered.drop(columns=[TIMESTAMP_COL])
    return engineered


@dataclass
class PreprocessingArtifacts:
    pipeline: ColumnTransformer
    label_encoder: Optional[Dict[str, int]]
    label_decoder: Optional[Dict[int, str]]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, directory / "pipeline.joblib")
        joblib.dump(self.label_encoder, directory / "label_encoder.joblib")
        joblib.dump(self.label_decoder, directory / "label_decoder.joblib")

    @classmethod
    def load(cls, directory: Path) -> "PreprocessingArtifacts":
        pipeline = joblib.load(directory / "pipeline.joblib")
        label_encoder = joblib.load(directory / "label_encoder.joblib")
        label_decoder = joblib.load(directory / "label_decoder.joblib")
        return cls(pipeline=pipeline, label_encoder=label_encoder, label_decoder=label_decoder)


def build_pipeline(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    pipeline = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return pipeline


def fit_preprocess(
    df: pd.DataFrame, processed_dir: Path, fit: bool = True, pipeline: ColumnTransformer | None = None
) -> Tuple[np.ndarray, Optional[np.ndarray], PreprocessingArtifacts]:
    engineered = engineer_features(df)

    label_mapping: Optional[Dict[str, int]] = None
    label_decoder: Optional[Dict[int, str]] = None

    if LABEL_COL in engineered.columns:
        labels = engineered[LABEL_COL].astype(str)
        unique_labels = sorted(labels.unique())
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        label_decoder = {idx: label for label, idx in label_mapping.items()}
        y = labels.map(label_mapping).to_numpy(dtype=np.int64)
    else:
        y = None

    feature_columns = [col for col in engineered.columns if col != LABEL_COL]
    numeric_features = [
        "timestamp_float",
        "payload_len",
        "src_port",
        "dst_port",
        "src_ip_int",
        "dst_ip_int",
        "hour_sin",
        "hour_cos",
    ]
    categorical_features = [
        "protocol",
        "flags",
        "src_ip_prefix",
        "dst_ip_prefix",
    ]

    missing_numeric = sorted(set(numeric_features) - set(feature_columns))
    missing_categorical = sorted(set(categorical_features) - set(feature_columns))
    if missing_numeric or missing_categorical:
        raise ValueError(
            f"Missing required features. Numeric: {missing_numeric}; Categorical: {missing_categorical}"
        )

    if pipeline is None:
        pipeline = build_pipeline(numeric_features, categorical_features)

    if fit:
        features = pipeline.fit_transform(engineered)
    else:
        features = pipeline.transform(engineered)

    artifacts = PreprocessingArtifacts(pipeline=pipeline, label_encoder=label_mapping, label_decoder=label_decoder)
    artifacts.save(processed_dir)
    return features.astype(np.float32), y, artifacts


def transform_with_artifacts(df: pd.DataFrame, artifacts: PreprocessingArtifacts) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    engineered = engineer_features(df)
    if LABEL_COL in engineered.columns and artifacts.label_encoder:
        labels = engineered[LABEL_COL].astype(str).map(artifacts.label_encoder)
        labels = labels.fillna(-1).astype(int)
        y = labels.to_numpy(dtype=np.int64)
    else:
        y = None
    features = artifacts.pipeline.transform(engineered)
    return features.astype(np.float32), y
