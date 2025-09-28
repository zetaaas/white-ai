"""Inference CLI for anomaly detection with explainability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader, TensorDataset

from .data_loader import load_for_inference
from .model import build_model
from .utils import get_device, is_autoencoder, load_checkpoint, load_config, setup_logging


def get_feature_names(artifacts) -> List[str]:
    try:
        return list(artifacts.pipeline.get_feature_names_out())
    except AttributeError:
        return []


def compute_attributions(
    model: torch.nn.Module,
    sequence: torch.Tensor,
    target: int | None,
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    sequence = sequence.unsqueeze(0).to(device)
    baseline = torch.zeros_like(sequence)

    if mode == "classifier":
        ig = IntegratedGradients(model)
        attributions = ig.attribute(sequence, baselines=baseline, target=target)
    else:
        class AutoencoderWrapper(torch.nn.Module):
            def __init__(self, autoencoder: torch.nn.Module):
                super().__init__()
                self.autoencoder = autoencoder

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                recon = self.autoencoder(x)
                loss = torch.mean((recon - x) ** 2, dim=(1, 2))
                return loss

        wrapper = AutoencoderWrapper(model)
        ig = IntegratedGradients(wrapper)
        attributions = ig.attribute(sequence, baselines=baseline)
    return attributions.squeeze(0).detach().cpu()


def summarize_topk(attributions: torch.Tensor, feature_names: List[str], top_k: int) -> List[Tuple[str, float]]:
    importance = attributions.abs().mean(dim=0)
    top_indices = torch.topk(importance, k=min(top_k, importance.numel())).indices.tolist()
    return [(feature_names[idx], float(importance[idx].item())) for idx in top_indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on network log sequences")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input CSV/Parquet file path")
    parser.add_argument("--out", type=str, required=True, help="Output CSV with predictions")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top contributing features")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    device = get_device(config["training"].get("device"))

    data_cfg = config["data"]
    sequences, artifacts = load_for_inference(
        Path(args.input),
        Path(data_cfg["processed_dir"]),
        window_size=data_cfg["window_size"],
        window_step=data_cfg["window_step"],
        file_format=data_cfg.get("format", "csv"),
    )

    checkpoint = load_checkpoint(Path(args.model), map_location=device)
    metadata = checkpoint.get("metadata", {})
    model_config = checkpoint.get("config", config)
    input_dim = metadata.get("input_dim", sequences.shape[-1])
    num_classes = metadata.get("num_classes", 1)
    model = build_model(model_config, input_dim, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    feature_names = get_feature_names(artifacts)
    if len(feature_names) != sequences.shape[-1]:
        feature_names = [f"feature_{i}" for i in range(sequences.shape[-1])]

    dataset = TensorDataset(torch.from_numpy(sequences).float())
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    mode = model_config["model"].get("mode", "classifier")
    normal_label = data_cfg.get("normal_label")
    normal_idx = metadata.get("label_encoder", {}).get(normal_label)

    results: List[Dict[str, Any]] = []

    for idx, (sequence_tensor,) in enumerate(loader):
        sequence_tensor = sequence_tensor.to(device)
        if not is_autoencoder(mode):
            with torch.no_grad():
                logits = model(sequence_tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            anomaly_prob = 1.0 - probs[normal_idx] if normal_idx is not None else 1.0 - probs[pred_idx]
            attributions = compute_attributions(model, sequence_tensor.squeeze(0), pred_idx, "classifier", device)
            top_features = summarize_topk(attributions, feature_names, args.top_k)
            results.append(
                {
                    "sequence_id": idx,
                    "pred_label": pred_idx,
                    "pred_label_name": metadata.get("label_decoder", {}).get(pred_idx, str(pred_idx)),
                    "max_prob": float(probs[pred_idx]),
                    "anomaly_prob": float(anomaly_prob),
                    "top_features": json.dumps(top_features),
                }
            )
        else:
            with torch.no_grad():
                recon = model(sequence_tensor)
                error = torch.mean((recon - sequence_tensor) ** 2, dim=(1, 2)).cpu().numpy()[0]
            mean_error = metadata.get("val_mean_error", float(error))
            std_error = metadata.get("val_std_error", 1.0)
            score = (error - mean_error) / (std_error + 1e-8)
            anomaly_prob = float(1.0 / (1.0 + np.exp(-score)))
            attributions = compute_attributions(model, sequence_tensor.squeeze(0), None, "autoencoder", device)
            top_features = summarize_topk(attributions, feature_names, args.top_k)
            results.append(
                {
                    "sequence_id": idx,
                    "reconstruction_error": float(error),
                    "anomaly_prob": anomaly_prob,
                    "top_features": json.dumps(top_features),
                }
            )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
