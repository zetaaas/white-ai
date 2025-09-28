"""Evaluation script producing detailed metrics and reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from .data_loader import load_datasets
from .model import build_model
from .train import collate_fn
from .utils import (
    compute_classification_metrics,
    compute_confusion_matrix,
    ensure_dir,
    generate_classification_report,
    get_device,
    is_autoencoder,
    load_checkpoint,
    load_config,
    setup_logging,
)


def evaluate_classifier(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    with torch.no_grad():
        for sequences, labels in loader:
            if labels is None:
                continue
            sequences = sequences.to(device)
            logits = model(sequences)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    probs_concat = np.concatenate(all_probs, axis=0)
    labels_concat = np.concatenate(all_labels, axis=0)
    preds = probs_concat.argmax(axis=1)
    return probs_concat, labels_concat, preds


def evaluate_autoencoder(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    model.eval()
    criterion = torch.nn.MSELoss(reduction="none")
    errors: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            recon = model(sequences)
            loss = criterion(recon, sequences).mean(dim=(1, 2))
            errors.append(loss.cpu().numpy())
            if labels is not None:
                labels_list.append(labels.numpy())
    error_array = np.concatenate(errors, axis=0)
    labels_array = np.concatenate(labels_list, axis=0) if labels_list else None
    return error_array, labels_array


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model on the test set")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="experiments/evaluation", help="Where to store reports")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(config["logging"].get("checkpoint_dir", "experiments/checkpoints")) / "best.pt"
    )

    device = get_device(config["training"].get("device"))
    train_dataset, val_dataset, test_dataset, label_encoder, label_decoder = load_datasets(config)

    checkpoint = load_checkpoint(Path(checkpoint_path), map_location=device)
    metadata = checkpoint.get("metadata", {})
    model_config = checkpoint.get("config", config)
    input_dim = metadata.get("input_dim", test_dataset.num_features)
    num_classes = metadata.get("num_classes", len(label_encoder))
    model = build_model(model_config, input_dim, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    output_dir = ensure_dir(args.output_dir)

    test_loader = DataLoader(test_dataset, batch_size=config["training"].get("batch_size", 64), shuffle=False, collate_fn=collate_fn)

    if not is_autoencoder(model_config["model"].get("mode", "classifier")):
        probs, labels, preds = evaluate_classifier(model, test_loader, device, num_classes)
        metrics = compute_classification_metrics(labels, preds, probs, num_classes)
        label_decoder_meta: Dict[int, str] = metadata.get("label_decoder", {})
        report = generate_classification_report(labels, preds, [label_decoder_meta.get(i, str(i)) for i in range(num_classes)])
        confusion = compute_confusion_matrix(labels, preds)

        normal_label_name = config["data"].get("normal_label")
        normal_label_idx = metadata.get("label_encoder", {}).get(normal_label_name)
        anomaly_prob = 1.0 - probs[:, normal_label_idx] if normal_label_idx is not None else 1.0 - probs.max(axis=1)

        pred_df = pd.DataFrame(
            {
                "sequence_id": np.arange(len(preds)),
                "true_label": labels,
                "pred_label": preds,
                "true_label_name": [label_decoder_meta.get(i, str(i)) for i in labels],
                "pred_label_name": [label_decoder_meta.get(i, str(i)) for i in preds],
                "max_prob": probs.max(axis=1),
                "anomaly_prob": anomaly_prob,
            }
        )
        pred_csv_path = output_dir / "predictions.csv"
        pred_df.to_csv(pred_csv_path, index=False)

        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        report_path = output_dir / "classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png")
        plt.close()

        print("Evaluation complete. Metrics saved to", output_dir)
    else:
        val_loader = DataLoader(
            val_dataset, batch_size=config["training"].get("batch_size", 64), shuffle=False, collate_fn=collate_fn
        )
        val_errors, val_labels = evaluate_autoencoder(model, val_loader, device)
        mean_error = float(np.mean(val_errors))
        std_error = float(np.std(val_errors) + 1e-8)
        threshold = mean_error + 3 * std_error

        test_errors, test_labels = evaluate_autoencoder(model, test_loader, device)
        scores = (test_errors - mean_error) / std_error
        anomaly_prob = 1.0 / (1.0 + np.exp(-scores))
        predictions = (test_errors > threshold).astype(int)

        normal_label_name = config["data"].get("normal_label")
        normal_idx = metadata.get("label_encoder", {}).get(normal_label_name)
        if test_labels is not None and normal_idx is not None:
            binary_labels = (test_labels != normal_idx).astype(int)
            precision = np.sum((predictions == 1) & (binary_labels == 1)) / max(np.sum(predictions == 1), 1)
            recall = np.sum((predictions == 1) & (binary_labels == 1)) / max(np.sum(binary_labels == 1), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            metrics = {"precision": precision, "recall": recall, "f1": f1}
        else:
            metrics = {"mean_val_error": mean_error, "std_val_error": std_error}

        pred_df = pd.DataFrame(
            {
                "sequence_id": np.arange(len(test_errors)),
                "reconstruction_error": test_errors,
                "anomaly_probability": anomaly_prob,
                "predicted_anomaly": predictions,
            }
        )
        pred_df.to_csv(output_dir / "autoencoder_predictions.csv", index=False)
        with open(output_dir / "autoencoder_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print("Autoencoder evaluation complete. Outputs saved to", output_dir)


if __name__ == "__main__":
    main()
