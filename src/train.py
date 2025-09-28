"""Training script for network anomaly detection models."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data_loader import NetworkLogsDataset, load_datasets
from .model import build_model
from .utils import (
    compute_classification_metrics,
    ensure_dir,
    get_device,
    is_autoencoder,
    load_config,
    save_checkpoint,
    set_seed,
    setup_logging,
)


def collate_fn(batch: List[Tuple[torch.Tensor, Optional[torch.Tensor]]]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    sequences, labels = zip(*batch)
    stacked_sequences = torch.stack(sequences)
    if labels[0] is None:
        return stacked_sequences, None
    stacked_labels = torch.stack([label for label in labels if label is not None])
    return stacked_sequences, stacked_labels


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, Dict[str, Any]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            if labels is None:
                continue
            labels = labels.to(device)
            logits = model(sequences)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())
    avg_loss = float(np.mean(losses)) if losses else math.inf
    probs_concat = np.concatenate(all_probs, axis=0)
    labels_concat = np.concatenate(all_labels, axis=0)
    preds = probs_concat.argmax(axis=1)
    metrics = compute_classification_metrics(labels_concat, preds, probs_concat, num_classes)
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    config: Dict[str, Any],
    num_classes: int,
    checkpoint_dir: Path,
    metadata: Dict[str, Any],
) -> None:
    criterion = nn.CrossEntropyLoss()
    patience = config["training"].get("patience", 5)
    gradient_clip = config["training"].get("gradient_clip", None)
    best_score = -math.inf
    epochs_without_improvement = 0

    for epoch in range(1, config["training"]["num_epochs"] + 1):
        model.train()
        epoch_losses = []
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            if labels is None:
                continue
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            if gradient_clip:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_loss, val_metrics = evaluate_classifier(model, val_loader, device, num_classes)
        macro_f1 = float(np.mean(val_metrics["f1"]))

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("F1/val_macro", macro_f1, epoch)
        writer.flush()

        if macro_f1 > best_score:
            best_score = macro_f1
            epochs_without_improvement = 0
            checkpoint_path = checkpoint_dir / "best.pt"
            metadata.update({"best_epoch": epoch, "val_macro_f1": macro_f1})
            save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "metadata": metadata,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"macro_f1={macro_f1:.4f}, roc_auc={val_metrics['roc_auc']:.4f}"
        )

    writer.close()


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    config: Dict[str, Any],
    checkpoint_dir: Path,
    metadata: Dict[str, Any],
) -> None:
    criterion = nn.MSELoss()
    patience = config["training"].get("patience", 5)
    gradient_clip = config["training"].get("gradient_clip", None)
    best_loss = math.inf
    epochs_without_improvement = 0

    for epoch in range(1, config["training"]["num_epochs"] + 1):
        model.train()
        epoch_losses = []
        for sequences, _ in train_loader:
            sequences = sequences.to(device)
            optimizer.zero_grad()
            recon = model(sequences)
            loss = criterion(recon, sequences)
            loss.backward()
            if gradient_clip:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        model.eval()
        val_losses = []
        val_errors_all: List[float] = []
        with torch.no_grad():
            for sequences, _ in val_loader:
                sequences = sequences.to(device)
                recon = model(sequences)
                loss = criterion(recon, sequences)
                val_losses.append(loss.item())
                errors = ((recon - sequences) ** 2).mean(dim=(1, 2))
                val_errors_all.extend(errors.cpu().numpy().tolist())
        val_loss = float(np.mean(val_losses)) if val_losses else math.inf

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.flush()

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            checkpoint_path = checkpoint_dir / "best.pt"
            if val_errors_all:
                metadata.update(
                    {
                        "val_mean_error": float(np.mean(val_errors_all)),
                        "val_std_error": float(np.std(val_errors_all) + 1e-8),
                    }
                )
            metadata.update({"best_epoch": epoch, "val_loss": val_loss})
            save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "metadata": metadata,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train network anomaly detection model")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    train_dataset, val_dataset, test_dataset, label_encoder, label_decoder = load_datasets(config)
    num_classes = len(label_encoder)
    if num_classes == 0:
        raise ValueError("No labels found in training data")

    device = get_device(config["training"].get("device"))

    model = build_model(config, train_dataset.num_features, num_classes)
    model.to(device)

    train_batch_size = config["training"].get("batch_size", 64)
    val_batch_size = config["training"].get("batch_size", 64)

    data_cfg = config["data"]
    normal_label_name = data_cfg.get("normal_label")
    normal_label_idx = label_encoder.get(normal_label_name, None)

    if is_autoencoder(config["model"]["mode"]):
        if normal_label_idx is None:
            raise ValueError("Autoencoder mode requires 'normal_label' present in label encoder")
        train_mask = train_dataset.labels == normal_label_idx
        val_mask = val_dataset.labels == normal_label_idx
        train_sequences = train_dataset.sequences[train_mask].cpu().numpy()
        val_sequences = val_dataset.sequences[val_mask].cpu().numpy()
        train_dataset = NetworkLogsDataset(train_sequences, None)
        val_dataset = NetworkLogsDataset(val_sequences, None)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = Adam(
        model.parameters(),
        lr=config["training"].get("learning_rate", 1e-3),
        weight_decay=config["training"].get("weight_decay", 0.0),
    )

    log_dir = ensure_dir(config["logging"].get("log_dir", "experiments/logs"))
    checkpoint_dir = ensure_dir(config["logging"].get("checkpoint_dir", "experiments/checkpoints"))
    writer = SummaryWriter(log_dir=str(log_dir))

    metadata: Dict[str, Any] = {
        "label_encoder": label_encoder,
        "label_decoder": label_decoder,
        "input_dim": train_dataset.num_features,
        "num_classes": num_classes,
        "normal_label": normal_label_name,
    }

    if is_autoencoder(config["model"]["mode"]):
        train_autoencoder(
            model,
            train_loader,
            val_loader,
            device,
            optimizer,
            writer,
            config,
            checkpoint_dir,
            metadata,
        )
    else:
        train_classifier(
            model,
            train_loader,
            val_loader,
            device,
            optimizer,
            writer,
            config,
            num_classes,
            checkpoint_dir,
            metadata,
        )


if __name__ == "__main__":
    main()
