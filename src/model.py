"""PyTorch models for network anomaly detection and attack classification."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        logits = self.fc(self.dropout(last_hidden))
        return logits


class CNNClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, dropout: float, num_classes: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=input_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        repeated = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        recon, _ = self.decoder(repeated)
        return recon


def build_model(config: Dict[str, Any], input_dim: int, num_classes: int) -> nn.Module:
    mode = config["model"].get("mode", "classifier")
    architecture = config["model"].get("architecture", "lstm")
    hidden_dim = config["model"].get("hidden_dim", 128)
    num_layers = config["model"].get("num_layers", 2)
    dropout = config["model"].get("dropout", 0.1)
    num_heads = config["model"].get("num_heads", 4)

    if mode == "classifier":
        if architecture == "lstm":
            return LSTMClassifier(input_dim, hidden_dim, num_layers, dropout, num_classes)
        if architecture == "cnn":
            return CNNClassifier(input_dim, hidden_dim, num_classes)
        if architecture == "transformer":
            return TransformerClassifier(input_dim, hidden_dim, num_heads, num_layers, dropout, num_classes)
        raise ValueError(f"Unsupported classifier architecture: {architecture}")

    if mode == "autoencoder":
        if architecture != "lstm":
            raise ValueError("Autoencoder mode currently supports only LSTM architecture")
        return LSTMAutoencoder(input_dim, hidden_dim, num_layers, dropout)

    raise ValueError(f"Unknown mode: {mode}")


__all__ = [
    "build_model",
    "LSTMClassifier",
    "CNNClassifier",
    "TransformerClassifier",
    "LSTMAutoencoder",
]
