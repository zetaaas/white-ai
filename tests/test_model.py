import torch

from src.model import build_model


def test_classifier_forward_pass():
    config = {
        "model": {
            "mode": "classifier",
            "architecture": "lstm",
            "hidden_dim": 32,
            "num_layers": 1,
            "dropout": 0.1,
        }
    }
    model = build_model(config, input_dim=16, num_classes=3)
    inputs = torch.randn(4, 5, 16)
    outputs = model(inputs)
    assert outputs.shape == (4, 3)


def test_autoencoder_forward_pass():
    config = {
        "model": {
            "mode": "autoencoder",
            "architecture": "lstm",
            "hidden_dim": 32,
            "num_layers": 1,
            "dropout": 0.1,
        }
    }
    model = build_model(config, input_dim=16, num_classes=3)
    inputs = torch.randn(4, 5, 16)
    outputs = model(inputs)
    assert outputs.shape == (4, 5, 16)
