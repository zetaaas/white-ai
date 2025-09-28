import pandas as pd
import torch

from src.data_loader import load_datasets


def create_mock_dataframe(num_rows: int = 20) -> pd.DataFrame:
    base = {
        "timestamp": pd.date_range("2024-01-01", periods=num_rows, freq="min"),
        "src_ip": [f"192.168.0.{i % 255}" for i in range(num_rows)],
        "dst_ip": [f"10.0.0.{(i * 3) % 255}" for i in range(num_rows)],
        "src_port": [1000 + i for i in range(num_rows)],
        "dst_port": [2000 + (i % 5) for i in range(num_rows)],
        "protocol": ["TCP" if i % 2 == 0 else "UDP" for i in range(num_rows)],
        "payload_len": [500 + i * 10 for i in range(num_rows)],
        "flags": ["S" if i % 2 == 0 else "A" for i in range(num_rows)],
        "label": ["normal" if i % 3 else "attack" for i in range(num_rows)],
    }
    return pd.DataFrame(base)


def test_load_datasets_creates_sequences(tmp_path):
    df = create_mock_dataframe()
    train_path = tmp_path / "train.csv"
    df.to_csv(train_path, index=False)

    config = {
        "data": {
            "train_path": str(train_path),
            "val_path": str(train_path),
            "test_path": str(train_path),
            "format": "csv",
            "processed_dir": str(tmp_path / "processed"),
            "window_size": 4,
            "window_step": 2,
            "normal_label": "normal",
        }
    }

    train_dataset, val_dataset, test_dataset, label_encoder, label_decoder = load_datasets(config)

    assert len(train_dataset) > 0
    assert train_dataset.sequences.shape[1] == config["data"]["window_size"]
    assert not torch.isnan(train_dataset.sequences).any()
    assert train_dataset.labels is not None
    assert isinstance(label_encoder, dict)
    assert isinstance(label_decoder, dict)
