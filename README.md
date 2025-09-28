# White-AI Network Threat Detection

This repository provides a production-ready PyTorch project for white-hat security research on network log anomaly detection and attack classification. It includes data preprocessing, sequential models (LSTM/CNN/Transformer), an autoencoder option for unsupervised anomaly detection, and utilities for training, evaluation, inference, and explainability via Captum.

## Project Structure

```
├── config.yaml             # Central configuration file
├── data/                   # Raw and processed datasets
│   └── processed/          # Cached preprocessing artifacts
├── experiments/            # Logs, checkpoints, experiment metadata
├── src/
│   ├── data_loader.py      # Data ingestion, preprocessing, dataset creation
│   ├── preprocess.py       # Feature engineering and preprocessing pipeline
│   ├── model.py            # PyTorch models (classifier + autoencoder)
│   ├── train.py            # Training script with metrics, early stopping
│   ├── evaluate.py         # Evaluation script (reports, confusion matrix)
│   ├── inference.py        # CLI for inference + explanations
│   └── utils.py            # Helpers (config, metrics, seeding, logging)
├── tests/
│   ├── test_data.py        # Unit tests for preprocessing pipeline
│   └── test_model.py       # Unit tests for model forward pass
├── requirements.txt        # Python dependencies
├── Dockerfile              # Containerization for training and inference
└── README.md
```

## Data Preparation

1. Place your raw CSV/Parquet files in `./data/` with the following columns:
   - `timestamp`, `src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol`, `payload_len`, `flags`, `label`
2. Update `config.yaml` to point to the correct training/validation/test paths.
3. The preprocessing pipeline:
   - Parses timestamps and engineers cyclical features.
   - Converts IP addresses to numeric and prefix features.
   - Normalizes numeric fields and encodes categorical fields.
   - Creates temporal sliding windows.
   - Caches processed tensors and encoders under `data/processed/`.

## Training

```bash
python -m src.train --config config.yaml
```

Features:
- Reproducible seeds
- TensorBoard logging
- Early stopping and checkpointing
- Metrics: per-class precision, recall, F1, ROC-AUC

## Evaluation

```bash
python -m src.evaluate --config config.yaml --checkpoint experiments/checkpoints/best.pt
```

Outputs include a classification report, confusion matrix (saved as PNG), and a CSV with predictions and probabilities.

## Inference & Explainability

```bash
python -m src.inference --config config.yaml --model experiments/checkpoints/best.pt \
    --input data/sample.csv --out data/preds.csv --top-k 5
```

For each sequence, the CLI returns:
- Predicted class and probability
- Probability of anomaly (1 - normal class probability or scaled reconstruction error)
- Top-k feature attributions computed using Captum Integrated Gradients

## Experiments & Reproducibility

- All runs are logged to `experiments/logs/` for visualization with TensorBoard:
  ```bash
  tensorboard --logdir experiments/logs
  ```
- Checkpoints are saved to `experiments/checkpoints/` with the best model tracked by validation F1 (classifier) or loss (autoencoder).

## Configuration

`config.yaml` centralizes data paths, model choices, training hyperparameters, and inference options. Modify it to experiment with different architectures (LSTM, 1D-CNN, Transformer) or switch between classifier and autoencoder modes.

## Docker Usage

Build and run the container for training or inference:

```bash
docker build -t white-ai .
docker run --gpus all -v $(pwd):/workspace white-ai python -m src.train --config config.yaml
```

For CPU-only environments, omit the `--gpus` flag.

## Requirements

- Python 3.10+
- Optional GPU with CUDA-capable PyTorch build for accelerated training

Install dependencies locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run unit tests to verify installation:

```bash
pytest
```

## Licensing

This project is intended for ethical security research and defensive purposes only. Ensure compliance with all applicable laws and policies when using real network data.
