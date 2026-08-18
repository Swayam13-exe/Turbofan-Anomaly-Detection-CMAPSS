"""
LSTM autoencoder for reconstruction-based anomaly detection.

Trained ONLY on healthy cycles (train_healthy.parquet, produced by
data_pipeline.py) using fixed-length sliding windows built within each
engine's own consecutive cycles (windows never cross engine boundaries).
The decoder reconstructs the input window from a single compressed latent
vector (the encoder's final hidden state, repeated across the decoder's
timesteps) -- a standard sequence-autoencoder architecture (Malhotra et al.,
2016). This is a RECONSTRUCTION model, not a forecasting model: it rebuilds
its own input window rather than predicting a future one.

IMPORTANT: unlike a prior project in this portfolio, unit_number here does
NOT identify the same physical entity across train/test splits -- engine
"1" in training and engine "1" in test are different physical engines that
happen to share a number. No per-unit embedding is used; the model must
learn general degradation patterns.

At evaluation time, reconstruction error is computed for the window ENDING
at each cycle (once at least `window_size` cycles have been observed for
that engine), giving one anomaly score per cycle -- directly comparable to
the PCA baseline's per-cycle SPE/T2 scores.

Usage:
    python -m src.lstm_autoencoder --config config.yaml --mode train
    python -m src.lstm_autoencoder --config config.yaml --mode evaluate
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.metrics import (
    mean_detection_lead_time,
    binarize_true_labels,
    precision_recall_f1,
)

FEATURE_PREFIX = ("op_setting_", "sensor_")


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]


def split_engines(unit_ids: list[int], val_fraction: float, seed: int = 42) -> tuple[list[int], list[int]]:
    """Splits by ENGINE, not by window -- windows from the same engine share
    information, so validating on entirely unseen engines is the only way
    to honestly assess generalization."""
    rng = np.random.RandomState(seed)
    shuffled = list(unit_ids)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


class WindowDataset(Dataset):
    """All valid consecutive windows of length window_size within each
    engine's OWN sequence -- windows never cross engine boundaries."""

    def __init__(self, df: pd.DataFrame, feature_cols: list[str], window_size: int, unit_ids: list[int]):
        self.window_size = window_size
        self.samples = []

        for uid in unit_ids:
            engine_df = df[df.unit_number == uid].sort_values("time_in_cycles")
            X = engine_df[feature_cols].values.astype(np.float32)
            n = len(X)
            for start in range(0, n - window_size + 1):
                self.samples.append(X[start:start + window_size])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx])


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0)
        self.output_proj = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        _, (h, c) = self.encoder(x)

        latent = h[-1]  # (batch, hidden_size)
        decoder_input = latent.unsqueeze(1).repeat(1, seq_len, 1)

        decoder_out, _ = self.decoder(decoder_input, (h, c))
        reconstruction = self.output_proj(decoder_out)
        return reconstruction


def train(cfg: dict):
    fd = cfg["data"]["fd"]
    processed_dir = Path(cfg["data"]["processed_dir"]) / fd
    window_size = cfg["data"]["window_size"]
    ae_cfg = cfg["lstm_autoencoder"]

    train_healthy = pd.read_parquet(processed_dir / "train_healthy.parquet")
    feature_cols = get_feature_cols(train_healthy)
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    unit_ids = sorted(train_healthy["unit_number"].unique())
    train_units, val_units = split_engines(unit_ids, ae_cfg.get("val_engine_fraction", 0.2))
    print(f"Engines: {len(train_units)} train, {len(val_units)} validation")

    train_ds = WindowDataset(train_healthy, feature_cols, window_size, train_units)
    val_ds = WindowDataset(train_healthy, feature_cols, window_size, val_units)
    print(f"Windows: {len(train_ds)} train, {len(val_ds)} validation (window_size={window_size})")

    train_loader = DataLoader(train_ds, batch_size=ae_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=ae_cfg["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LSTMAutoencoder(
        n_features=len(feature_cols),
        hidden_size=ae_cfg["hidden_size"],
        num_layers=ae_cfg["num_layers"],
        dropout=ae_cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=ae_cfg["learning_rate"])
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    Path("models").mkdir(exist_ok=True)

    for epoch in range(ae_cfg["max_epochs"]):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{ae_cfg['max_epochs']} [train]")
        for batch in train_bar:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
            train_bar.set_postfix(batch_loss=f"{loss.item():.4f}")
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                val_loss += criterion(recon, batch).item() * batch.size(0)
        val_loss /= len(val_ds)

        print(f"Epoch {epoch+1}/{ae_cfg['max_epochs']}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "feature_cols": feature_cols,
                "window_size": window_size,
                "config": ae_cfg,
            }, f"models/lstm_ae_{fd}_best.pt")
            print(f"  -> new best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= ae_cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
                break

    print(f"Training complete. Best model saved to models/lstm_ae_{fd}_best.pt")


def compute_rolling_reconstruction_error(model, engine_df: pd.DataFrame, feature_cols: list[str],
                                          window_size: int, device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (errors, ruls) for every cycle from window_size onward in this
    engine's sequence -- one score per cycle, using the window ENDING at
    that cycle."""
    engine_df = engine_df.sort_values("time_in_cycles")
    X = engine_df[feature_cols].values.astype(np.float32)
    ruls = engine_df["RUL"].values
    n = len(X)

    if n < window_size:
        return np.array([]), np.array([])

    windows = np.stack([X[i:i + window_size] for i in range(n - window_size + 1)])
    windows_t = torch.tensor(windows).to(device)

    model.eval()
    with torch.no_grad():
        recon = model(windows_t).cpu().numpy()

    errors = np.mean((windows - recon) ** 2, axis=(1, 2))
    aligned_ruls = ruls[window_size - 1:]
    return errors, aligned_ruls


def evaluate(cfg: dict, checkpoint_path: str | None = None):
    fd = cfg["data"]["fd"]
    processed_dir = Path(cfg["data"]["processed_dir"]) / fd

    if checkpoint_path is None:
        checkpoint_path = f"models/lstm_ae_{fd}_best.pt"
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_cols = checkpoint["feature_cols"]
    window_size = checkpoint["window_size"]
    ae_cfg = checkpoint["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(
        n_features=len(feature_cols),
        hidden_size=ae_cfg["hidden_size"],
        num_layers=ae_cfg["num_layers"],
        dropout=ae_cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    train_healthy = pd.read_parquet(processed_dir / "train_healthy.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    print("Computing threshold from healthy training windows ...")
    healthy_errors = []
    for uid in tqdm(train_healthy["unit_number"].unique(), desc="Scoring healthy engines"):
        engine_df = train_healthy[train_healthy.unit_number == uid]
        errors, _ = compute_rolling_reconstruction_error(model, engine_df, feature_cols, window_size, device)
        healthy_errors.append(errors)
    healthy_errors = np.concatenate([e for e in healthy_errors if len(e) > 0])

    pct = cfg["evaluation"]["threshold_percentile"]
    threshold = float(np.percentile(healthy_errors, pct))
    print(f"Reconstruction error threshold (p{pct} of healthy data): {threshold:.6f}")

    print("Scoring test engines ...")
    per_engine_errors = []
    per_engine_ruls = []
    all_y_true = []
    all_y_pred = []
    anomaly_window = cfg["evaluation"]["anomaly_window"]

    for uid in tqdm(test_df["unit_number"].unique(), desc="Evaluating test engines"):
        engine_df = test_df[test_df.unit_number == uid]
        errors, ruls = compute_rolling_reconstruction_error(model, engine_df, feature_cols, window_size, device)
        if len(errors) == 0:
            continue
        per_engine_errors.append(errors)
        per_engine_ruls.append(ruls)

        y_true = binarize_true_labels(ruls, anomaly_window)
        y_pred = (errors >= threshold).astype(int)
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

    persistence = cfg["evaluation"].get("persistence_cycles", 3)
    lead_time_result = mean_detection_lead_time(per_engine_errors, per_engine_ruls, threshold, persistence=persistence)

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)
    prf_result = precision_recall_f1(y_true_all, y_pred_all)

    results = {**lead_time_result, **prf_result, "threshold": threshold, "window_size": window_size}
    print("\nLSTM autoencoder results:", results)

    import json
    Path("reports").mkdir(exist_ok=True)
    out_path = f"reports/lstm_ae_{fd}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.mode == "train":
        train(cfg)
    else:
        evaluate(cfg, checkpoint_path=args.checkpoint)