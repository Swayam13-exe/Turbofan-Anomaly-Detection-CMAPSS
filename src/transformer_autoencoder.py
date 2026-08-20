"""
Transformer autoencoder for reconstruction-based anomaly detection.

Directly motivated by a specific finding from the LSTM autoencoder: doubling
its hidden_size (32 -> 64) produced almost no change in val_loss or any
evaluation metric, which pointed away from "needs more capacity" and toward
a structural limitation -- the LSTM decoder reconstructs the entire window
from a SINGLE fixed vector (the encoder's final hidden state, repeated
across every decoder timestep). No matter how large that vector is, forcing
30 timesteps of information through one compressed summary is lossy by
design.

This model is built specifically to test whether removing that bottleneck
helps: the encoder produces a FULL sequence of contextualized
representations (one per input timestep, not pooled down to one vector),
and the decoder reconstructs the window using cross-attention to that
entire sequence -- so it can reference any specific encoded timestep
directly, not just a single blended summary. The decoder's "queries" carry
no information about the actual input values (only learned positional
embeddings), so it is still forced to pull everything from the encoder via
attention rather than trivially copying the input through.

Reuses WindowDataset, split_engines, get_feature_cols, and
compute_rolling_reconstruction_error from lstm_autoencoder.py, since none
of that logic depends on the model architecture -- only train()/evaluate()
and the model class itself differ here.

Usage:
    python -m src.transformer_autoencoder --config config.yaml --mode train
    python -m src.transformer_autoencoder --config config.yaml --mode evaluate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.lstm_autoencoder import (
    get_feature_cols,
    split_engines,
    WindowDataset,
    compute_rolling_reconstruction_error,
)
from src.metrics import (
    mean_detection_lead_time,
    binarize_true_labels,
    precision_recall_f1,
)


class TransformerAutoencoder(nn.Module):
    def __init__(self, n_features: int, window_size: int, d_model: int = 32,
                 nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder_queries = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, n_features)

    def forward(self, x):
        batch_size = x.size(0)
        x_proj = self.input_proj(x) + self.pos_encoding
        memory = self.encoder(x_proj)

        queries = self.decoder_queries.repeat(batch_size, 1, 1)
        decoded = self.decoder(queries, memory)

        reconstruction = self.output_proj(decoded)
        return reconstruction


def train(cfg: dict):
    fd = cfg["data"]["fd"]
    processed_dir = Path(cfg["data"]["processed_dir"]) / fd
    window_size = cfg["data"]["window_size"]
    tf_cfg = cfg["transformer_autoencoder"]

    train_healthy = pd.read_parquet(processed_dir / "train_healthy.parquet")
    feature_cols = get_feature_cols(train_healthy)
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    unit_ids = sorted(train_healthy["unit_number"].unique())
    train_units, val_units = split_engines(unit_ids, tf_cfg.get("val_engine_fraction", 0.2))
    print(f"Engines: {len(train_units)} train, {len(val_units)} validation")

    train_ds = WindowDataset(train_healthy, feature_cols, window_size, train_units)
    val_ds = WindowDataset(train_healthy, feature_cols, window_size, val_units)
    print(f"Windows: {len(train_ds)} train, {len(val_ds)} validation (window_size={window_size})")

    train_loader = DataLoader(train_ds, batch_size=tf_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tf_cfg["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TransformerAutoencoder(
        n_features=len(feature_cols),
        window_size=window_size,
        d_model=tf_cfg["d_model"],
        nhead=tf_cfg["nhead"],
        num_layers=tf_cfg["num_layers"],
        dropout=tf_cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=tf_cfg["learning_rate"])
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    Path("models").mkdir(exist_ok=True)

    for epoch in range(tf_cfg["max_epochs"]):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{tf_cfg['max_epochs']} [train]")
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

        print(f"Epoch {epoch+1}/{tf_cfg['max_epochs']}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "feature_cols": feature_cols,
                "window_size": window_size,
                "config": tf_cfg,
            }, f"models/transformer_ae_{fd}_best.pt")
            print(f"  -> new best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= tf_cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
                break

    print(f"Training complete. Best model saved to models/transformer_ae_{fd}_best.pt")


def evaluate(cfg: dict, checkpoint_path: str | None = None):
    fd = cfg["data"]["fd"]
    processed_dir = Path(cfg["data"]["processed_dir"]) / fd

    if checkpoint_path is None:
        checkpoint_path = f"models/transformer_ae_{fd}_best.pt"
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_cols = checkpoint["feature_cols"]
    window_size = checkpoint["window_size"]
    tf_cfg = checkpoint["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerAutoencoder(
        n_features=len(feature_cols),
        window_size=window_size,
        d_model=tf_cfg["d_model"],
        nhead=tf_cfg["nhead"],
        num_layers=tf_cfg["num_layers"],
        dropout=tf_cfg["dropout"],
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
    print("\nTransformer autoencoder results:", results)

    Path("reports").mkdir(exist_ok=True)
    out_path = f"reports/transformer_ae_{fd}_results.json"
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