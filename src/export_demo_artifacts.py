"""
Run this ONCE after all three models are trained/evaluated, before running
the Streamlit demo.

Exports:
1. models/pca_detector_{fd}.pkl -- the fitted PCA detector, so the demo uses
   the EXACT same detector as the reported evaluation results.
2. demo/sample_engines_{fd}.parquet -- a handful of representative test
   engines (full sequences + RUL), small enough to commit to the repo.
3. demo/thresholds_{fd}.json -- the exact detection thresholds already
   computed during evaluation, so the demo matches documented results
   rather than recomputing thresholds from a smaller bundled sample.

Usage:
    python -m src.export_demo_artifacts --config config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.baselines import PCADetector
from src.lstm_autoencoder import get_feature_cols

N_DEMO_ENGINES = 6


def export_pca_detector(cfg: dict, train_healthy: pd.DataFrame, feature_cols: list[str], fd: str):
    detector = PCADetector(n_components=cfg["pca_baseline"]["n_components"])
    detector.fit(train_healthy[feature_cols].values)
    Path("models").mkdir(exist_ok=True)
    out_path = f"models/pca_detector_{fd}.pkl"
    detector.save(out_path)
    print(f"Saved {out_path} ({detector.k_} components)")


def export_sample_engines(test_df: pd.DataFrame, window_size: int, fd: str, n_engines: int = N_DEMO_ENGINES):
    """Picks the N engines closest to actual failure at truncation (lowest
    final RUL), so every demo engine shows a real degradation trajectory --
    not a spread across the whole RUL range, which can include engines
    truncated while still mostly healthy and showing no real story."""
    engine_summary = test_df.groupby("unit_number").agg(
        length=("time_in_cycles", "count"),
        final_rul=("RUL", "min"),
    )

    MIN_LENGTH = window_size + 20  # ensures a decent number of scorable windows
    candidates = engine_summary[engine_summary["length"] >= MIN_LENGTH].sort_values("final_rul")

    chosen = candidates.head(n_engines).index.tolist()

    sample = test_df[test_df.unit_number.isin(chosen)].copy()
    Path("demo").mkdir(exist_ok=True)
    out_path = f"demo/sample_engines_{fd}.parquet"
    sample.to_parquet(out_path, index=False)
    print(f"Saved {out_path} ({len(chosen)} engines: {chosen})")
    print(f"  final RULs: {engine_summary.loc[chosen, 'final_rul'].to_dict()}")


def export_thresholds(fd: str):
    thresholds = {}

    pca_path = Path(f"reports/pca_baseline_{fd}_results.json")
    if pca_path.exists():
        pca = json.loads(pca_path.read_text())
        thresholds["pca_spe"] = pca["spe_threshold"]
        thresholds["pca_t2"] = pca["t2_threshold"]

    lstm_path = Path(f"reports/lstm_ae_{fd}_results.json")
    if lstm_path.exists():
        lstm = json.loads(lstm_path.read_text())
        thresholds["lstm"] = lstm["threshold"]

    tf_path = Path(f"reports/transformer_ae_{fd}_results.json")
    if tf_path.exists():
        tf = json.loads(tf_path.read_text())
        thresholds["transformer"] = tf["threshold"]

    if not thresholds:
        raise RuntimeError(
            "No results JSONs found in reports/ -- run the evaluation scripts first."
        )

    Path("demo").mkdir(exist_ok=True)
    out_path = f"demo/thresholds_{fd}.json"
    with open(out_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Saved {out_path}: {thresholds}")


def main(cfg: dict):
    fd = cfg["data"]["fd"]
    processed_dir = Path(cfg["data"]["processed_dir"]) / fd
    window_size = cfg["data"]["window_size"]

    train_healthy = pd.read_parquet(processed_dir / "train_healthy.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    feature_cols = get_feature_cols(train_healthy)

    export_pca_detector(cfg, train_healthy, feature_cols, fd)
    export_sample_engines(test_df, window_size, fd)
    export_thresholds(fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)