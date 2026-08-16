"""
Evaluates the PCA baseline (SPE + Hotelling's T2) on C-MAPSS test data.

Both SPE and Hotelling's T2 are evaluated independently and reported
separately, since they catch different kinds of anomalies (SPE: deviation
outside the healthy correlation structure; T2: unusual combinations within
it) -- this keeps the comparison honest rather than collapsing two
different signals into one number.

Thresholds are set entirely from HEALTHY training data (a fixed percentile),
never from test data, to avoid leakage. Detection lead time requires a
SUSTAINED crossing (persistence_cycles consecutive cycles above threshold),
not a single spike -- otherwise noise-driven false triggers inflate
apparent lead time on any detector with a nonzero false-positive rate.

Usage:
    python -m src.evaluate_pca_baseline --config config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.baselines import PCADetector
from src.metrics import (
    mean_detection_lead_time,
    binarize_true_labels,
    precision_recall_f1,
)

FEATURE_PREFIX = ("op_setting_", "sensor_")


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]


def evaluate_detector(score_fn, test_df: pd.DataFrame, feature_cols: list[str],
                       threshold: float, anomaly_window: int, persistence: int) -> dict:
    per_engine_errors = []
    per_engine_ruls = []
    all_y_true = []
    all_y_pred = []

    for unit_id, engine_df in test_df.groupby("unit_number"):
        engine_df = engine_df.sort_values("time_in_cycles")
        X = engine_df[feature_cols].values
        ruls = engine_df["RUL"].values

        scores = score_fn(X)
        per_engine_errors.append(scores)
        per_engine_ruls.append(ruls)

        y_true = binarize_true_labels(ruls, anomaly_window)
        y_pred = (scores >= threshold).astype(int)
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

    lead_time_result = mean_detection_lead_time(per_engine_errors, per_engine_ruls, threshold, persistence=persistence)

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)
    prf_result = precision_recall_f1(y_true_all, y_pred_all)

    return {**lead_time_result, **prf_result}


def main(cfg: dict):
    fd = cfg["data"]["fd"]
    processed_dir = Path(cfg["data"]["processed_dir"]) / fd

    train_healthy = pd.read_parquet(processed_dir / "train_healthy.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    feature_cols = get_feature_cols(train_healthy)
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    print("Fitting PCA on healthy training data ...")
    detector = PCADetector(n_components=cfg["pca_baseline"]["n_components"])
    detector.fit(train_healthy[feature_cols].values)
    print(f"  retained {detector.k_} components")

    healthy_spe = detector.spe(train_healthy[feature_cols].values)
    healthy_t2 = detector.hotelling_t2(train_healthy[feature_cols].values)
    pct = cfg["evaluation"]["threshold_percentile"]
    spe_threshold = float(np.percentile(healthy_spe, pct))
    t2_threshold = float(np.percentile(healthy_t2, pct))
    print(f"SPE threshold (p{pct} of healthy data): {spe_threshold:.4f}")
    print(f"T2 threshold (p{pct} of healthy data): {t2_threshold:.4f}")

    anomaly_window = cfg["evaluation"]["anomaly_window"]
    persistence = cfg["evaluation"].get("persistence_cycles", 3)
    print(f"Using persistence={persistence} (require this many consecutive cycles above threshold before counting a detection)")

    print("\nEvaluating SPE detector on test set ...")
    spe_results = evaluate_detector(
        lambda X: detector.spe(X), test_df, feature_cols, spe_threshold, anomaly_window, persistence
    )
    print("SPE results:", spe_results)

    print("\nEvaluating Hotelling's T2 detector on test set ...")
    t2_results = evaluate_detector(
        lambda X: detector.hotelling_t2(X), test_df, feature_cols, t2_threshold, anomaly_window, persistence
    )
    print("T2 results:", t2_results)

    results = {
        "spe": spe_results,
        "t2": t2_results,
        "spe_threshold": spe_threshold,
        "t2_threshold": t2_threshold,
        "n_components": detector.k_,
        "n_features": len(feature_cols),
        "persistence_cycles": persistence,
    }

    Path("reports").mkdir(exist_ok=True)
    out_path = f"reports/pca_baseline_{fd}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)