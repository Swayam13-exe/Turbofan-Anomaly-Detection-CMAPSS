"""
Data pipeline for NASA C-MAPSS Turbofan Engine Degradation Simulation dataset.

Source: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
(look for "Turbofan Engine Degradation Simulation Data Set")

Raw file format (train_FD00X.txt, test_FD00X.txt):
    Space-separated, no header, one row per (engine, cycle):
    unit_number, time_in_cycles, op_setting_1..3, sensor_1..21  (26 columns)

    - train_FD00X.txt: each engine runs to failure (last row = failure cycle)
    - test_FD00X.txt: each engine truncated at some point BEFORE failure
    - RUL_FD00X.txt: one value per engine in the test set -- the true
      remaining cycles at the point of truncation (ground truth,
      evaluation-only, never used for training)

This script:
    1. Loads train/test/RUL files for a given sub-dataset (FD001-FD004)
    2. Computes RUL for every row (train: max_cycle - current_cycle; test:
       RUL_file value + cycles remaining within the truncated sequence)
    3. Identifies near-constant sensors (near-zero variance in training
       data) and drops them -- they carry no degradation signal and only
       add normalization noise
    4. Z-score normalizes using stats from HEALTHY cycles only (the first
       `healthy_fraction` of each training engine's life) -- the
       autoencoder should learn "what normal looks like," so it must be
       normalized against data that doesn't already include degradation
    5. Writes processed parquet files: train_full (complete run-to-failure,
       for building evaluation windows), train_healthy (what the
       autoencoder actually trains on), test (truncated sequences + RUL)

Usage:
    python src/data_pipeline.py --fd FD001 --raw_dir data/raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_OP_SETTINGS = 3
N_SENSORS = 21
CONSTANT_VAR_THRESHOLD = 1e-5  # sensors with variance below this are dropped

COLUMN_NAMES = (
    ["unit_number", "time_in_cycles"]
    + [f"op_setting_{i}" for i in range(1, N_OP_SETTINGS + 1)]
    + [f"sensor_{i}" for i in range(1, N_SENSORS + 1)]
)


def load_raw_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLUMN_NAMES)]  # drop any trailing empty columns
    df.columns = COLUMN_NAMES
    return df


def add_train_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    """RUL at each cycle = (engine's final cycle) - (current cycle)."""
    df = train_df.copy()
    max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["RUL"] = max_cycle - df["time_in_cycles"]
    return df


def add_test_rul(test_df: pd.DataFrame, rul_file: pd.DataFrame) -> pd.DataFrame:
    """Test engines are truncated before failure. RUL_FD00X.txt gives the
    true remaining cycles AT THE POINT OF TRUNCATION for each engine. RUL at
    any earlier cycle in the truncated sequence = that value + cycles
    remaining within the truncated sequence itself."""
    df = test_df.copy()
    rul_at_truncation = rul_file.reset_index(drop=True)
    rul_at_truncation.index = rul_at_truncation.index + 1  # engine IDs are 1-indexed
    rul_map = rul_at_truncation[0].to_dict()

    max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
    cycles_remaining_in_sequence = max_cycle - df["time_in_cycles"]
    final_rul = df["unit_number"].map(rul_map)
    df["RUL"] = cycles_remaining_in_sequence + final_rul
    return df


def identify_constant_sensors(train_df: pd.DataFrame) -> list[str]:
    sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]
    variances = train_df[sensor_cols].var()
    constant = variances[variances < CONSTANT_VAR_THRESHOLD].index.tolist()
    return constant


def compute_healthy_stats(train_df: pd.DataFrame, feature_cols: list[str],
                           healthy_fraction: float) -> tuple[pd.Series, pd.Series]:
    """Mean/std computed ONLY from the first `healthy_fraction` of each
    engine's life -- normalizing against data the model is meant to treat
    as 'normal', not the full run-to-failure trajectory which already
    includes degradation."""
    df = train_df.copy()
    max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["life_fraction"] = df["time_in_cycles"] / max_cycle
    healthy = df[df["life_fraction"] <= healthy_fraction]

    mean = healthy[feature_cols].mean()
    std = healthy[feature_cols].std().replace(0, 1e-6)
    return mean, std


def normalize(df: pd.DataFrame, feature_cols: list[str], mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = (df[feature_cols] - mean) / std
    return df


def main(fd: str, raw_dir: str, out_dir: str, healthy_fraction: float):
    raw_dir = Path(raw_dir) / fd    # e.g. data/raw/FD001/
    out_dir = Path(out_dir) / fd    # e.g. data/processed/FD001/
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {fd} train/test/RUL files ...")
    train_df = load_raw_file(raw_dir / f"train_{fd}.txt")
    test_df = load_raw_file(raw_dir / f"test_{fd}.txt")
    rul_df = pd.read_csv(raw_dir / f"RUL_{fd}.txt", sep=r"\s+", header=None)

    print("Computing RUL labels ...")
    train_df = add_train_rul(train_df)
    test_df = add_test_rul(test_df, rul_df)

    print("Identifying near-constant sensors ...")
    constant_sensors = identify_constant_sensors(train_df)
    print(f"  dropping {len(constant_sensors)} constant sensors: {constant_sensors}")
    train_df = train_df.drop(columns=constant_sensors)
    test_df = test_df.drop(columns=constant_sensors)

    feature_cols = [c for c in train_df.columns if c.startswith("op_setting_") or c.startswith("sensor_")]

    print(f"Computing normalization stats from healthy cycles (first {healthy_fraction*100:.0f}% of life) ...")
    mean, std = compute_healthy_stats(train_df, feature_cols, healthy_fraction)

    train_norm = normalize(train_df, feature_cols, mean, std)
    test_norm = normalize(test_df, feature_cols, mean, std)

    max_cycle = train_norm.groupby("unit_number")["time_in_cycles"].transform("max")
    life_fraction = train_norm["time_in_cycles"] / max_cycle
    train_healthy = train_norm[life_fraction <= healthy_fraction].copy()

    print(f"  train (full): {train_norm.shape}, train (healthy-only): {train_healthy.shape}, test: {test_norm.shape}")

    train_norm.to_parquet(out_dir / "train_full.parquet", index=False)
    train_healthy.to_parquet(out_dir / "train_healthy.parquet", index=False)
    test_norm.to_parquet(out_dir / "test.parquet", index=False)
    mean.to_frame("mean").join(std.to_frame("std")).to_csv(out_dir / "norm_stats.csv")

    print(f"Wrote processed files to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=str, default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--healthy_fraction", type=float, default=0.3)
    args = parser.parse_args()

    main(args.fd, args.raw_dir, args.out_dir, args.healthy_fraction)