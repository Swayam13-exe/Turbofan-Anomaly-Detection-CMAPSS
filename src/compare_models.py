"""
Pulls together PCA, LSTM autoencoder, and Transformer autoencoder results
into one comparison table (CSV + markdown) and a summary plot.

Usage:
    python -m src.compare_models --config config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import yaml

METRIC_COLS = ["mean_lead_time", "median_lead_time", "miss_rate", "precision", "recall", "f1", "false_positive_rate"]


def load_results(fd: str) -> pd.DataFrame:
    rows = []

    pca_path = Path(f"reports/pca_baseline_{fd}_results.json")
    if pca_path.exists():
        pca = json.loads(pca_path.read_text())
        for name, key in [("PCA (SPE)", "spe"), ("PCA (Hotelling's T2)", "t2")]:
            rows.append({"model": name, **{c: pca[key].get(c) for c in METRIC_COLS}})
    else:
        print(f"Warning: {pca_path} not found, skipping PCA results")

    lstm_path = Path(f"reports/lstm_ae_{fd}_results.json")
    if lstm_path.exists():
        lstm = json.loads(lstm_path.read_text())
        rows.append({"model": "LSTM Autoencoder", **{c: lstm.get(c) for c in METRIC_COLS}})
    else:
        print(f"Warning: {lstm_path} not found, skipping LSTM results")

    tf_path = Path(f"reports/transformer_ae_{fd}_results.json")
    if tf_path.exists():
        tf = json.loads(tf_path.read_text())
        rows.append({"model": "Transformer Autoencoder", **{c: tf.get(c) for c in METRIC_COLS}})
    else:
        print(f"Warning: {tf_path} not found, skipping Transformer results")

    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Manual markdown table writer -- avoids adding a `tabulate` dependency
    just for this one script."""
    cols = df.columns.tolist()
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        formatted = []
        for c in cols:
            v = row[c]
            formatted.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def plot_comparison(df: pd.DataFrame, fd: str, out_dir: str = "reports/figures"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(df["model"], df["mean_lead_time"], color="#1f77b4")
    axes[0].set_title("Mean Detection Lead Time (cycles)")
    axes[0].set_ylabel("Cycles before failure")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(df["model"], df["f1"], color="#ff7f0e")
    axes[1].set_title("F1 Score")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(df["model"], df["false_positive_rate"], color="#d62728")
    axes[2].set_title("False Positive Rate")
    axes[2].tick_params(axis="x", rotation=30)

    fig.suptitle(f"Model Comparison -- {fd}")
    fig.tight_layout()
    out_path = f"{out_dir}/model_comparison_{fd}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main(cfg: dict):
    fd = cfg["data"]["fd"]
    df = load_results(fd)

    if df.empty:
        print("No results found -- run the baseline/model evaluation scripts first.")
        return

    print("\nModel comparison:")
    print(df.to_string(index=False))

    Path("reports").mkdir(exist_ok=True)
    csv_path = f"reports/model_comparison_{fd}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    md_path = f"reports/model_comparison_{fd}.md"
    with open(md_path, "w") as f:
        f.write(dataframe_to_markdown(df))
    print(f"Saved {md_path}")

    plot_comparison(df, fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)