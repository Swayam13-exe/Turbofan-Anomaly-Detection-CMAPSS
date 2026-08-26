"""
Streamlit demo: compares all three detectors (PCA, LSTM autoencoder,
Transformer autoencoder) on the same test engine, showing how each one's
anomaly score evolves as the engine approaches failure.

Usage:
    streamlit run src/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml
import plotly.graph_objects as go

from src.baselines import PCADetector
from src.lstm_autoencoder import get_feature_cols, compute_rolling_reconstruction_error, LSTMAutoencoder
from src.transformer_autoencoder import TransformerAutoencoder

st.set_page_config(page_title="Turbofan Anomaly Detection", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    fd = cfg["data"]["fd"]

    demo_engines = pd.read_parquet(f"demo/sample_engines_{fd}.parquet")
    thresholds = json.loads(Path(f"demo/thresholds_{fd}.json").read_text())

    pca_detector = PCADetector.load(f"models/pca_detector_{fd}.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lstm_ckpt = torch.load(f"models/lstm_ae_{fd}_best.pt", map_location="cpu", weights_only=False)
    lstm_model = LSTMAutoencoder(
        n_features=len(lstm_ckpt["feature_cols"]),
        hidden_size=lstm_ckpt["config"]["hidden_size"],
        num_layers=lstm_ckpt["config"]["num_layers"],
        dropout=lstm_ckpt["config"]["dropout"],
    ).to(device)
    lstm_model.load_state_dict(lstm_ckpt["model_state"])
    lstm_model.eval()

    tf_ckpt = torch.load(f"models/transformer_ae_{fd}_best.pt", map_location="cpu", weights_only=False)
    tf_model = TransformerAutoencoder(
        n_features=len(tf_ckpt["feature_cols"]),
        window_size=tf_ckpt["window_size"],
        d_model=tf_ckpt["config"]["d_model"],
        nhead=tf_ckpt["config"]["nhead"],
        num_layers=tf_ckpt["config"]["num_layers"],
        dropout=tf_ckpt["config"]["dropout"],
    ).to(device)
    tf_model.load_state_dict(tf_ckpt["model_state"])
    tf_model.eval()

    feature_cols = get_feature_cols(demo_engines)
    window_size = cfg["data"]["window_size"]

    return {
        "cfg": cfg, "fd": fd, "demo_engines": demo_engines, "thresholds": thresholds,
        "pca_detector": pca_detector, "lstm_model": lstm_model, "tf_model": tf_model,
        "feature_cols": feature_cols, "window_size": window_size, "device": device,
    }


def compute_all_scores(engine_df: pd.DataFrame, artifacts: dict) -> dict:
    feature_cols = artifacts["feature_cols"]
    window_size = artifacts["window_size"]
    device = artifacts["device"]

    engine_df = engine_df.sort_values("time_in_cycles")
    X = engine_df[feature_cols].values
    ruls = engine_df["RUL"].values

    spe = artifacts["pca_detector"].spe(X)
    t2 = artifacts["pca_detector"].hotelling_t2(X)

    lstm_errors, lstm_ruls = compute_rolling_reconstruction_error(
        artifacts["lstm_model"], engine_df, feature_cols, window_size, device
    )
    tf_errors, tf_ruls = compute_rolling_reconstruction_error(
        artifacts["tf_model"], engine_df, feature_cols, window_size, device
    )

    return {
        "ruls": ruls, "spe": spe, "t2": t2,
        "lstm_ruls": lstm_ruls, "lstm_errors": lstm_errors,
        "tf_ruls": tf_ruls, "tf_errors": tf_errors,
    }


def build_comparison_chart(scores: dict, thresholds: dict) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=scores["ruls"], y=np.array(scores["spe"]) / thresholds["pca_spe"],
        mode="lines", name="PCA (SPE)", line=dict(color="#1f77b4"),
    ))
    fig.add_trace(go.Scatter(
        x=scores["ruls"], y=np.array(scores["t2"]) / thresholds["pca_t2"],
        mode="lines", name="PCA (Hotelling's T2)", line=dict(color="#2ca02c"),
    ))
    if len(scores["lstm_errors"]) > 0:
        fig.add_trace(go.Scatter(
            x=scores["lstm_ruls"], y=np.array(scores["lstm_errors"]) / thresholds["lstm"],
            mode="lines", name="LSTM Autoencoder", line=dict(color="#ff7f0e"),
        ))
    if len(scores["tf_errors"]) > 0:
        fig.add_trace(go.Scatter(
            x=scores["tf_ruls"], y=np.array(scores["tf_errors"]) / thresholds["transformer"],
            mode="lines", name="Transformer Autoencoder", line=dict(color="#d62728"),
        ))

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                   annotation_text="Detection threshold (each model, normalized)")
    fig.add_vline(x=0, line_dash="dot", line_color="black", annotation_text="Failure")

    fig.update_layout(
        xaxis_title="Remaining Useful Life (cycles) -- 0 = failure",
        yaxis_title="Anomaly score / detection threshold (log scale)",
        yaxis_type="log",
        xaxis=dict(autorange="reversed"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=550,
    )
    return fig


def main():
    st.title("Turbofan Anomaly Detection -- Model Comparison Demo")
    st.caption(
        "Compares PCA (SPE + Hotelling's T2), an LSTM autoencoder, and a Transformer "
        "autoencoder on the same engine's approach to failure. Each detector's score "
        "is normalized against its own detection threshold, so 1.0 always means "
        "'this detector would flag an anomaly here', regardless of the very different "
        "raw error scales each technique produces."
    )

    try:
        artifacts = load_artifacts()
    except (RuntimeError, FileNotFoundError) as e:
        st.error(str(e))
        st.info("Run `python -m src.export_demo_artifacts` first.")
        return

    engine_ids = sorted(artifacts["demo_engines"]["unit_number"].unique().tolist())
    engine_id = st.selectbox("Select a test engine", engine_ids)

    engine_df = artifacts["demo_engines"][artifacts["demo_engines"].unit_number == engine_id]
    final_rul = int(engine_df["RUL"].min())
    st.caption(f"Engine {engine_id}: observed for {len(engine_df)} cycles, "
               f"truncated at RUL={final_rul} cycles remaining at last observation.")

    with st.spinner("Scoring engine with all three detectors..."):
        scores = compute_all_scores(engine_df, artifacts)

    fig = build_comparison_chart(scores, artifacts["thresholds"])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Reading this chart:** the x-axis is remaining life (counting down to failure "
        "at 0, left to right). A line crossing above 1.0 means that detector's "
        "reconstruction error exceeded its own threshold at that point -- the further "
        "right that crossing happens, the earlier the warning. Notice how differently "
        "each technique behaves: PCA's SPE rarely crosses at all (low sensitivity), "
        "while the LSTM and Transformer tend to cross earlier but also drift above the "
        "threshold occasionally even well before real degradation begins (false alarms)."
    )


if __name__ == "__main__":
    main()