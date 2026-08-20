"""
Evaluation metrics for anomaly/failure detection -- deliberately different
from the point-forecast metrics (MAE/RMSE/SMAPE) used in prior projects,
since this is a detection problem, not a regression problem.
"""
from __future__ import annotations

import numpy as np


def detection_lead_time(reconstruction_errors: np.ndarray, ruls: np.ndarray, threshold: float,
                         persistence: int = 1) -> float | None:
    """Given a single engine's per-cycle reconstruction errors (chronological
    order) and the corresponding RUL at each cycle, returns the RUL at the
    FIRST cycle where error crosses the threshold AND STAYS above it for at
    least `persistence` consecutive cycles -- i.e. "how many cycles of
    warning did the model give before failure." Returns None if no sustained
    crossing occurred (a missed detection).

    persistence=1 (default) reduces to a single-cycle crossing, which is
    vulnerable to noise-driven false triggers inflating apparent lead time
    on detectors with any meaningful false-positive rate -- persistence=3
    or higher is recommended whenever the detector's healthy-data FPR is
    much above ~1%, so a lucky early noise spike doesn't get counted as a
    genuine early warning.
    """
    above = reconstruction_errors >= threshold
    if persistence <= 1:
        crossed = np.where(above)[0]
        if len(crossed) == 0:
            return None
        return float(ruls[crossed[0]])

    for i in range(len(above) - persistence + 1):
        if above[i:i + persistence].all():
            return float(ruls[i])
    return None


def mean_detection_lead_time(per_engine_errors: list[np.ndarray], per_engine_ruls: list[np.ndarray],
                              threshold: float, persistence: int = 1) -> dict:
    """Aggregates detection_lead_time across all engines. Returns mean/median
    lead time among engines where a (sustained) detection occurred, plus the
    fraction of engines where the model never crossed the threshold at all
    (missed)."""
    lead_times = []
    n_missed = 0
    for errors, ruls in zip(per_engine_errors, per_engine_ruls):
        lt = detection_lead_time(errors, ruls, threshold, persistence=persistence)
        if lt is None:
            n_missed += 1
        else:
            lead_times.append(lt)

    return {
        "mean_lead_time": float(np.mean(lead_times)) if lead_times else None,
        "median_lead_time": float(np.median(lead_times)) if lead_times else None,
        "n_detected": len(lead_times),
        "n_missed": n_missed,
        "miss_rate": n_missed / (len(lead_times) + n_missed) if (lead_times or n_missed) else None,
    }


def binarize_true_labels(ruls: np.ndarray, anomaly_window: int) -> np.ndarray:
    """True label = 1 (anomalous) for any cycle within `anomaly_window`
    cycles of failure (RUL <= anomaly_window), else 0 (healthy)."""
    return (ruls <= anomaly_window).astype(int)


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def reconstruction_error(actual: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """Per-sample mean squared reconstruction error across features."""
    return np.mean((actual - reconstructed) ** 2, axis=-1)