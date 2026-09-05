import numpy as np
from src.metrics import (
    detection_lead_time,
    mean_detection_lead_time,
    binarize_true_labels,
    precision_recall_f1,
    reconstruction_error,
)


def test_detection_lead_time_finds_first_crossing():
    errors = np.array([0.1, 0.1, 0.5, 0.6, 0.9])
    ruls = np.array([40, 30, 20, 10, 0])
    lt = detection_lead_time(errors, ruls, threshold=0.5)
    assert lt == 20


def test_detection_lead_time_none_when_never_crosses():
    errors = np.array([0.1, 0.1, 0.1])
    ruls = np.array([20, 10, 0])
    lt = detection_lead_time(errors, ruls, threshold=0.9)
    assert lt is None


def test_mean_detection_lead_time_aggregates_correctly():
    errors_list = [np.array([0.1, 0.9]), np.array([0.1, 0.1])]
    ruls_list = [np.array([10, 0]), np.array([10, 0])]
    result = mean_detection_lead_time(errors_list, ruls_list, threshold=0.5)
    assert result["n_detected"] == 1
    assert result["n_missed"] == 1
    assert result["mean_lead_time"] == 0.0


def test_binarize_true_labels():
    ruls = np.array([50, 20, 15, 5, 0])
    labels = binarize_true_labels(ruls, anomaly_window=15)
    assert list(labels) == [0, 0, 1, 1, 1]


def test_precision_recall_f1_perfect_prediction():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    result = precision_recall_f1(y_true, y_pred)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["false_positive_rate"] == 0.0


def test_precision_recall_f1_all_wrong():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    result = precision_recall_f1(y_true, y_pred)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_reconstruction_error_zero_when_perfect():
    actual = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = reconstruction_error(actual, actual)
    assert np.allclose(result, 0.0)


def test_reconstruction_error_positive_when_different():
    actual = np.array([[1.0, 2.0]])
    reconstructed = np.array([[2.0, 3.0]])
    result = reconstruction_error(actual, reconstructed)
    assert result[0] == 1.0