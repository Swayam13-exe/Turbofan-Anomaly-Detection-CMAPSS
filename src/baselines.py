"""
PCA-based statistical process control baseline for anomaly detection.

Fits PCA on HEALTHY-only training data (mirrors how the autoencoder models
will be trained later). Two complementary detection statistics are computed
for any new sample:

- SPE (Squared Prediction Error, aka Q-statistic): reconstruction error
  using only the retained principal components -- literally a LINEAR
  autoencoder, which makes this baseline a fair, directly comparable
  reference point once the LSTM/Transformer autoencoders are built (same
  reconstruction-error framing, just linear vs. nonlinear).
- Hotelling's T-squared: measures whether a sample's position WITHIN the
  retained principal-component subspace is unusual (an abnormal
  *combination* of otherwise-normal-looking sensor values) -- something SPE
  alone cannot catch, since SPE only measures deviation OUTSIDE the modeled
  subspace.

This is a classical statistical-process-control technique (Kresta et al.,
1991) -- a genuinely different technique family from anything used in
Projects 1-3 of this portfolio (no gradient boosting, no attention/RNNs
here -- purely linear algebra).
"""
from __future__ import annotations

import numpy as np


class PCADetector:
    def __init__(self, n_components: int | float = 0.95):
        """n_components: if int, exact number of components to retain. If
        float in (0, 1), retain enough components to explain that fraction
        of variance (matches sklearn's PCA convention)."""
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None  # (k, n_features)
        self.explained_variance_: np.ndarray | None = None  # (k,)
        self.k_: int | None = None

    def fit(self, X: np.ndarray) -> "PCADetector":
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        eigenvalues = (S ** 2) / max(X.shape[0] - 1, 1)

        if isinstance(self.n_components, float):
            explained_ratio = eigenvalues / eigenvalues.sum()
            cumulative = np.cumsum(explained_ratio)
            k = int(np.searchsorted(cumulative, self.n_components) + 1)
            k = min(k, len(eigenvalues))
        else:
            k = min(self.n_components, len(eigenvalues))
        k = max(k, 1)

        self.components_ = Vt[:k]
        self.explained_variance_ = np.clip(eigenvalues[:k], 1e-8, None)  # avoid div-by-zero in T2
        self.k_ = k
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project onto retained principal components -> scores, shape (n, k)."""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        scores = self.transform(X)
        return scores @ self.components_ + self.mean_

    def spe(self, X: np.ndarray) -> np.ndarray:
        """Squared Prediction Error -- reconstruction error using only the
        retained components. Rises when a sample doesn't fit the normal
        correlation structure learned from healthy data."""
        reconstructed = self.reconstruct(X)
        return np.sum((X - reconstructed) ** 2, axis=1)

    def hotelling_t2(self, X: np.ndarray) -> np.ndarray:
        """Hotelling's T-squared -- how unusual a sample's position is
        WITHIN the retained principal-component subspace, normalized by
        each component's variance."""
        scores = self.transform(X)
        return np.sum((scores ** 2) / self.explained_variance_, axis=1)