"""Probability calibration used after the ranking model is selected."""
from __future__ import annotations

import numpy as np
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class BlendedCalibrator:
    """Average isotonic and logistic calibration fitted on the same holdout."""

    def fit(self, raw_probability, labels):
        raw_probability = np.asarray(raw_probability, dtype=float)
        labels = np.asarray(labels)
        if raw_probability.ndim != 1 or labels.ndim != 1 or len(raw_probability) != len(labels):
            raise ValueError("Calibration probabilities and labels must be aligned vectors")
        if not np.isfinite(raw_probability).all() or ((raw_probability < 0) | (raw_probability > 1)).any():
            raise ValueError("Calibration inputs must be probabilities")
        if len(np.unique(labels)) != 2:
            raise ValueError("Calibration requires both classes")
        self.isotonic = IsotonicRegression(out_of_bounds="clip").fit(raw_probability, labels)
        self.sigmoid = LogisticRegression(C=1e4).fit(self._log_odds(raw_probability), labels)
        return self

    def predict(self, raw_probability):
        raw_probability = np.asarray(raw_probability, dtype=float)
        if not np.isfinite(raw_probability).all() or ((raw_probability < 0) | (raw_probability > 1)).any():
            raise ValueError("Prediction inputs must be probabilities")
        isotonic = self.isotonic.predict(raw_probability)
        sigmoid = self.sigmoid.predict_proba(self._log_odds(raw_probability))[:, 1]
        return (isotonic + sigmoid) / 2

    @staticmethod
    def _log_odds(raw_probability):
        return logit(np.clip(raw_probability, 1e-6, 1 - 1e-6)).reshape(-1, 1)
