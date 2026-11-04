import numpy as np
import pytest

from abuse_risk.calibration import BlendedCalibrator


def test_blended_calibrator_outputs_probabilities_and_rejects_bad_inputs():
    raw = np.array([0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9, 0.99])
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    calibrated = BlendedCalibrator().fit(raw, labels).predict(raw)
    assert np.all((0 <= calibrated) & (calibrated <= 1))
    assert np.all(np.diff(calibrated) >= 0)
    with pytest.raises(ValueError):
        BlendedCalibrator().fit(raw, np.zeros(len(raw)))
    with pytest.raises(ValueError):
        BlendedCalibrator().fit(raw, labels).predict(np.array([1.2]))
