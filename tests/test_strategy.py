import numpy as np
import pytest

from abuse_risk.strategy import Costs, actions_from_thresholds, compare_policies, evaluate, optimize, top_k_actions


def test_expected_review_errors_and_cost():
    y = np.array([1, 0, 1, 0])
    actions = np.array([0, 1, 1, 2])
    result = evaluate(y, actions, Costs(false_negative=2, false_positive=5,
                                        review=0.5, review_accuracy=0.8))
    assert result["expected_fn"] == pytest.approx(1.2)
    assert result["expected_fp"] == pytest.approx(1.2)
    assert result["cost"] == pytest.approx(9.4)


def test_capacity_and_monotonic_actions():
    p = np.array([0.01, 0.1, 0.25, 0.5, 0.8, 0.99])
    y = np.array([0, 0, 0, 1, 1, 1])
    cuts = optimize(y, p, Costs(), capacity_fraction=1/3, grid_size=8)
    action = actions_from_thresholds(p, cuts["s3_low"], cuts["s3_high"])
    assert np.count_nonzero(action == 1) <= 2
    assert np.all(np.diff(action) >= 0)
    assert top_k_actions(p, 1/3).sum() == 2
    assert len(compare_policies(y, p, Costs(), 1/3, cuts)) == 4


def test_invalid_costs_and_thresholds():
    with pytest.raises(ValueError):
        Costs(review_accuracy=1.1)
    with pytest.raises(ValueError):
        actions_from_thresholds(np.array([0.5]), 0.8, 0.2)
