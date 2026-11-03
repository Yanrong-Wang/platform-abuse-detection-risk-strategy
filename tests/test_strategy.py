import numpy as np
import pytest
from itertools import product

from abuse_risk.strategy import (Costs, actions_from_thresholds, budgeted_expected_loss_actions,
                                 compare_policies, evaluate, optimize, top_k_actions)


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
    assert len(compare_policies(y, p, Costs(), 1/3, cuts)) == 5


def test_budgeted_review_uses_positive_expected_benefit_without_labels():
    p = np.array([0.01, 0.14, 0.5, 0.8, 0.99])
    costs = Costs(false_positive=1, review=0.1, review_accuracy=1)
    actions = budgeted_expected_loss_actions(p, costs, capacity_fraction=0.4)
    assert actions.tolist() == [0, 0, 1, 1, 2]
    assert np.count_nonzero(actions == 1) == 2
    assert budgeted_expected_loss_actions(p, costs, 0).tolist() == [0, 0, 0, 2, 2]
    with pytest.raises(ValueError):
        budgeted_expected_loss_actions(np.array([1.2]), costs, 0.2)


def test_budgeted_policy_minimizes_estimated_additive_loss():
    p = np.array([0.03, 0.23, 0.47, 0.78, 0.96])
    costs = Costs(false_negative=2, false_positive=3, review=0.12, review_accuracy=0.9)
    chosen = budgeted_expected_loss_actions(p, costs, capacity_fraction=0.4)

    def estimated_loss(actions):
        allow = p * costs.false_negative
        block = (1 - p) * costs.false_positive
        review = costs.review + (1 - costs.review_accuracy) * (allow + block)
        return np.choose(actions, [allow, review, block]).sum()

    best = min(estimated_loss(np.array(actions)) for actions in product(range(3), repeat=len(p))
               if actions.count(1) <= 2)
    assert estimated_loss(chosen) == pytest.approx(best)


def test_invalid_costs_and_thresholds():
    with pytest.raises(ValueError):
        Costs(review_accuracy=1.1)
    with pytest.raises(ValueError):
        actions_from_thresholds(np.array([0.5]), 0.8, 0.2)
