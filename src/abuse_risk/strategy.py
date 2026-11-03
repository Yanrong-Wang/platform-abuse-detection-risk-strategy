"""Offline three-action strategy with explicit review accuracy and cost assumptions."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Costs:
    false_negative: float = 1.0
    false_positive: float = 5.0
    review: float = 0.15
    review_accuracy: float = 1.0

    def __post_init__(self):
        if min(self.false_negative, self.false_positive, self.review) < 0:
            raise ValueError("Costs must be nonnegative")
        if not 0 <= self.review_accuracy <= 1:
            raise ValueError("review_accuracy must be in [0, 1]")


def actions_from_thresholds(p, low: float, high: float):
    if low > high:
        raise ValueError("low threshold cannot exceed high threshold")
    p = np.asarray(p)
    return np.where(p >= high, 2, np.where(p >= low, 1, 0))


def evaluate(y, actions, costs: Costs):
    """Expected offline outcomes; review decisions are correct with probability q."""
    y, actions = np.asarray(y), np.asarray(actions)
    if len(y) != len(actions) or not np.isin(actions, [0, 1, 2]).all():
        raise ValueError("Invalid action array")
    allow, review, block = actions == 0, actions == 1, actions == 2
    q = costs.review_accuracy
    fn = float((allow & (y == 1)).sum() + (1 - q) * (review & (y == 1)).sum())
    fp = float((block & (y == 0)).sum() + (1 - q) * (review & (y == 0)).sum())
    positives, negatives = max(int((y == 1).sum()), 1), max(int((y == 0).sum()), 1)
    detected = positives - fn
    return {"allow": int(allow.sum()), "review": int(review.sum()), "block": int(block.sum()),
            "expected_fn": fn, "expected_fp": fp, "expected_recall": float(detected / positives),
            "expected_fpr": float(fp / negatives),
            "cost": float(costs.false_negative * fn + costs.false_positive * fp + costs.review * review.sum()),
            "cost_per_1000": float(1000 * (costs.false_negative * fn + costs.false_positive * fp + costs.review * review.sum()) / len(y))}


def top_k_actions(p, fraction):
    p = np.asarray(p)
    k = min(len(p), max(0, int(np.floor(len(p) * fraction))))
    actions = np.zeros(len(p), dtype=np.int8)
    actions[np.argsort(-p, kind="stable")[:k]] = 1
    return actions


def optimize(y, p, costs: Costs, capacity_fraction: float, grid_size: int = 60):
    """Select S1 and S3 cutoffs on a strategy-validation split only."""
    if not 0 <= capacity_fraction <= 1:
        raise ValueError("capacity_fraction must be in [0,1]")
    p = np.asarray(p)
    # Include endpoints so all-allow and all-block remain feasible.
    cuts = np.unique(np.r_[-np.inf, np.quantile(p, np.linspace(0, 1, grid_size)), np.inf])
    best_s1 = (float("inf"), None)
    best_s3 = (float("inf"), None)
    max_review = int(np.floor(len(p) * capacity_fraction))
    for high in cuts:
        s1 = evaluate(y, actions_from_thresholds(p, high, high), costs)
        if s1["cost"] < best_s1[0]:
            best_s1 = (s1["cost"], float(high))
        for low in cuts[cuts <= high]:
            action = actions_from_thresholds(p, low, high)
            if np.count_nonzero(action == 1) > max_review:
                continue
            result = evaluate(y, action, costs)
            if result["cost"] < best_s3[0]:
                best_s3 = (result["cost"], (float(low), float(high)))
    return {"s1_threshold": best_s1[1], "s3_low": best_s3[1][0],
            "s3_high": best_s3[1][1], "capacity_fraction": capacity_fraction}


def compare_policies(y, p, costs: Costs, capacity_fraction: float, thresholds):
    policies = {"S0 allow all": np.zeros(len(p), dtype=np.int8),
                "S1 block threshold": actions_from_thresholds(p, thresholds["s1_threshold"], thresholds["s1_threshold"]),
                "S2 top-K review": top_k_actions(p, capacity_fraction),
                "S3 allow/review/block": actions_from_thresholds(p, thresholds["s3_low"], thresholds["s3_high"])}
    return {name: evaluate(y, action, costs) for name, action in policies.items()}
