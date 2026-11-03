"""Metrics for scoring and finite review capacity."""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)


def ranking_metrics(y, p, fraction=0.05):
    y, p = np.asarray(y), np.asarray(p)
    k = max(1, int(np.ceil(len(y) * fraction)))
    picked = np.argsort(-p, kind="stable")[:k]
    return {"pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "precision_at_5pct": float(y[picked].mean()),
            "recall_at_5pct": float(y[picked].sum() / max(y.sum(), 1))}


def curve_points(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    return {"precision": precision.tolist(), "recall": recall.tolist(),
            "thresholds": thresholds.tolist()}
