"""评估模块：AUC / EER / Accuracy / Precision / Recall / F1（开题报告 3.5）。"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def _check_inputs(y_true: Sequence, y_score: Sequence) -> tuple:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if y_true.size != y_score.size:
        raise ValueError(f"标签与分数长度不一致: {y_true.size} vs {y_score.size}")
    if y_true.size == 0:
        raise ValueError("空输入无法评估")
    return y_true, y_score


def compute_eer(y_true: Sequence, y_score: Sequence) -> tuple[float, float]:
    """返回 (EER, 对应阈值)，其中分数越高越倾向"伪造"。"""
    from sklearn.metrics import roc_curve

    y_true, y_score = _check_inputs(y_true, y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(fpr[idx]), float(thresholds[idx])


def compute_metrics(
    y_true: Sequence,
    y_score: Sequence,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """综合指标；threshold 缺省时用 0.5。"""
    from sklearn.metrics import roc_auc_score

    y_true, y_score = _check_inputs(y_true, y_score)
    thr = 0.5 if threshold is None else float(threshold)
    pred = (y_score >= thr).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)

    auc = float("nan")
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_score))
    eer, eer_thr = compute_eer(y_true, y_score)

    return {
        "auc": auc,
        "eer": eer,
        "eer_threshold": eer_thr,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": thr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "num_samples": int(y_true.size),
        "num_real": int((y_true == 0).sum()),
        "num_fake": int((y_true == 1).sum()),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    keys = ["auc", "eer", "accuracy", "precision", "recall", "f1"]
    return " | ".join(f"{k}={metrics[k]:.4f}" for k in keys if k in metrics)
