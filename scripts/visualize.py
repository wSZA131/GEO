"""可视化：ROC、分数分布、延迟对比、鲁棒性曲线（开题报告 3.3 第 6 行"可视化"）。

用法：
    python scripts/visualize.py --eval-json runs/audio/eval_test.json \
        --bench-json runs/audio/bench.json --out runs/figures
    python scripts/visualize.py --eval-json runs/exp1/eval_val.json \
        --robustness-json runs/wild/robustness_val.json --out runs/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo_deepfake.utils import ensure_dir, setup_logger  # noqa: E402


def load_scores(payload: dict) -> tuple[np.ndarray, np.ndarray] | None:
    if "per_sample" in payload:
        data = payload["per_sample"]
        return np.array([d["label"] for d in data]), np.array([d["score"] for d in data])
    if "per_video" in payload:
        data = payload["per_video"]
        return np.array([d["label"] for d in data]), np.array([d["fake_prob"] for d in data])
    return None


def plot_roc(labels, scores, metrics: dict, out: Path) -> None:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {metrics.get('auc', float('nan')):.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    eer = metrics.get("eer")
    if eer is not None and eer == eer:
        ax.scatter([eer], [1 - eer], c="crimson", zorder=5, label=f"EER = {eer * 100:.2f}%")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "roc.png", dpi=150)
    plt.close(fig)


def plot_score_dist(labels, scores, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores[labels == 0], bins=40, alpha=0.65, label="real / bonafide", density=True)
    ax.hist(scores[labels == 1], bins=40, alpha=0.65, label="fake / spoof", density=True)
    ax.set_xlabel("P(fake)")
    ax.set_ylabel("density")
    ax.set_title("Score distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "score_distribution.png", dpi=150)
    plt.close(fig)


def plot_latency(bench: dict, out: Path) -> None:
    backends = bench.get("backends", {})
    if not backends:
        return
    names = list(backends)
    p50 = [backends[n]["p50_ms"] for n in names]
    p95 = [backends[n]["p95_ms"] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, p50, width=0.4, label="p50")
    ax.bar(x + 0.2, p95, width=0.4, label="p95")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("latency (ms, per call)")
    ax.set_title(f"Backend latency ({bench.get('modality', '')})")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "latency.png", dpi=150)
    plt.close(fig)


def plot_robustness(payload: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    base = payload["baseline"]["auc"]
    for ax, (group, key, xlabel) in zip(
        axes,
        (("jpeg", "quality", "JPEG quality"), ("resolution", "scale", "resolution scale")),
    ):
        items = payload.get(group, {})
        if not items:
            continue
        xs = [float(k) for k in items]
        aucs = [items[k]["auc"] for k in items]
        ax.plot(xs, aucs, "o-", label="AUC")
        ax.axhline(base, ls="--", lw=1, color="gray", label="baseline")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("AUC")
        ax.set_title(f"{group} degradation")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", default=None)
    parser.add_argument("--bench-json", default=None)
    parser.add_argument("--robustness-json", default=None)
    parser.add_argument("--out", default="runs/figures")
    args = parser.parse_args()

    logger = setup_logger()
    out = ensure_dir(args.out)

    if args.eval_json and Path(args.eval_json).exists():
        payload = json.loads(Path(args.eval_json).read_text())
        scores = load_scores(payload)
        metrics = payload.get("metrics") or payload.get("video_level") or {}
        if scores is not None:
            labels, values = scores
            plot_roc(labels, values, metrics, out)
            plot_score_dist(labels, values, out)
            logger.info("ROC 与分数分布已输出（%d 条样本）", len(labels))
        else:
            logger.warning("评测 JSON 缺少逐样本得分，跳过 ROC")

    if args.bench_json and Path(args.bench_json).exists():
        plot_latency(json.loads(Path(args.bench_json).read_text()), out)
        logger.info("延迟对比图已输出")

    if args.robustness_json and Path(args.robustness_json).exists():
        plot_robustness(json.loads(Path(args.robustness_json).read_text()), out)
        logger.info("鲁棒性曲线已输出")

    logger.info("图片目录: %s -> %s", out, [p.name for p in sorted(out.glob("*.png"))])


if __name__ == "__main__":
    main()
