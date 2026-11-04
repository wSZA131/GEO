"""消融实验（开题报告 3.5）：时序聚合方式 / 融合权重 / GEO 信号有效性。

用法：
    python scripts/ablation.py --exp all --manifest data/dummy/manifest.csv --out runs/ablation
    python scripts/ablation.py --exp geo      # 只跑确定性部分（无需训练）

三类实验：
1. temporal：mean vs attn 帧得分聚合（各训练一个小模型后比较验证集 AUC/EER）
2. fusion  ：视觉/音频权重组合对综合概率的影响，含音频缺失时的自动归一化
3. geo     ：GEO 可信信号（来源凭证 / 内容重复度）对 trust_score 的影响
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo_deepfake.models.fusion import FusionScorer, geo_trust_score  # noqa: E402
from geo_deepfake.utils import ensure_dir, save_json, setup_logger  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run_temporal(args, out_dir: Path, logger) -> dict:
    results = {}
    for mode in ("mean", "attn"):
        ckpt_dir = out_dir / f"temporal_{mode}"
        cmd = [
            sys.executable, str(ROOT / "scripts" / "train.py"),
            "--manifest", args.manifest,
            "--out", str(ckpt_dir),
            "--epochs", str(args.epochs),
            "--temporal", mode,
        ]
        logger.info("训练 temporal=%s ...", mode)
        subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
        history = json.loads((ckpt_dir / "train_log.json").read_text())["history"]
        best = min(
            (h["val"] for h in history if h["val"].get("eer") == h["val"].get("eer")),
            key=lambda v: v["eer"],
        )
        results[mode] = {"auc": best["auc"], "eer": best["eer"], "f1": best["f1"]}
        logger.info("temporal=%-4s AUC=%.4f EER=%.4f", mode, best["auc"], best["eer"])
    if "mean" in results and "attn" in results:
        results["delta_eer_attn_minus_mean"] = results["attn"]["eer"] - results["mean"]["eer"]
    return results


def run_fusion(args, logger) -> dict:
    threshold = 0.5
    cases = [
        ("visual_only", 0.8, None),
        ("audio_only", None, 0.9),
        ("both_agree_fake", 0.9, 0.85),
        ("both_conflict", 0.9, 0.1),
    ]
    weight_sets = {
        "0.8/0.2": {"visual": 0.8, "audio": 0.2},
        "0.5/0.5": {"visual": 0.5, "audio": 0.5},
        "0.2/0.8": {"visual": 0.2, "audio": 0.8},
    }
    table = {}
    for wname, weights in weight_sets.items():
        scorer = FusionScorer(weights=weights, threshold=threshold)
        table[wname] = {}
        for cname, visual, audio in cases:
            result = scorer.score(visual_prob=visual, audio_prob=audio)
            table[wname][cname] = {
                "fake_prob": round(result.fake_prob, 4),
                "verdict": result.verdict,
                "notes": result.notes,
            }
    logger.info("fusion 权重消融完成：%d 组权重 × %d 个场景", len(weight_sets), len(cases))
    return {"threshold": threshold, "weights": table}


def run_geo(args, logger) -> dict:
    grid = []
    for provenance, duplication in product((0.0, 0.5, 1.0), (0.0, 0.5, 1.0)):
        for fake_prob in (0.1, 0.5, 0.9):
            score = geo_trust_score(
                fake_prob, meta={"provenance": provenance, "duplication": duplication}
            )
            grid.append(
                {
                    "fake_prob": fake_prob,
                    "provenance": provenance,
                    "duplication": duplication,
                    "trust_score": round(score["trust_score"], 4),
                    "geo_penalty": round(score["geo_penalty"], 4),
                }
            )
    # 关键对照：同一伪造概率下，有/无来源凭证的可信分差
    contrast = []
    for fake_prob in (0.1, 0.3, 0.5, 0.7, 0.9):
        good = geo_trust_score(fake_prob, meta={"provenance": 1.0, "duplication": 0.0})
        bad = geo_trust_score(fake_prob, meta={"provenance": 0.0, "duplication": 1.0})
        contrast.append(
            {
                "fake_prob": fake_prob,
                "trust_good_provenance": round(good["trust_score"], 4),
                "trust_bad_provenance": round(bad["trust_score"], 4),
                "drop": round(good["trust_score"] - bad["trust_score"], 4),
            }
        )
    logger.info("GEO 信心: 最差来源使 trust 最大下降 %.3f", max(c["drop"] for c in contrast))
    return {"grid": grid, "contrast": contrast}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all", choices=["all", "temporal", "fusion", "geo"])
    parser.add_argument("--manifest", default="data/dummy/manifest.csv")
    parser.add_argument("--out", default="runs/ablation")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    logger = setup_logger()
    out_dir = ensure_dir(args.out)
    report = {}

    if args.exp in ("all", "temporal"):
        if not Path(args.manifest).exists():
            logger.warning("清单不存在，跳过 temporal 消融: %s", args.manifest)
        else:
            report["temporal"] = run_temporal(args, out_dir, logger)
    if args.exp in ("all", "fusion"):
        report["fusion"] = run_fusion(args, logger)
    if args.exp in ("all", "geo"):
        report["geo"] = run_geo(args, logger)

    path = save_json(report, out_dir / f"ablation_{args.exp}.json")
    logger.info("消融结果已保存: %s", path)


if __name__ == "__main__":
    main()
