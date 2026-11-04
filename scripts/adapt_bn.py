"""测试时自适应（BN-only）：用目标域**无标签**帧更新 BatchNorm 统计量，缓解跨库域偏移。

做法（TENT/BN-adapt 的最简形式）：
1. 载入源域（如 FF++）训练的模型；
2. 用目标域（如 WildDeepfake）**训练划分**的帧做前向，只更新 BN 的 running_mean/var，不改权重；
3. 在目标域**验证划分**上评测，与未自适应结果对比；
4. 同时在源域验证集上评测，确认没有把源域能力破坏掉。

用法：
    python scripts/adapt_bn.py --ckpt runs/ffpp/best.pt \
        --adapt-manifest data/wild/manifest.csv --adapt-split train \
        --eval-manifest data/wild/manifest.csv --eval-split val --steps 60
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo_deepfake.config import Config  # noqa: E402
from geo_deepfake.data.dataset import load_manifest_rows  # noqa: E402
from geo_deepfake.data.preprocess import read_image  # noqa: E402
from geo_deepfake.engine.metrics import compute_metrics  # noqa: E402
from geo_deepfake.engine.predictor import DeepfakePredictor  # noqa: E402
from geo_deepfake.utils import save_json, setup_logger  # noqa: E402


def group_by_video(rows):
    groups = OrderedDict()
    for row in sorted(rows, key=lambda r: (r["video_id"], r["path"])):
        groups.setdefault(row["video_id"], {"label": row["label"], "paths": []})
        groups[row["video_id"]]["paths"].append(row["path"])
    return groups


def eval_video_level(predictor, groups, frames_per_video: int) -> dict:
    labels, scores = [], []
    for info in groups.values():
        frames = [read_image(p) for p in info["paths"][:frames_per_video]]
        result = predictor.predict_frames(frames)
        labels.append(info["label"])
        scores.append(result["fake_prob"])
    return compute_metrics(labels, scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/ffpp/best.pt")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--adapt-manifest", default="data/wild/manifest.csv")
    parser.add_argument("--adapt-split", default="train")
    parser.add_argument("--adapt-limit", type=int, default=2000, help="用于自适应的无标签帧数")
    parser.add_argument("--eval-manifest", default="data/wild/manifest.csv")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--source-manifest", default="data/ffpp/manifest.csv")
    parser.add_argument("--source-split", default="val")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0, help=">0 时启用 TENT 式熵最小化（只更新 BN 仿射参数）")
    parser.add_argument("--momentum", type=float, default=0.05, help="BN running stats 动量，越小越保守")
    parser.add_argument("--source-mix", type=float, default=0.0,
                        help="源域无标签帧混入比例（0~1），用于防止源域能力被破坏")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch

    logger = setup_logger()
    cfg = Config.load(args.config)
    predictor = DeepfakePredictor(cfg=cfg, ckpt=args.ckpt)
    device = predictor.device
    model = predictor.model

    target_rows = [r for r in load_manifest_rows(args.adapt_manifest) if r["split"] == args.adapt_split]
    if args.adapt_limit:
        target_rows = target_rows[: args.adapt_limit]
    logger.info("自适应无标签帧: %d（来自 %s/%s）", len(target_rows), args.adapt_manifest, args.adapt_split)

    # ---------------- 自适应前基线
    target_groups = group_by_video(
        [r for r in load_manifest_rows(args.eval_manifest) if r["split"] == args.eval_split]
    )
    before = eval_video_level(predictor, target_groups, cfg.data.frames_per_video)
    logger.info("自适应前 目标域 AUC=%.4f EER=%.4f", before["auc"], before["eer"])

    source_groups = group_by_video(
        [r for r in load_manifest_rows(args.source_manifest) if r["split"] == args.source_split]
    )
    source_before = eval_video_level(predictor, source_groups, cfg.data.frames_per_video)
    logger.info("自适应前 源域   AUC=%.4f EER=%.4f", source_before["auc"], source_before["eer"])

    # ---------------- BN 自适应（无标签）
    model.train()  # 切换到 train 模式，让 BN 更新 running stats
    bn_modules = [m for m in model.modules() if m.__class__.__name__.startswith("BatchNorm")]
    for module in bn_modules:
        module.momentum = args.momentum
    optimizer = None
    if args.lr > 0:  # TENT：只更新 BN 的仿射参数
        params = [p for m in bn_modules for p in m.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)

    source_rows = []
    if args.source_mix > 0:
        source_rows = [
            r for r in load_manifest_rows(args.source_manifest) if r["split"] == args.adapt_split
        ]
        logger.info("源域混合比例 %.1f，可用源域帧 %d", args.source_mix, len(source_rows))

    idx = 0
    for step in range(args.steps):
        batch_rows = target_rows[idx : idx + args.batch_size]
        if source_rows and args.source_mix > 0:
            n_src = int(len(batch_rows) * args.source_mix)
            start = (idx * 7) % max(len(source_rows) - n_src, 1)
            batch_rows = batch_rows + source_rows[start : start + n_src]
        if len(batch_rows) < 4:
            idx = 0
            batch_rows = target_rows[idx : idx + args.batch_size]
        if source_rows and args.source_mix > 0:
            n_src = int(len(batch_rows) * args.source_mix)
            start = (idx * 7) % max(len(source_rows) - n_src, 1)
            batch_rows = batch_rows + source_rows[start : start + n_src]
        idx += args.batch_size
        frames = [read_image(r["path"]) for r in batch_rows]
        tensor = torch.from_numpy(predictor.prepare_frames(frames)).to(device)
        if optimizer is not None:
            logits = model.forward_frames(tensor)
            prob = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
            loss = -(prob * torch.log(prob) + (1 - prob) * torch.log(1 - prob)).mean()  # 熵最小化
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model.forward_frames(tensor)
        if (step + 1) % 20 == 0:
            logger.info("  自适应 step %d/%d", step + 1, args.steps)

    model.eval()
    after = eval_video_level(predictor, target_groups, cfg.data.frames_per_video)
    source_after = eval_video_level(predictor, source_groups, cfg.data.frames_per_video)
    logger.info("自适应后 目标域 AUC=%.4f EER=%.4f（ΔAUC %+.4f）",
                after["auc"], after["eer"], after["auc"] - before["auc"])
    logger.info("自适应后 源域   AUC=%.4f EER=%.4f（ΔAUC %+.4f）",
                source_after["auc"], source_after["eer"], source_after["auc"] - source_before["auc"])

    report = {
        "ckpt": args.ckpt,
        "mode": "tent" if args.lr > 0 else "bn_only",
        "steps": args.steps,
        "target": {
            "manifest": args.eval_manifest,
            "split": args.eval_split,
            "before": before,
            "after": after,
            "delta_auc": after["auc"] - before["auc"],
        },
        "source": {
            "manifest": args.source_manifest,
            "split": args.source_split,
            "before": source_before,
            "after": source_after,
            "delta_auc": source_after["auc"] - source_before["auc"],
        },
    }
    out = args.out or str(Path(args.ckpt).parent / "bn_adapt.json")
    save_json(report, out)
    logger.info("结果已保存: %s", out)


if __name__ == "__main__":
    main()
