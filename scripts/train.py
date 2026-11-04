"""视觉检测模型训练脚本（帧级分类）。

用法：
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --manifest data/dummy/manifest.csv --epochs 3 --backbone mobilenetv3_small_100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo_deepfake.config import Config  # noqa: E402
from geo_deepfake.data.dataset import FrameManifestDataset, load_manifest_rows  # noqa: E402
from geo_deepfake.data.preprocess import FrameTransform, build_face_cropper  # noqa: E402
from geo_deepfake.engine.metrics import compute_metrics, format_metrics  # noqa: E402
from geo_deepfake.models.factory import build_visual_detector  # noqa: E402
from geo_deepfake.models.visual import count_params  # noqa: E402
from geo_deepfake.utils import ensure_dir, resolve_device, save_json, set_seed, setup_logger  # noqa: E402


class SoftFrameDataset(FrameManifestDataset):
    """附带教师软标签的帧数据集（蒸馏用，需在模块级定义以便 DataLoader 多进程 pickle）。"""

    def __init__(self, rows, transform, seed: int = 42, soft_map=None):
        super().__init__(rows, transform, seed=seed)
        self.soft_map = soft_map or {}

    def __getitem__(self, index: int):
        image, label, video_id = super().__getitem__(index)
        soft = self.soft_map.get(self.rows[index]["path"], label)
        return image, label, video_id, torch.tensor(float(soft), dtype=torch.float32)


def build_splits(rows, split_name: str):
    return [r for r in rows if r["split"] == split_name]


@torch.inference_mode()
def evaluate_frames(model, loader, device) -> dict:
    model.eval()
    scores, labels = [], []
    for images, targets, _ in loader:
        logits = model.forward_frames(images.to(device))
        scores.extend(torch.sigmoid(logits).float().cpu().tolist())
        labels.extend(targets.tolist())
    if not scores:
        return {"auc": float("nan"), "num_samples": 0}
    return compute_metrics(labels, scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--temporal", default=None, choices=["mean", "attn"],
                        help="帧得分聚合方式（消融用）")
    parser.add_argument("--arch", default=None, choices=["cnn", "patch"],
                        help="视觉架构：cnn=轻量骨干，patch=LaDeDa 式 patch 检测器")
    parser.add_argument("--patch-pool", default=None, choices=["mean", "topk"],
                        help="patch 分数聚合方式")
    parser.add_argument("--aug", action="store_true",
                        help="启用跨域增强（高频扰动 + 二次转码 + 色度噪声）")
    parser.add_argument("--balanced", action="store_true", help="类别平衡采样（缓解 real:fake 失衡）")
    parser.add_argument("--soft-labels", default=None, help="教师模型软标签 csv（path,prob），用于蒸馏")
    parser.add_argument("--soft-alpha", type=float, default=1.0,
                        help="蒸馏损失中软标签的权重：1.0=纯软标签，0.5=软硬各半")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pretrained", action="store_true", help="使用 timm 预训练权重（需联网）")
    args = parser.parse_args()

    cfg = Config.load(args.config) if Path(args.config).exists() else Config()
    if args.manifest:
        cfg.data.manifest = args.manifest
    if args.backbone:
        cfg.model.backbone = args.backbone
    if args.temporal:
        cfg.model.temporal = args.temporal
    if args.arch:
        cfg.model.arch = args.arch
    if args.patch_pool:
        cfg.model.patch_pool = args.patch_pool
    if args.aug:
        cfg.data.hf_prob = 0.5
        cfg.data.recompress_prob = 0.5
        cfg.data.chroma_noise_prob = 0.5
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.lr:
        cfg.train.lr = args.lr
    if args.out:
        cfg.train.out_dir = args.out
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.pretrained:
        cfg.model.pretrained = True

    logger = setup_logger()
    set_seed(cfg.train.seed)
    device = torch.device(resolve_device(cfg.infer.device))
    out_dir = ensure_dir(cfg.train.out_dir)

    rows = load_manifest_rows(cfg.data.manifest)
    train_rows, val_rows = build_splits(rows, "train"), build_splits(rows, "val")
    if not train_rows:
        raise SystemExit("训练集为空，请检查 manifest 的 split 字段")
    if not val_rows:
        logger.warning("验证集为空，使用训练集前 20% 作为验证集")
        val_rows = train_rows[: max(1, len(train_rows) // 5)]

    cropper = (
        build_face_cropper("haar", margin=cfg.data.face_margin) if cfg.data.use_face_crop else None
    )
    train_tf = FrameTransform(
        size=cfg.data.image_size,
        train=True,
        jpeg_quality=cfg.data.jpeg_quality,
        blur_prob=cfg.data.blur_prob,
        brightness_delta=cfg.data.brightness_delta,
        flip_prob=cfg.data.flip_prob,
        face_cropper=cropper,
        hf_prob=cfg.data.hf_prob,
        hf_alpha=tuple(cfg.data.hf_alpha),
        recompress_prob=cfg.data.recompress_prob,
        chroma_noise_prob=cfg.data.chroma_noise_prob,
    )
    val_tf = FrameTransform(size=cfg.data.image_size, train=False, face_cropper=cropper)

    soft_map = {}
    if args.soft_labels:
        with open(args.soft_labels, "r", encoding="utf-8") as fh:
            next(fh, None)
            for line in fh:
                path, prob = line.rsplit(",", 1)
                soft_map[path.strip()] = float(prob)

    train_ds = FrameManifestDataset(train_rows, train_tf, seed=cfg.train.seed)
    if soft_map:
        train_ds = SoftFrameDataset(train_rows, train_tf, seed=cfg.train.seed, soft_map=soft_map)
        logger.info("启用蒸馏软标签：%d 条", len(soft_map))
    val_ds = FrameManifestDataset(val_rows, val_tf, seed=cfg.train.seed + 1)
    sampler = None
    if args.balanced:
        # 类别平衡采样：按类别频率取倒数作为权重，缓解 real:fake 失衡
        counts = np.bincount([r["label"] for r in train_rows], minlength=2).astype(np.float64)
        class_w = 1.0 / np.maximum(counts, 1)
        sample_w = torch.as_tensor([class_w[r["label"]] for r in train_rows], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        logger.info("启用类别平衡采样，类别计数 %s", counts.astype(int).tolist())
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        drop_last=len(train_ds) > cfg.train.batch_size,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers)

    model = build_visual_detector(cfg).to(device)
    logger.info(
        "arch=%s backbone=%s params=%.2fM device=%s train=%d val=%d",
        cfg.model.arch, cfg.model.backbone, count_params(model) / 1e6, device, len(train_ds), len(val_ds),
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    use_amp = bool(cfg.train.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_auc = [], -1.0
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        start = time.perf_counter()
        running, count = 0.0, 0
        for batch in train_loader:
            images, targets = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            soft = batch[3].to(device, non_blocking=True) if len(batch) == 4 else targets
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model.forward_frames(images)
                if len(batch) == 4 and args.soft_alpha < 1.0:
                    loss = args.soft_alpha * criterion(logits, soft) + (1 - args.soft_alpha) * criterion(logits, targets)
                elif len(batch) == 4:
                    loss = criterion(logits, soft)
                else:
                    loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * targets.size(0)
            count += targets.size(0)

        val_metrics = evaluate_frames(model, val_loader, device)
        train_loss = running / max(count, 1)
        elapsed = time.perf_counter() - start
        logger.info(
            "epoch %d loss=%.4f %s (%.1fs)",
            epoch, train_loss, format_metrics(val_metrics), elapsed,
        )
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val": {k: (None if isinstance(v, float) and np.isnan(v) else v)
                                for k, v in val_metrics.items()},
                        "seconds": elapsed})

        auc = val_metrics.get("auc", float("-inf"))
        auc = -1.0 if auc != auc else auc  # NaN -> -1
        if auc > best_auc:
            best_auc = auc
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg.to_dict(),
                    "epoch": epoch,
                    "val": val_metrics,
                    "backbone": cfg.model.backbone,
                },
                out_dir / "best.pt",
            )
            logger.info("保存最优权重 -> %s (val auc=%.4f)", out_dir / "best.pt", auc)

    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, out_dir / "last.pt")
    save_json({"history": history, "best_auc": best_auc, "config": cfg.to_dict()}, out_dir / "train_log.json")
    logger.info("训练结束，最优 val AUC=%.4f，输出目录: %s", best_auc, out_dir)


if __name__ == "__main__":
    main()
