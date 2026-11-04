"""音频检测模型训练（ASVspoof 2019 LA/PA）。

用法：
    python scripts/train_audio.py --manifest data/asvspoof/manifest.csv --epochs 5
    python scripts/train_audio.py --manifest data/asvspoof/manifest.csv --max-batches 200  # 快速试跑
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo_deepfake.data.audio_dataset import AudioManifestDataset, load_audio_manifest  # noqa: E402
from geo_deepfake.engine.metrics import compute_metrics, format_metrics  # noqa: E402
from geo_deepfake.models.audio import _build_model  # noqa: E402
from geo_deepfake.utils import ensure_dir, resolve_device, save_json, set_seed, setup_logger  # noqa: E402


@torch.inference_mode()
def evaluate(model, loader, device, max_batches: int = 0):
    model.eval()
    scores, labels = [], []
    for i, (feats, targets, _) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        logits = model(feats.to(device))
        scores.extend(torch.sigmoid(logits).float().cpu().tolist())
        labels.extend(targets.tolist())
    if not scores:
        return {"auc": float("nan"), "num_samples": 0}
    return compute_metrics(labels, scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/asvspoof/manifest.csv")
    parser.add_argument("--out", default="runs/audio")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--duration", type=float, default=1.5,
        help="训练裁剪时长（秒）。应与 --window-s 一致，否则训练/流式推理分布不匹配",
    )
    parser.add_argument("--feature", default="lfcc", choices=["lfcc", "logmel"],
                        help="前端特征：lfcc（ASVspoof 常用）或 logmel")
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--n-ceps", type=int, default=20)
    parser.add_argument("--n-filters", type=int, default=40)
    parser.add_argument("--pos-weight", default="auto",
                        help="BCE 正类权重：auto 按训练集频率自动计算，none 关闭，或给具体数值")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0, help="每 epoch 最多迭代多少 batch（调试）")
    parser.add_argument("--max-val-batches", type=int, default=0, help="验证集最多评估多少 batch（调试）")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-s", type=float, default=1.5)
    parser.add_argument("--hop-s", type=float, default=0.5)
    args = parser.parse_args()

    logger = setup_logger()
    set_seed(args.seed)
    device = torch.device(resolve_device(args.device))
    out_dir = ensure_dir(args.out)

    rows = load_audio_manifest(args.manifest)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"] or [r for r in rows if r["split"] == "test"]
    if not train_rows:
        raise SystemExit("训练集为空，请检查清单 split 字段")
    if not val_rows:
        raise SystemExit("缺少 val/test 划分，无法评估")

    feat_kwargs = dict(feature=args.feature, n_mels=args.n_mels,
                       n_ceps=args.n_ceps, n_filters=args.n_filters)
    train_ds = AudioManifestDataset(train_rows, duration_s=args.duration, train=True,
                                    seed=args.seed, **feat_kwargs)
    val_ds = AudioManifestDataset(val_rows, duration_s=args.duration, train=False,
                                  seed=args.seed + 1, **feat_kwargs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=len(train_ds) > args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = _build_model(train_ds.feature.dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # 类别不平衡：pos_weight = 负样本数 / 正样本数（ASVspoof 训练集约 1:8.8）
    n_pos = sum(1 for r in train_rows if r["label"] == 1)
    n_neg = len(train_rows) - n_pos
    if str(args.pos_weight).lower() == "auto":
        pos_weight = torch.tensor([max(n_neg, 1) / max(n_pos, 1)], device=device)
    elif str(args.pos_weight).lower() == "none":
        pos_weight = None
    else:
        pos_weight = torch.tensor([float(args.pos_weight)], device=device)

    logger.info(
        "train=%d val=%d params=%.1fk feature=%s(dim=%d) pos_weight=%s device=%s",
        len(train_ds), len(val_ds), n_params / 1e3, args.feature, train_ds.feature.dim,
        "none" if pos_weight is None else round(float(pos_weight.item()), 3), device,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_auc = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        running, count = 0.0, 0
        for i, (feats, targets, _) in enumerate(train_loader):
            if args.max_batches and i >= args.max_batches:
                break
            feats, targets = feats.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(feats), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * targets.size(0)
            count += targets.size(0)

        metrics = evaluate(model, val_loader, device, args.max_val_batches)
        elapsed = time.perf_counter() - start
        logger.info("epoch %d loss=%.4f %s (%.1fs)", epoch, running / max(count, 1),
                    format_metrics(metrics), elapsed)
        history.append({"epoch": epoch, "train_loss": running / max(count, 1), "val": metrics})

        auc = metrics.get("auc", float("-inf"))
        auc = -1.0 if auc != auc else auc
        if auc > best_auc:
            best_auc = auc
            torch.save(
                {
                    "model": model.state_dict(),
                    "audio": {
                        "feature": args.feature,
                        "n_mels": args.n_mels,
                        "n_ceps": args.n_ceps,
                        "n_filters": args.n_filters,
                        "sample_rate": 16000,
                        "window_s": args.window_s,
                        "hop_s": args.hop_s,
                        "duration_s": args.duration,
                        "channels": (16, 32, 64, 64),
                        "threshold": 0.5,
                    },
                    "val": metrics,
                    "epoch": epoch,
                },
                out_dir / "best.pt",
            )
            logger.info("保存最优权重 -> %s (val auc=%.4f)", out_dir / "best.pt", auc)

    torch.save({"model": model.state_dict()}, out_dir / "last.pt")
    save_json({"history": history, "best_auc": best_auc}, out_dir / "train_log.json")
    logger.info("训练结束，最优 val AUC=%.4f", best_auc)


if __name__ == "__main__":
    main()
