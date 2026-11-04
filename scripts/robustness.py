"""鲁棒性实验：不同 JPEG 压缩质量与输入分辨率下的性能衰减（开题报告 3.5）。

用法：
    python scripts/robustness.py --ckpt runs/wild/best.pt --manifest data/wild/manifest.csv \
        --split val --max-videos 40 --out runs/wild/robustness.json
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo_deepfake.config import Config  # noqa: E402
from geo_deepfake.data.dataset import load_manifest_rows  # noqa: E402
from geo_deepfake.data.preprocess import read_image  # noqa: E402
from geo_deepfake.engine.metrics import compute_metrics  # noqa: E402
from geo_deepfake.engine.predictor import DeepfakePredictor  # noqa: E402
from geo_deepfake.utils import save_json, setup_logger  # noqa: E402


def compress_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else image


def change_resolution(image: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return image
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(8, int(w * scale)), max(8, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def group_by_video(rows):
    groups = OrderedDict()
    for row in sorted(rows, key=lambda r: (r["video_id"], r["path"])):
        groups.setdefault(row["video_id"], {"label": row["label"], "paths": []})
        groups[row["video_id"]]["paths"].append(row["path"])
    return groups


def evaluate_condition(predictor, groups, frames_per_video: int, transform) -> dict:
    labels, scores, latencies = [], [], []
    for info in groups.values():
        frames = [transform(read_image(p)) for p in info["paths"][:frames_per_video]]
        result = predictor.predict_frames(frames)
        labels.append(info["label"])
        scores.append(result["fake_prob"])
        latencies.append(result["latency_ms"])
    metrics = compute_metrics(labels, scores)
    metrics["latency_ms_mean"] = float(np.mean(latencies))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", default="runs/wild/best.pt")
    parser.add_argument("--manifest", default="data/wild/manifest.csv")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-videos", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logger = setup_logger()
    cfg = Config.load(args.config)
    predictor = DeepfakePredictor(cfg=cfg, ckpt=args.ckpt, device=args.device)
    predictor.warmup(1)

    rows = [r for r in load_manifest_rows(args.manifest) if r["split"] == args.split]
    groups = group_by_video(rows)
    if args.max_videos and len(groups) > args.max_videos:
        # 随机分层抽样，避免按 video_id 排序取前 N 导致只剩单一类别
        rng = np.random.default_rng(args.seed)
        by_label = {0: [], 1: []}
        for vid, info in groups.items():
            by_label[info["label"]].append((vid, info))
        picked = []
        for label, items in by_label.items():
            idx = rng.permutation(len(items))[: max(1, args.max_videos // 2)]
            picked.extend(items[i] for i in idx)
        groups = OrderedDict(sorted(picked, key=lambda kv: kv[0]))
    logger.info("评测 %d 个视频序列，ckpt=%s", len(groups), args.ckpt)

    results = {"baseline": {}, "jpeg": {}, "resolution": {}}
    identity = lambda img: img  # noqa: E731
    results["baseline"] = evaluate_condition(predictor, groups, args.frames_per_video, identity)

    for quality in (95, 75, 50, 30, 10):
        fn = lambda img, q=quality: compress_jpeg(img, q)  # noqa: E731
        metrics = evaluate_condition(predictor, groups, args.frames_per_video, fn)
        results["jpeg"][str(quality)] = metrics
        logger.info("JPEG q=%3d  AUC=%.4f  EER=%.4f  F1=%.4f", quality,
                    metrics["auc"], metrics["eer"], metrics["f1"])

    for scale in (1.0, 0.75, 0.5, 0.35, 0.25):
        fn = lambda img, s=scale: change_resolution(img, s)  # noqa: E731
        metrics = evaluate_condition(predictor, groups, args.frames_per_video, fn)
        results["resolution"][str(scale)] = metrics
        logger.info("分辨率 x%.2f AUC=%.4f  EER=%.4f  F1=%.4f", scale,
                    metrics["auc"], metrics["eer"], metrics["f1"])

    base_auc = results["baseline"]["auc"]
    for group in ("jpeg", "resolution"):
        key = "quality" if group == "jpeg" else "scale"
        for cond, metrics in results[group].items():
            metrics["auc_drop"] = float(base_auc - metrics["auc"])
            metrics[key] = cond

    out = args.out or str(Path(args.ckpt).parent / f"robustness_{args.split}.json")
    save_json({"ckpt": args.ckpt, "split": args.split, "videos": len(groups), **results}, out)
    logger.info("鲁棒性结果已保存: %s", out)


if __name__ == "__main__":
    main()
