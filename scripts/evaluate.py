"""视频级评测：按 video_id 聚合帧得分，输出 AUC / EER / F1 等指标（开题报告 3.5）。

用法：
    python scripts/evaluate.py --config configs/default.yaml --split val
    python scripts/evaluate.py --ckpt runs/exp1/best.pt --manifest data/dummy/manifest.csv --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo_deepfake.config import Config  # noqa: E402
from geo_deepfake.data.dataset import load_manifest_rows  # noqa: E402
from geo_deepfake.engine.metrics import compute_metrics, format_metrics  # noqa: E402
from geo_deepfake.engine.predictor import DeepfakePredictor  # noqa: E402
from geo_deepfake.utils import save_json, setup_logger  # noqa: E402


def group_by_video(rows):
    groups = OrderedDict()
    for row in sorted(rows, key=lambda r: (r["video_id"], r["path"])):
        groups.setdefault(row["video_id"], {"label": row["label"], "paths": []})
        groups[row["video_id"]]["paths"].append(row["path"])
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logger = setup_logger()
    cfg = Config.load(args.config) if Path(args.config).exists() else Config()
    if args.manifest:
        cfg.data.manifest = args.manifest
    if args.ckpt:
        cfg.infer.ckpt = args.ckpt

    rows = [r for r in load_manifest_rows(cfg.data.manifest) if r["split"] == args.split]
    if not rows:
        raise SystemExit(f"split={args.split} 的样本为空")

    predictor = DeepfakePredictor(cfg=cfg, ckpt=cfg.infer.ckpt, device=args.device)
    predictor.warmup(1)

    per_video, video_labels, video_scores = [], [], []
    frame_labels, frame_scores = [], []
    groups = group_by_video(rows)
    if args.max_videos:
        groups = OrderedDict(list(groups.items())[: args.max_videos])

    for vid, info in groups.items():
        paths = info["paths"][: cfg.data.frames_per_video]
        result = predictor.predict_frame_paths(paths)
        per_video.append(
            {
                "video_id": vid,
                "label": info["label"],
                "fake_prob": result["fake_prob"],
                "verdict": result["verdict"],
                "trust_score": result["trust_score"],
                "num_frames": result["num_frames"],
                "latency_ms": result["latency_ms"],
            }
        )
        video_labels.append(info["label"])
        video_scores.append(result["fake_prob"])
        frame_labels.extend([info["label"]] * len(result["frame_probs"]))
        frame_scores.extend(result["frame_probs"])

    video_metrics = compute_metrics(video_labels, video_scores)
    frame_metrics = compute_metrics(frame_labels, frame_scores)
    latencies = [v["latency_ms"] for v in per_video]
    latency_stats = {
        "videos": len(latencies),
        "mean_ms": sum(latencies) / len(latencies),
        "max_ms": max(latencies),
        "per_frame_mean_ms": sum(latencies) / max(sum(v["num_frames"] for v in per_video), 1),
    }

    logger.info("视频级 %s", format_metrics(video_metrics))
    logger.info("帧级   %s", format_metrics(frame_metrics))
    logger.info(
        "延迟   %d 个视频，均值 %.1f ms，单帧均值 %.1f ms",
        latency_stats["videos"], latency_stats["mean_ms"], latency_stats["per_frame_mean_ms"],
    )

    payload = {
        "split": args.split,
        "ckpt": str(cfg.infer.ckpt),
        "device": predictor.device_name,
        "video_level": video_metrics,
        "frame_level": frame_metrics,
        "latency": latency_stats,
        "per_video": per_video,
    }
    out_path = args.out or (Path(cfg.train.out_dir) / f"eval_{args.split}.json")
    save_json(payload, out_path)
    logger.info("评测结果已保存: %s", out_path)
    print(json.dumps({k: payload[k] for k in ("video_level", "frame_level", "latency")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
