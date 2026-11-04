"""音频检测评测：流式推理 → EER / AUC / F1 + 实时性（开题报告 3.5）。

用法：
    python scripts/evaluate_audio.py --ckpt runs/audio/best.pt --split test
    python scripts/evaluate_audio.py --ckpt runs/audio/best.pt --split val --limit 500
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo_deepfake.data.audio import read_audio  # noqa: E402
from geo_deepfake.data.audio_dataset import load_audio_manifest  # noqa: E402
from geo_deepfake.engine.metrics import compute_metrics, format_metrics  # noqa: E402
from geo_deepfake.models.audio import AudioDetector  # noqa: E402
from geo_deepfake.utils import save_json, setup_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/asvspoof/manifest.csv")
    parser.add_argument("--ckpt", default="runs/audio/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=0, help="按清单顺序取前 N 条")
    parser.add_argument("--sample", type=int, default=0, help="随机抽样 N 条（更接近无偏估计）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logger = setup_logger()
    rows = [r for r in load_audio_manifest(args.manifest) if r["split"] == args.split]
    if not rows:
        raise SystemExit(f"split={args.split} 无样本")
    if args.sample and args.sample < len(rows):
        import random

        rows = random.Random(args.seed).sample(rows, args.sample)
    elif args.limit:
        rows = rows[: args.limit]

    detector = AudioDetector(ckpt=args.ckpt, device=args.device)
    if not detector.available:
        raise SystemExit(f"权重加载失败: {args.ckpt}")

    # 预热：避免首次 CUDA 初始化把延迟均值拉高
    detector.predict_waveform(read_audio(rows[0]["path"], detector.sample_rate))

    labels, scores, latencies, rtf, audio_len = [], [], [], [], []
    per_sample = []
    for row in rows:
        wav = read_audio(row["path"], detector.sample_rate)
        result = detector.predict_waveform(wav)
        if result.fake_prob is None:
            continue
        seconds = max(
            (result.num_chunks - 1) * detector.hop_s + detector.window_s, 1e-3
        )
        labels.append(row["label"])
        scores.append(result.fake_prob)
        latencies.append(result.latency_ms)
        audio_len.append(seconds)
        rtf.append(result.latency_ms / (seconds * 1000.0))
        per_sample.append(
            {"utt_id": row["utt_id"], "label": row["label"], "score": result.fake_prob}
        )

    if not scores:
        raise SystemExit("没有可评估的样本")

    metrics = compute_metrics(labels, scores, threshold=detector.threshold)
    report = {
        "split": args.split,
        "ckpt": args.ckpt,
        "device": detector.device,
        "metrics": metrics,
        "per_sample": per_sample,
        "latency": {
            "utterances": len(latencies),
            "mean_ms": statistics.fmean(latencies),
            "p50_ms": statistics.median(latencies),
            "p95_ms": sorted(latencies)[int(len(latencies) * 0.95) - 1],
            "rtf_mean": statistics.fmean(rtf) if rtf else None,
            "audio_seconds_mean": statistics.fmean(audio_len) if audio_len else None,
        },
    }
    logger.info("音频 %s", format_metrics(metrics))
    logger.info("逐条流式推理：均值 %.1f ms，p95 %.1f ms，RTF %.3f",
                report["latency"]["mean_ms"], report["latency"]["p95_ms"],
                report["latency"]["rtf_mean"] or 0.0)

    out_path = args.out or f"runs/audio/eval_{args.split}.json"
    save_json(report, out_path)
    logger.info("结果已保存: %s", out_path)


if __name__ == "__main__":
    main()
