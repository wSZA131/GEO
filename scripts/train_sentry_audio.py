"""训练 SENTRY 风格音频检测器（整数小波特征 + GBDT + OOD）。

用法：
    python scripts/train_sentry_audio.py --manifest data/asvspoof_bal/manifest.csv \
        --out runs/audio_sentry --eval-manifest data/asvspoof/manifest.csv --sample 10000
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo_deepfake.data.audio import iter_chunks, read_audio  # noqa: E402
from geo_deepfake.data.audio_dataset import load_audio_manifest  # noqa: E402
from geo_deepfake.data.wavelet_features import SentryFeatures, feature_dim  # noqa: E402
from geo_deepfake.models.sentry_audio import SentryAudioDetector  # noqa: E402
from geo_deepfake.engine.metrics import compute_metrics, format_metrics  # noqa: E402
from geo_deepfake.utils import ensure_dir, save_json, setup_logger  # noqa: E402

FEATS = SentryFeatures()


def _featurize(job):
    path, label, chunks_per_utt = job
    wav = read_audio(path, 16000)
    chunks = list(iter_chunks(wav, 16000, 4.0, 2.0))[:chunks_per_utt]
    if not chunks:
        return None
    X = np.stack([FEATS(c) for c in chunks])
    return X, np.full(X.shape[0], label, dtype=np.int64)


def build_matrix(rows, chunks_per_utt: int, workers: int, logger, tag: str):
    jobs = [(r["path"], r["label"], chunks_per_utt) for r in rows]
    Xs, ys = [], []
    start = time.perf_counter()
    if workers > 1:
        with mp.Pool(workers) as pool:
            for i, out in enumerate(pool.imap(_featurize, jobs, chunksize=16), 1):
                if out is None:
                    continue
                Xs.append(out[0])
                ys.append(out[1])
                if i % 1000 == 0:
                    logger.info("  %s 特征 %d/%d", tag, i, len(jobs))
    else:
        for i, job in enumerate(jobs, 1):
            out = _featurize(job)
            if out is None:
                continue
            Xs.append(out[0])
            ys.append(out[1])
            if i % 1000 == 0:
                logger.info("  %s 特征 %d/%d", tag, i, len(jobs))
    X = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, feature_dim()))
    y = np.concatenate(ys, axis=0) if ys else np.zeros(0, dtype=np.int64)
    logger.info("%s: %d 个 chunk 样本，维度 %d，用时 %.1fs", tag, len(y), X.shape[1],
                time.perf_counter() - start)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/asvspoof_bal/manifest.csv")
    parser.add_argument("--eval-manifest", default=None)
    parser.add_argument("--out", default="runs/audio_sentry")
    parser.add_argument("--chunks-per-utt", type=int, default=2)
    parser.add_argument("--eval-sample", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    logger = setup_logger()
    out_dir = ensure_dir(args.out)
    rows = load_audio_manifest(args.manifest)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"] or [r for r in rows if r["split"] == "test"]
    logger.info("train=%d val=%d", len(train_rows), len(val_rows))

    X_tr, y_tr = build_matrix(train_rows, args.chunks_per_utt, args.workers, logger, "train")
    X_va, y_va = build_matrix(val_rows, 1, args.workers, logger, "val")

    detector = SentryAudioDetector().fit(X_tr, y_tr, max_iter=args.max_iter, learning_rate=args.lr)
    val_prob = detector.predict_features(X_va)
    val_metrics = compute_metrics(y_va, val_prob)
    logger.info("val %s", format_metrics(val_metrics))

    model_path = detector.save(out_dir / "sentry.joblib")
    logger.info("模型已保存: %s", model_path)

    report = {"val": val_metrics, "train_chunks": int(len(y_tr)), "feature_dim": int(X_tr.shape[1])}

    if args.eval_manifest:
        eval_rows = [r for r in load_audio_manifest(args.eval_manifest) if r["split"] == "test"]
        if args.eval_sample and args.eval_sample < len(eval_rows):
            import random

            eval_rows = random.Random(42).sample(eval_rows, args.eval_sample)
        labels, scores, latencies = [], [], []
        for i, row in enumerate(eval_rows, 1):
            result = detector.predict_file(row["path"])
            if result.fake_prob is None:
                continue
            labels.append(row["label"])
            scores.append(result.fake_prob)
            latencies.append(result.latency_ms)
            if i % 2000 == 0:
                logger.info("  评测 %d/%d", i, len(eval_rows))
        metrics = compute_metrics(labels, scores, threshold=detector.threshold)
        logger.info("eval %s", format_metrics(metrics))
        logger.info("推理延迟 均值 %.2f ms / p95 %.2f ms",
                    statistics.fmean(latencies), sorted(latencies)[int(len(latencies) * 0.95) - 1])
        report["eval"] = metrics
        report["latency_ms"] = {
            "mean": statistics.fmean(latencies),
            "p95": sorted(latencies)[int(len(latencies) * 0.95) - 1],
        }
        report["per_sample"] = [
            {"label": int(l), "score": float(s)} for l, s in zip(labels, scores)
        ]

    save_json(report, out_dir / "result.json")
    logger.info("结果已保存: %s", out_dir / "result.json")


if __name__ == "__main__":
    main()
