"""实时性与效率基准：PyTorch / ONNX / INT8 三种后端的延迟、吞吐、参数量、FLOPs、显存。

用法：
    # 视觉（端到端 + 单模型），并对比导出的 ONNX
    python scripts/benchmark.py --modality visual --ckpt runs/exp1/best.pt \
        --onnx runs/exp1/model.onnx --onnx-int8 runs/exp1/model.int8.onnx
    # 音频（4 s 窗）
    python scripts/benchmark.py --modality audio --ckpt runs/audio/best.pt \
        --onnx runs/audio/audio.onnx --onnx-int8 runs/audio/audio.int8.onnx
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo_deepfake.config import Config  # noqa: E402
from geo_deepfake.engine.onnx_engine import OnnxClassifier  # noqa: E402
from geo_deepfake.engine.predictor import DeepfakePredictor  # noqa: E402
from geo_deepfake.models.visual import count_params  # noqa: E402
from geo_deepfake.utils import save_json, setup_logger  # noqa: E402


def _sync(cuda: bool) -> None:
    if cuda:
        import torch

        torch.cuda.synchronize()


def timeit(fn, iters: int, warmup: int, cuda: bool) -> dict:
    for _ in range(warmup):
        fn()
    _sync(cuda)
    timings = []
    for _ in range(iters):
        _sync(cuda)
        start = time.perf_counter()
        fn()
        _sync(cuda)
        timings.append((time.perf_counter() - start) * 1000.0)
    timings.sort()
    return {
        "mean_ms": statistics.fmean(timings),
        "p50_ms": statistics.median(timings),
        "p95_ms": timings[max(int(len(timings) * 0.95) - 1, 0)],
        "min_ms": timings[0],
    }


def measure_flops(model, sample_shape) -> float | None:
    try:
        import torch
        from thop import profile
    except ImportError:
        return None
    dummy = torch.zeros(*sample_shape, device=next(model.parameters()).device)
    try:
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(macs) / 1e9
    except Exception:
        return None


def peak_gpu_mb() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)


def bench_visual(args, logger) -> dict:
    cfg = Config.load(args.config) if Path(args.config).exists() else Config()
    if args.ckpt:
        cfg.infer.ckpt = args.ckpt
    predictor = DeepfakePredictor(cfg=cfg, ckpt=cfg.infer.ckpt, device=args.device)
    predictor.warmup(args.warmup)

    rng = np.random.default_rng(0)
    frames = [
        rng.integers(0, 255, (cfg.data.image_size, cfg.data.image_size, 3), dtype=np.uint8)
        for _ in range(args.frames)
    ]

    # 端到端（含人脸裁剪、归一化、时序聚合）
    e2e = timeit(lambda: predictor.predict_frames(frames), args.iters, args.warmup, predictor.device.type == "cuda")
    e2e["per_frame_ms"] = {k: v / args.frames for k, v in e2e.items() if k.endswith("_ms")}
    e2e["fps"] = 1000.0 / e2e["per_frame_ms"]["p50_ms"]

    # 模型前向（三后端公平对比）
    import torch

    tensor = torch.from_numpy(predictor.prepare_frames(frames)).to(predictor.device)
    model = predictor.model
    with torch.inference_mode():
        torch_fn = lambda: model.forward_frames(tensor)  # noqa: E731
        torch_stats = timeit(torch_fn, args.iters, args.warmup, predictor.device.type == "cuda")

    backends = {
        "torch_fp32": {**torch_stats, "device": predictor.device_name,
                       "params_M": count_params(model) / 1e6,
                       "flops_G": measure_flops(model, (1, 3, cfg.data.image_size, cfg.data.image_size)),
                       "peak_gpu_MB": peak_gpu_mb()},
    }
    cpu_tensor = tensor.detach().cpu().numpy()
    for name, path in (("onnx_fp32", args.onnx), ("onnx_int8", args.onnx_int8)):
        if not path or not Path(path).exists():
            continue
        session = OnnxClassifier(path)
        stats = timeit(lambda s=session: s.logits(cpu_tensor), args.iters, args.warmup, False)
        backends[name] = {**stats, "device": ",".join(session.providers),
                          "size_MB": round(Path(path).stat().st_size / 1e6, 2)}

    for name, stats in backends.items():
        logger.info("%-10s p50=%.2f ms  p95=%.2f ms  (%.1f FPS, %d 帧/次)",
                    name, stats["p50_ms"], stats["p95_ms"], 1000.0 / stats["p50_ms"], args.frames)

    return {
        "modality": "visual",
        "backbone": cfg.model.backbone,
        "frames_per_clip": args.frames,
        "iters": args.iters,
        "end_to_end": e2e,
        "backends": backends,
    }


def bench_audio(args, logger) -> dict:
    from geo_deepfake.data.audio import read_audio
    from geo_deepfake.models.audio import AudioDetector

    detector = AudioDetector(ckpt=args.ckpt or "runs/audio/best.pt", device=args.device)
    if not detector.available:
        raise SystemExit("音频权重未加载，先运行 train_audio.py")

    wav_file = args.audio_file
    if wav_file and Path(wav_file).exists():
        wav = read_audio(wav_file, detector.sample_rate)
    else:
        wav = np.random.default_rng(0).normal(0, 0.05, int(detector.window_s * detector.sample_rate)).astype(np.float32)
    feat = detector.feature(wav)
    import torch

    tensor = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).to(detector.device)
    is_cuda = str(detector.device).startswith("cuda")
    with torch.inference_mode():
        torch_stats = timeit(lambda: detector.model(tensor), args.iters, args.warmup, is_cuda)

    backends = {"torch_fp32": {**torch_stats, "device": detector.device}}
    cpu_tensor = tensor.detach().cpu().numpy()
    for name, path in (("onnx_fp32", args.onnx), ("onnx_int8", args.onnx_int8)):
        if not path or not Path(path).exists():
            continue
        session = OnnxClassifier(path)
        stats = timeit(lambda s=session: s.logits(cpu_tensor), args.iters, args.warmup, False)
        backends[name] = {**stats, "device": ",".join(session.providers),
                          "size_MB": round(Path(path).stat().st_size / 1e6, 2)}

    for name, stats in backends.items():
        logger.info("%-10s p50=%.2f ms  p95=%.2f ms  RTF=%.4f",
                    name, stats["p50_ms"], stats["p95_ms"], stats["p50_ms"] / (detector.window_s * 1000))

    return {
        "modality": "audio",
        "window_s": detector.window_s,
        "feature": detector.feature_type,
        "feature_dim": int(detector.feature.dim),
        "iters": args.iters,
        "backends": backends,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--modality", default="visual", choices=["visual", "audio"])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--onnx-int8", default=None)
    parser.add_argument("--audio-file", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logger = setup_logger()
    report = bench_visual(args, logger) if args.modality == "visual" else bench_audio(args, logger)
    if args.out:
        save_json(report, args.out)
        logger.info("基准结果已保存: %s", args.out)


if __name__ == "__main__":
    main()
