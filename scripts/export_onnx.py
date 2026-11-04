"""导出 ONNX 并做 INT8 动态量化（开题报告 3.2 / 3.3 实时推理引擎）。

用法：
    # 视觉模型
    python scripts/export_onnx.py --kind visual --ckpt runs/exp1/best.pt --out runs/exp1/model.onnx
    # 音频模型
    python scripts/export_onnx.py --kind audio --ckpt runs/audio/best.pt --out runs/audio/model.onnx

默认同时生成 INT8 版本（*.int8.onnx）并校验与 PyTorch 的输出偏差。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo_deepfake.config import Config  # noqa: E402
from geo_deepfake.engine.onnx_engine import OnnxClassifier  # noqa: E402
from geo_deepfake.utils import save_json, setup_logger  # noqa: E402


def build_visual(ckpt: Path, image_size: int):
    import torch

    from geo_deepfake.models.visual import VisualDetector

    payload = torch.load(str(ckpt), map_location="cpu")
    cfg = Config.from_dict(payload["config"]) if payload.get("config") else Config()
    model = VisualDetector(
        backbone=cfg.model.backbone,
        pretrained=False,
        dropout=cfg.model.dropout,
        temporal=cfg.model.temporal,
    )
    model.load_state_dict(payload["model"])
    model.eval()
    dummy = torch.zeros(1, 3, image_size, image_size)
    return model, dummy, {"image_size": image_size, "backbone": cfg.model.backbone}


def build_audio(ckpt: Path, freq_bins: int = 64, frames: int = 398):
    import torch

    from geo_deepfake.models.audio import _build_model

    payload = torch.load(str(ckpt), map_location="cpu")
    meta = payload.get("audio", {})
    dim = int(meta.get("n_ceps" if meta.get("feature") == "lfcc" else "n_mels", freq_bins))
    model = _build_model(dim, channels=meta.get("channels", (16, 32, 64, 64)))
    model.load_state_dict(payload["model"])
    model.eval()
    dummy = torch.zeros(1, 1, dim, frames)
    return model, dummy, {**meta, "freq_bins": dim, "frames": frames}


def build_calibration_reader(input_name: str, manifest: str | None, count: int = 32):
    """静态量化的校准数据读取器；优先用真实帧，缺失时退化为随机张量。"""
    from onnxruntime.quantization import CalibrationDataReader

    arrays = []
    if manifest and Path(manifest).exists():
        import cv2

        from geo_deepfake.data.dataset import load_manifest_rows
        from geo_deepfake.data.preprocess import FrameTransform, read_image

        rows = load_manifest_rows(manifest)[:count]
        transform = FrameTransform(size=224, train=False)
        for row in rows:
            try:
                arrays.append(transform(read_image(row["path"]))[None].astype(np.float32))
            except Exception:
                continue
        del cv2
    if not arrays:
        rng = np.random.default_rng(0)
        arrays = [rng.normal(0, 1, (1, 3, 224, 224)).astype(np.float32) for _ in range(8)]

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self._iter = iter(arrays)

        def get_next(self):
            batch = next(self._iter, None)
            return None if batch is None else {input_name: batch}

    return _Reader()


def export_and_quantize(model, dummy, onnx_path: Path, input_name: str, dims: dict, int8: bool,
                        calib_manifest: str | None = None, static: bool = True) -> dict:
    import torch

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=[input_name],
        output_names=["logits"],
        dynamic_axes={input_name: dims, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # 旧版导出器对 dynamic_axes 支持更稳定
    )
    report = {"onnx": str(onnx_path), "size_MB": round(onnx_path.stat().st_size / 1e6, 2)}

    with torch.inference_mode():
        torch_logits = model(dummy).numpy()
    onnx_logits = OnnxClassifier(onnx_path).logits(dummy.numpy())
    report["fp32_max_abs_diff"] = float(np.abs(torch_logits - onnx_logits).max())

    if int8:
        from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                              quantize_dynamic, quantize_static)
        from onnxruntime.quantization.shape_inference import quant_pre_process

        int8_path = onnx_path.with_suffix(".int8.onnx")
        pre_path = onnx_path.with_suffix(".pre.onnx")
        # 量化前必须做预处理（shape inference + 图优化），否则量化后的图会明显变慢
        quant_pre_process(str(onnx_path), str(pre_path), skip_symbolic_shape=True)
        method = "dynamic"
        if static:
            reader = build_calibration_reader(input_name, calib_manifest)
            quantize_static(
                str(pre_path),
                str(int8_path),
                reader,
                quant_format=QuantFormat.QDQ,
                per_channel=True,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8,
                calibrate_method=CalibrationMethod.MinMax,
            )
            method = "static-QDQ"
        else:
            quantize_dynamic(str(pre_path), str(int8_path), weight_type=QuantType.QInt8)
        int8_logits = OnnxClassifier(int8_path).logits(dummy.numpy())
        report.update(
            {
                "int8": str(int8_path),
                "int8_method": method,
                "int8_size_MB": round(int8_path.stat().st_size / 1e6, 2),
                "int8_max_abs_diff": float(np.abs(torch_logits - int8_logits).max()),
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="visual", choices=["visual", "audio"])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--no-int8", action="store_true")
    parser.add_argument("--no-static", action="store_true", help="改用动态量化（默认静态 QDQ）")
    parser.add_argument("--calib-manifest", default="data/dummy/manifest.csv",
                        help="静态量化校准集（帧级清单），缺失时退化为随机张量")
    args = parser.parse_args()

    logger = setup_logger()
    if args.kind == "visual":
        ckpt = Path(args.ckpt or "runs/exp1/best.pt")
        image_size = args.image_size or Config.load("configs/default.yaml").data.image_size
        model, dummy, meta = build_visual(ckpt, image_size)
        out = Path(args.out or ckpt.parent / "model.onnx")
        report = export_and_quantize(model, dummy, out, "frames", {0: "batch"}, not args.no_int8,
                                     calib_manifest=args.calib_manifest, static=not args.no_static)
    else:
        ckpt = Path(args.ckpt or "runs/audio/best.pt")
        model, dummy, meta = build_audio(ckpt)
        out = Path(args.out or ckpt.parent / "audio.onnx")
        # 音频输入形状与图像校准集不匹配，使用动态量化（模型仅 0.24 MB）
        report = export_and_quantize(model, dummy, out, "feat", {0: "batch", 3: "frames"},
                                     not args.no_int8, static=False)

    report.update(meta)
    save_json(report, out.with_suffix(".json"))
    logger.info("导出完成: %s", report)
    if report["fp32_max_abs_diff"] > 1e-3:
        logger.warning("ONNX 与 PyTorch 输出偏差偏大: %.5f", report["fp32_max_abs_diff"])
    if "int8_max_abs_diff" in report and report["int8_max_abs_diff"] > 0.5:
        logger.warning("INT8 输出偏差较大: %.5f（通常 0.01~0.1 可接受）", report["int8_max_abs_diff"])


if __name__ == "__main__":
    main()
