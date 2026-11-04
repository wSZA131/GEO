"""评估与接口层：REST API（对应开题报告 3.3 第 6 行）。

启动：
    python -m geo_deepfake.api.server --config configs/default.yaml --ckpt runs/exp1/best.pt

接口：
    GET  /healthz          健康检查（设备、骨干、音频模块状态）
    POST /predict/image    上传图片 → 伪造概率 / 判决 / 可信评分
    POST /predict/video    上传视频 → 抽帧聚合后的结果
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from ..config import Config
from ..engine.predictor import DeepfakePredictor
from ..utils import setup_logger

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB


def _meta_from_query(provenance: Optional[float], duplication: Optional[float]) -> Optional[Dict[str, float]]:
    if provenance is None and duplication is None:
        return None
    return {"provenance": float(provenance or 0.0), "duplication": float(duplication or 0.0)}


def create_app(config_path: Optional[str] = None, ckpt: Optional[str] = None) -> FastAPI:
    logger = setup_logger()
    app = FastAPI(title="GEO 营销场景实时深度伪造检测 API", version="0.1.0")
    state: Dict[str, Any] = {"predictor": None}

    cfg_path = config_path or os.environ.get("GEO_DF_CONFIG") or "configs/default.yaml"
    ckpt_path = ckpt or os.environ.get("GEO_DF_CKPT")

    def get_predictor() -> DeepfakePredictor:
        if state["predictor"] is None:
            cfg = Config.load(cfg_path) if Path(cfg_path).exists() else Config()
            if ckpt_path:
                cfg.infer.ckpt = ckpt_path
            if not Path(cfg.infer.ckpt).exists():
                logger.warning("未找到权重 %s，使用随机初始化模型（仅用于接口联调）", cfg.infer.ckpt)
            state["predictor"] = DeepfakePredictor(cfg=cfg, ckpt=cfg.infer.ckpt)
            logger.info("模型加载完成: backbone=%s device=%s", cfg.model.backbone, state["predictor"].device_name)
        return state["predictor"]

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        predictor = get_predictor()
        return {
            "status": "ok",
            "device": predictor.device_name,
            "backbone": predictor.backbone,
            "image_size": predictor.image_size,
            "frames_per_video": predictor.frames_per_video,
            "threshold": predictor.threshold,
            "audio_module": predictor.audio.available,
            "audio_ckpt": predictor.audio.ckpt,
        }

    async def _read_upload(file: UploadFile) -> bytes:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="上传内容为空")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="文件超出大小限制")
        return data

    def _write_temp(data: bytes, suffix: str, bucket: list) -> str:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            bucket.append(tmp.name)
        return tmp.name

    @app.post("/predict/image")
    async def predict_image(
        file: UploadFile = File(...),
        provenance: Optional[float] = Query(None, ge=0.0, le=1.0, description="来源凭证完备度"),
        duplication: Optional[float] = Query(None, ge=0.0, le=1.0, description="内容重复度"),
    ) -> Dict[str, Any]:
        data = await _read_upload(file)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="无法解码图片")
        predictor = get_predictor()
        try:
            result = predictor.predict_image(image, meta=_meta_from_query(provenance, duplication))
        except Exception as exc:  # pragma: no cover - 运行时保护
            raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from exc
        return result

    @app.post("/predict/video")
    async def predict_video(
        file: UploadFile = File(...),
        audio: Optional[UploadFile] = File(None, description="可选：同一素材的音轨，用于多模态融合"),
        num_frames: Optional[int] = Query(None, ge=1, le=64, description="均匀抽帧数量"),
        provenance: Optional[float] = Query(None, ge=0.0, le=1.0),
        duplication: Optional[float] = Query(None, ge=0.0, le=1.0),
    ) -> Dict[str, Any]:
        data = await _read_upload(file)
        suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
        predictor = get_predictor()
        tmp_paths = []
        try:
            video_path = _write_temp(data, suffix, tmp_paths)
            audio_path = None
            if audio is not None:
                audio_data = await _read_upload(audio)
                audio_suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
                audio_path = _write_temp(audio_data, audio_suffix, tmp_paths)
            result = predictor.predict_video(
                video_path,
                num_frames=num_frames,
                meta=_meta_from_query(provenance, duplication),
                audio_path=audio_path,
            )
        except Exception as exc:  # pragma: no cover - 运行时保护
            raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from exc
        finally:
            for path in tmp_paths:
                Path(path).unlink(missing_ok=True)
        return result

    @app.post("/predict/audio")
    async def predict_audio(
        file: UploadFile = File(...),
        provenance: Optional[float] = Query(None, ge=0.0, le=1.0),
        duplication: Optional[float] = Query(None, ge=0.0, le=1.0),
    ) -> Dict[str, Any]:
        data = await _read_upload(file)
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        predictor = get_predictor()
        if not predictor.audio.available:
            raise HTTPException(status_code=503, detail="音频检测权重未加载，请先用 train_audio.py 训练")
        tmp_paths = []
        try:
            audio_path = _write_temp(data, suffix, tmp_paths)
            result = predictor.predict_audio(
                audio_path, meta=_meta_from_query(provenance, duplication)
            )
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - 运行时保护
            raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from exc
        finally:
            for path in tmp_paths:
                Path(path).unlink(missing_ok=True)
        return result

    return app


app = create_app()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("GEO_DF_CONFIG", "configs/default.yaml"))
    parser.add_argument("--ckpt", default=os.environ.get("GEO_DF_CKPT"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global app
    app = create_app(args.config, args.ckpt)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
