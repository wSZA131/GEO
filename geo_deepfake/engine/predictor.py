"""实时推理引擎：图像 / 视频 → 帧级伪造概率 + 视频级判决 + GEO 可信评分。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import Config
from ..data.audio import read_audio
from ..data.preprocess import FaceCropper, FrameTransform, build_face_cropper, read_image, read_video_frames
from ..models.audio import AudioDetector
from ..models.factory import build_visual_detector
from ..models.fusion import FusionScorer
from ..utils import resolve_device


class DeepfakePredictor:
    """加载权重并完成端到端推理（预处理 → 视觉检测 → 融合评分）。"""

    def __init__(
        self,
        cfg: Optional[Config] = None,
        ckpt: Optional[str | Path] = None,
        device: Optional[str] = None,
        backbone: Optional[str] = None,
        pretrained: bool = False,
        temporal: Optional[str] = None,
        threshold: Optional[float] = None,
        image_size: Optional[int] = None,
        frames_per_video: Optional[int] = None,
        use_face_crop: Optional[bool] = None,
    ):
        import torch

        self.cfg = cfg or Config()
        self.device = torch.device(resolve_device(device or self.cfg.infer.device))
        self.threshold = float(threshold if threshold is not None else self.cfg.infer.threshold)
        self.image_size = int(image_size or self.cfg.data.image_size)
        self.frames_per_video = int(frames_per_video or self.cfg.data.frames_per_video)
        self.use_face_crop = (
            self.cfg.data.use_face_crop if use_face_crop is None else bool(use_face_crop)
        )

        ckpt_path = Path(ckpt or self.cfg.infer.ckpt)
        state, ckpt_cfg = self._read_ckpt(ckpt_path)
        if ckpt_cfg is not None:
            # 只沿用训练时的模型结构参数；运行时路径（音频权重/融合权重）以当前配置为准，
            # 否则旧权重里固化的旧配置会覆盖新加的字段。
            for name in ("audio_enabled", "audio_ckpt", "audio_feature", "audio_n_mels", "audio_n_ceps",
                         "audio_n_filters", "audio_window_s", "audio_hop_s", "audio_threshold",
                         "fusion_weights", "geo_signals"):
                setattr(ckpt_cfg.model, name, getattr(self.cfg.model, name))
            self.cfg = ckpt_cfg
        self.backbone = backbone or self.cfg.model.backbone
        self.temporal = temporal or self.cfg.model.temporal

        self.cfg.model.pretrained = pretrained
        self.model = build_visual_detector(self.cfg, image_size=self.image_size)
        if state is not None:
            self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.cropper = (
            build_face_cropper("haar", margin=self.cfg.data.face_margin, size=None)
            if self.use_face_crop
            else None
        )
        self.transform = FrameTransform(size=self.image_size, train=False, face_cropper=self.cropper)
        self.fusion = FusionScorer(
            weights=self.cfg.model.fusion_weights,
            geo_weights=self.cfg.model.geo_signals,
            threshold=self.threshold,
        )
        # 音频检测模块：权重缺失时 available=False，融合自动降级
        self.audio = AudioDetector(
            ckpt=self.cfg.model.audio_ckpt,
            feature=self.cfg.model.audio_feature,
            n_mels=self.cfg.model.audio_n_mels,
            n_ceps=self.cfg.model.audio_n_ceps,
            n_filters=self.cfg.model.audio_n_filters,
            window_s=self.cfg.model.audio_window_s,
            hop_s=self.cfg.model.audio_hop_s,
            threshold=self.cfg.model.audio_threshold,
            device=self.device_name,
        )
        self._warmed_up = False

    # ---------------------------------------------------------------- 权重
    @staticmethod
    def _read_ckpt(path: Path):
        import torch

        if not path.exists():
            return None, None
        payload = torch.load(str(path), map_location="cpu")
        if isinstance(payload, dict) and "model" in payload:
            cfg = Config.from_dict(payload.get("config")) if payload.get("config") else None
            return payload["model"], cfg
        return payload, None

    @property
    def device_name(self) -> str:
        return str(self.device)

    # ---------------------------------------------------------------- 预处理
    def prepare_frames(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack([self.transform(frame) for frame in frames], axis=0)

    # ---------------------------------------------------------------- 推理
    def predict_frames(
        self,
        frames: Sequence[np.ndarray],
        meta: Optional[Dict[str, float]] = None,
        audio_prob: Optional[float] = None,
    ) -> Dict[str, Any]:
        """BGR 帧序列 → 判决结果。帧为空时抛 ValueError。"""
        import torch

        if len(frames) == 0:
            raise ValueError("输入帧为空")
        start = time.perf_counter()
        array = self.prepare_frames(frames)
        batch = torch.from_numpy(array).unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        with torch.inference_mode():
            flat = batch.reshape(-1, *batch.shape[2:])
            frame_logits = self.model.forward_frames(flat)
            video_logit = self.model.aggregator(frame_logits.unsqueeze(0))
            frame_probs = torch.sigmoid(frame_logits).float().cpu().numpy()
            video_prob = float(torch.sigmoid(video_logit).float().item())
        latency_ms = (time.perf_counter() - start) * 1000.0

        fused = self.fusion.score(visual_prob=video_prob, audio_prob=audio_prob, meta=meta)
        return {
            "fake_prob": video_prob,
            "verdict": fused.verdict,
            "trust_score": fused.trust_score,
            "frame_probs": [float(p) for p in frame_probs],
            "num_frames": len(frames),
            "latency_ms": latency_ms,
            "per_frame_ms": latency_ms / len(frames),
            "device": self.device_name,
            "fusion": fused.to_dict(),
        }

    def predict_image(self, image: str | Path | np.ndarray, meta: Optional[Dict[str, float]] = None):
        frame = read_image(image) if isinstance(image, (str, Path)) else image
        return self.predict_frames([frame], meta=meta)

    def predict_video(
        self,
        path: str | Path,
        num_frames: Optional[int] = None,
        meta: Optional[Dict[str, float]] = None,
        audio_path: Optional[str | Path] = None,
    ):
        frames = read_video_frames(path, num_frames or self.frames_per_video)
        if not frames:
            raise ValueError(f"视频未抽取到帧: {path}")
        audio_prob = None
        if audio_path is not None:
            audio_result = self.audio.predict_file(audio_path)
            audio_prob = audio_result.fake_prob
        return self.predict_frames(frames, meta=meta, audio_prob=audio_prob)

    def predict_audio(
        self,
        path: str | Path,
        meta: Optional[Dict[str, float]] = None,
        visual_prob: Optional[float] = None,
    ) -> Dict[str, Any]:
        """音频单模态推理（流式分块），也可与视觉概率直接融合。"""
        result = self.audio.predict_file(path)
        if result.fake_prob is None:
            raise RuntimeError(result.note or "音频检测模块不可用")
        fused = self.fusion.score(visual_prob=visual_prob, audio_prob=result.fake_prob, meta=meta)
        return {
            "fake_prob": result.fake_prob,
            "verdict": fused.verdict,
            "trust_score": fused.trust_score,
            "num_chunks": result.num_chunks,
            "latency_ms": result.latency_ms,
            "audio_seconds": (result.num_chunks - 1) * self.audio.hop_s + self.audio.window_s,
            "device": self.device_name,
            "fusion": fused.to_dict(),
        }

    def predict_frame_paths(
        self,
        paths: Sequence[str | Path],
        meta: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        frames = [read_image(p) for p in paths]
        return self.predict_frames(frames, meta=meta)

    # ---------------------------------------------------------------- 其他
    def warmup(self, rounds: int = 3) -> None:
        import torch

        dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        with torch.inference_mode():
            for _ in range(rounds):
                self.predict_frames([dummy] * 2)
        self._warmed_up = True
