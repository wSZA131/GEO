"""音频检测模块：流式语音克隆/合成语音检测。

对应开题报告 3.3 第 3 行"流式语音克隆与合成检测：音频流 → 伪造概率"。
实现路线（对齐 3.2 的"流式特征 + 轻量分类器"）：
    flac/wav → 16 kHz 单声道 → 1.5 s 窗 / 0.5 s 跳步切块 → 64 维 log-Mel
    → 轻量 CNN（约 6 万参数）→ 块级 P(fake) → 滑窗平均

未加载权重时 `available = False`，上层融合自动降级为视觉单模态。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from ..data.audio import SAMPLE_RATE, build_feature, iter_chunks, read_audio


@dataclass
class AudioResult:
    fake_prob: Optional[float]
    num_chunks: int
    latency_ms: float
    available: bool
    note: str = ""


def _build_model(n_mels: int = 64, channels: Sequence[int] = (16, 32, 64, 64), dropout: float = 0.2):
    import torch.nn as nn

    class AudioCNN(nn.Module):
        """轻量 2D CNN：(B,1,n_mels,T) → logit。"""

        def __init__(self):
            super().__init__()
            layers = []
            in_ch = 1
            for out_ch in channels:
                layers += [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                ]
                in_ch = out_ch
            self.features = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_ch, 1))

        def forward(self, x):
            return self.head(self.pool(self.features(x)).flatten(1)).squeeze(-1)

    return AudioCNN()


class AudioDetector:
    """流式音频伪造检测器；加载权重后 available=True。"""

    available: bool = False

    def __init__(
        self,
        ckpt: str | Path = "",
        sample_rate: int = SAMPLE_RATE,
        feature: str = "logmel",
        n_mels: int = 64,
        n_ceps: int = 20,
        n_filters: int = 40,
        window_s: float = 1.5,
        hop_s: float = 0.5,
        threshold: float = 0.5,
        device: str | None = None,
    ):
        self.ckpt = str(ckpt or "")
        self.sample_rate = sample_rate
        self.feature_type = feature
        self.n_mels = n_mels
        self.n_ceps = n_ceps
        self.n_filters = n_filters
        self.window_s = window_s
        self.hop_s = hop_s
        self.threshold = threshold
        self.feature = build_feature(feature, n_mels=n_mels, n_ceps=n_ceps, n_filters=n_filters)
        self.chunk_ms = int(window_s * 1000)
        self.model = None
        self.device = device
        if self.ckpt and Path(self.ckpt).exists():
            self._load(self.ckpt)

    # ---------------------------------------------------------------- 权重
    def _load(self, path: str | Path) -> None:
        import torch

        payload = torch.load(str(path), map_location="cpu")
        state = payload.get("model", payload)
        meta = payload.get("audio", {}) if isinstance(payload, dict) else {}
        self.feature_type = str(meta.get("feature", self.feature_type))
        self.n_mels = int(meta.get("n_mels", self.n_mels))
        self.n_ceps = int(meta.get("n_ceps", self.n_ceps))
        self.n_filters = int(meta.get("n_filters", self.n_filters))
        self.sample_rate = int(meta.get("sample_rate", self.sample_rate))
        self.window_s = float(meta.get("window_s", self.window_s))
        self.hop_s = float(meta.get("hop_s", self.hop_s))
        self.threshold = float(meta.get("threshold", self.threshold))
        self.feature = build_feature(
            self.feature_type, n_mels=self.n_mels, n_ceps=self.n_ceps, n_filters=self.n_filters
        )

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _build_model(self.feature.dim, channels=meta.get("channels", (16, 32, 64, 64)))
        self.model.load_state_dict(state)
        self.device = device
        self.model.to(device).eval()
        self.available = True

    @classmethod
    def load(cls, ckpt: str | Path = "", **kwargs) -> "AudioDetector":
        return cls(ckpt=ckpt, **kwargs)

    # ---------------------------------------------------------------- 推理
    def _infer_chunk(self, chunk: np.ndarray) -> Optional[float]:
        if self.model is None:
            return None
        import torch

        feat = self.feature(np.asarray(chunk, dtype=np.float32))  # (n_mels, T)
        tensor = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logit = self.model(tensor)
            return float(torch.sigmoid(logit).float().item())

    def predict_stream(self, chunks: Iterable[np.ndarray]) -> AudioResult:
        """对音频块序列打分，返回平均伪造概率。"""
        import time

        start = time.perf_counter()
        if not self.available:
            return AudioResult(
                fake_prob=None,
                num_chunks=0,
                latency_ms=(time.perf_counter() - start) * 1000,
                available=False,
                note="音频检测权重未加载，融合层将只使用视觉模态",
            )
        probs = []
        for chunk in chunks:
            prob = self._infer_chunk(np.asarray(chunk))
            if prob is not None:
                probs.append(float(prob))
        return AudioResult(
            fake_prob=float(np.mean(probs)) if probs else None,
            num_chunks=len(probs),
            latency_ms=(time.perf_counter() - start) * 1000,
            available=True,
        )

    def predict_waveform(self, wav: np.ndarray) -> AudioResult:
        return self.predict_stream(
            iter_chunks(wav, self.sample_rate, self.window_s, self.hop_s)
        )

    def predict_file(self, path: str | Path) -> AudioResult:
        return self.predict_waveform(read_audio(path, self.sample_rate))
