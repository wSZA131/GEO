"""SENTRY 风格音频检测器：梯度提升树 + Mahalanobis 离群检测。

对应论文：SENTRY (Banerjee et al., 2026) 的"监督 GBDT + 仅用真实语音训练的
Mahalanobis OOD 检测器 + 轻量 logistic 融合"三件套。

与 `models/audio.py` 的 log-Mel CNN 保持同一对外契约（`predict_waveform` /
`predict_file` 返回 `AudioResult`），便于在融合层与 API 中互换。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..data.audio import iter_chunks, read_audio
from ..data.wavelet_features import SentryFeatures


@dataclass
class AudioResult:
    fake_prob: Optional[float]
    num_chunks: int
    latency_ms: float
    available: bool
    note: str = ""


class SentryAudioDetector:
    """整数小波特征 + GBDT(+OOD) 的流式音频伪造检测器。"""

    available: bool = False

    def __init__(
        self,
        model_path: str | Path = "",
        sample_rate: int = 16000,
        window_s: float = 4.0,
        hop_s: float = 2.0,
        threshold: float = 0.5,
    ):
        self.model_path = str(model_path or "")
        self.sample_rate = sample_rate
        self.window_s = window_s
        self.hop_s = hop_s
        self.threshold = threshold
        self.chunk_ms = int(window_s * 1000)
        self.features = SentryFeatures()
        self.clf = None
        self.ood_mean: Optional[np.ndarray] = None
        self.ood_prec: Optional[np.ndarray] = None
        self.ood_tau: float = 0.0
        self.fusion_alpha: float = 0.8
        if self.model_path and Path(self.model_path).exists():
            self.load(self.model_path)

    # ---------------------------------------------------------------- 训练 / 存取
    def fit(self, X: np.ndarray, y: np.ndarray, authentic: np.ndarray | None = None,
            max_iter: int = 300, learning_rate: float = 0.1):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.clf = HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=31,
            early_stopping=False, random_state=42,
        )
        self.clf.fit(X, y)

        # 无监督 OOD：只用真实语音估计高斯分布
        if authentic is None:
            authentic = X[y == 0]
        self.ood_mean = authentic.mean(axis=0)
        cov = np.cov(authentic - self.ood_mean, rowvar=False) + np.eye(X.shape[1]) * 1e-3
        self.ood_prec = np.linalg.pinv(cov)
        dist = self._mahalanobis(authentic)
        self.ood_tau = float(np.percentile(dist, 95))
        self.available = True
        return self

    def _mahalanobis(self, X: np.ndarray) -> np.ndarray:
        assert self.ood_mean is not None and self.ood_prec is not None
        diff = X - self.ood_mean
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, self.ood_prec, diff), 0.0))

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "clf": self.clf,
                "ood_mean": self.ood_mean,
                "ood_prec": self.ood_prec,
                "ood_tau": self.ood_tau,
                "fusion_alpha": self.fusion_alpha,
                "meta": {
                    "sample_rate": self.sample_rate,
                    "window_s": self.window_s,
                    "hop_s": self.hop_s,
                    "threshold": self.threshold,
                    "feature_dim": int(self.features(np.zeros(16000, dtype=np.float32)).size),
                },
            },
            path,
        )
        return path

    def load(self, path: str | Path) -> "SentryAudioDetector":
        import joblib

        payload = joblib.load(path)
        self.clf = payload["clf"]
        self.ood_mean = payload["ood_mean"]
        self.ood_prec = payload["ood_prec"]
        self.ood_tau = float(payload["ood_tau"])
        self.fusion_alpha = float(payload.get("fusion_alpha", 0.8))
        meta = payload.get("meta", {})
        self.sample_rate = int(meta.get("sample_rate", self.sample_rate))
        self.window_s = float(meta.get("window_s", self.window_s))
        self.hop_s = float(meta.get("hop_s", self.hop_s))
        self.threshold = float(meta.get("threshold", self.threshold))
        self.available = True
        return self

    # ---------------------------------------------------------------- 推理
    def predict_features(self, X: np.ndarray) -> np.ndarray:
        """特征矩阵 → 融合后的 P(fake)（GBDT 概率 + Mahalanobis OOD）。"""
        prob = self.clf.predict_proba(X)[:, 1]
        if self.ood_mean is not None and self.ood_prec is not None:
            ood = 1.0 / (1.0 + np.exp(-(self._mahalanobis(X) - self.ood_tau) / max(self.ood_tau, 1e-6)))
            prob = self.fusion_alpha * prob + (1.0 - self.fusion_alpha) * ood
        return prob.astype(np.float32)

    def chunk_probs(self, wav: np.ndarray) -> np.ndarray:
        chunks = list(iter_chunks(wav, self.sample_rate, self.window_s, self.hop_s))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        X = np.stack([self.features(c) for c in chunks])
        return self.predict_features(X)

    def predict_waveform(self, wav: np.ndarray) -> AudioResult:
        import time

        start = time.perf_counter()
        if not self.available:
            return AudioResult(None, 0, (time.perf_counter() - start) * 1000, False,
                               "SENTRY 权重未加载")
        prob = self.chunk_probs(wav)
        return AudioResult(
            fake_prob=float(prob.mean()) if prob.size else None,
            num_chunks=int(prob.size),
            latency_ms=(time.perf_counter() - start) * 1000,
            available=True,
        )

    def predict_file(self, path: str | Path) -> AudioResult:
        return self.predict_waveform(read_audio(path, self.sample_rate))
