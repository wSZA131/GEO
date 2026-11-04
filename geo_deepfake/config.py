"""工程配置（对应开题报告 3.4 开发环境 / 3.3 模块参数）。"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class DataConfig:
    """预处理模块参数。"""

    manifest: str = "data/dummy/manifest.csv"
    image_size: int = 224
    frames_per_video: int = 8
    use_face_crop: bool = True
    face_margin: float = 0.3
    num_workers: int = 2
    # 多维增强：模拟真实营销分发链路中的压缩与画质衰减
    jpeg_quality: tuple = (30, 95)
    blur_prob: float = 0.2
    brightness_delta: float = 0.15
    flip_prob: float = 0.5
    # 跨域增强（默认关闭，跨库实验时打开）
    hf_prob: float = 0.0
    hf_alpha: tuple = (0.4, 1.6)
    recompress_prob: float = 0.0
    chroma_noise_prob: float = 0.0


@dataclass
class ModelConfig:
    """视觉/音频/融合模块参数。"""

    backbone: str = "efficientnet_b0"
    pretrained: bool = False  # 离线环境默认不下载权重
    dropout: float = 0.2
    temporal: str = "mean"  # mean | attn，对应帧级 → 视频级时序聚合
    # 视觉架构：cnn=轻量骨干；patch=LaDeDa 式小感受野 patch 检测器
    arch: str = "cnn"
    patch_size: int = 32
    patch_stride: int = 16
    patch_pool: str = "mean"  # mean | topk
    patch_topk: float = 0.25
    # 音频检测模块（ASVspoof 2019 训练的轻量 CNN）
    audio_enabled: bool = False
    audio_ckpt: str = ""
    audio_feature: str = "lfcc"  # lfcc | logmel
    audio_n_mels: int = 64
    audio_n_ceps: int = 20
    audio_n_filters: int = 40
    audio_window_s: float = 1.5
    audio_hop_s: float = 0.5
    audio_threshold: float = 0.5
    # 多模态融合权重（音频不可用时自动在视觉上归一化）
    fusion_weights: Dict[str, float] = field(default_factory=lambda: {"visual": 0.8, "audio": 0.2})
    # GEO 可信评分的元数据信号权重
    geo_signals: Dict[str, float] = field(default_factory=lambda: {"provenance": 0.5, "duplication": 0.5})


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    amp: bool = True
    seed: int = 42
    out_dir: str = "runs/exp1"


@dataclass
class InferConfig:
    ckpt: str = "runs/exp1/best.pt"
    device: str = "cuda"
    threshold: float = 0.5


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    api: ApiConfig = field(default_factory=ApiConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "Config":
        payload = payload or {}
        sections = {
            "data": DataConfig,
            "model": ModelConfig,
            "train": TrainConfig,
            "infer": InferConfig,
            "api": ApiConfig,
        }
        kwargs = {}
        for name, klass in sections.items():
            raw = dict(payload.get(name) or {})
            valid = {f.name for f in dataclasses.fields(klass)}
            unknown = set(raw) - valid
            if unknown:
                raise ValueError(f"配置段 {name} 含未知字段: {sorted(unknown)}")
            if name == "data" and "jpeg_quality" in raw:
                raw["jpeg_quality"] = tuple(raw["jpeg_quality"])
            kwargs[name] = klass(**raw)
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))


def load_config(path: str | Path | None = None) -> Config:
    return Config.load(path)
