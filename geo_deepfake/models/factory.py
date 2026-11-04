"""视觉检测器工厂：按配置在轻量骨干（cnn）与 LaDeDa 式 patch 检测器（patch）间切换。"""

from __future__ import annotations

from ..config import Config
from .patch_visual import PatchDetector
from .visual import VisualDetector


def build_visual_detector(cfg: Config, image_size: int | None = None):
    model_cfg = cfg.model
    arch = (model_cfg.arch or "cnn").lower()
    if arch == "patch":
        return PatchDetector(
            pretrained=model_cfg.pretrained,
            dropout=model_cfg.dropout,
            temporal=model_cfg.temporal,
            image_size=image_size or cfg.data.image_size,
            patch_size=model_cfg.patch_size,
            patch_stride=model_cfg.patch_stride,
            patch_pool=model_cfg.patch_pool,
            topk_ratio=model_cfg.patch_topk,
        )
    if arch == "cnn":
        return VisualDetector(
            backbone=model_cfg.backbone,
            pretrained=model_cfg.pretrained,
            dropout=model_cfg.dropout,
            temporal=model_cfg.temporal,
        )
    raise ValueError(f"未知视觉架构: {arch}")
