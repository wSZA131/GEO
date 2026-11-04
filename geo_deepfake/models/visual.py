"""视觉检测模块：帧级深度伪造识别 + 帧得分时序聚合。

开题报告 3.2 要求"图像端基于轻量骨干实现帧级分类与时序聚合"，此处：
- 骨干来自 timm（默认 efficientnet_b0，另有 mobilenetv3 / resnet18 等可选）；
- 帧级二分类输出单 logit（P(fake) = sigmoid(logit)）；
- 视频级通过 mean / attention 聚合帧得分，避免引入重型时序网络以保实时性。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def create_backbone(name: str = "efficientnet_b0", pretrained: bool = False, drop_path: float = 0.0):
    """返回 (backbone, feature_dim)；backbone 输出全局池化后的特征。"""
    import timm

    model = timm.create_model(name, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path)
    return model, int(getattr(model, "num_features"))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class TemporalAggregator(nn.Module):
    """帧级 logit → 视频级 logit。"""

    def __init__(self, mode: str = "mean"):
        super().__init__()
        if mode not in ("mean", "attn"):
            raise ValueError(f"unsupported temporal mode: {mode}")
        self.mode = mode
        if mode == "attn":
            self.scorer = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))

    def forward(self, frame_logits: torch.Tensor) -> torch.Tensor:
        """frame_logits: (B, T) → (B,)"""
        if self.mode == "mean":
            return frame_logits.mean(dim=1)
        weights = torch.softmax(self.scorer(frame_logits.unsqueeze(-1)).squeeze(-1), dim=1)
        return (weights * frame_logits).sum(dim=1)


class VisualDetector(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = False,
        dropout: float = 0.2,
        temporal: str = "mean",
    ):
        super().__init__()
        self.backbone_name = backbone
        self.temporal_mode = temporal
        self.backbone, feat_dim = create_backbone(backbone, pretrained=pretrained)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, 1))
        self.aggregator = TemporalAggregator(temporal)

    def forward_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (N, 3, H, W) → logits (N,)"""
        feats = self.backbone(frames)
        return self.head(feats).squeeze(-1)

    def forward(self, video: torch.Tensor, chunk: int = 16) -> torch.Tensor:
        """video: (B, T, 3, H, W) → 视频级 logits (B,)。

        分块前向以控制显存；T 较大时避免一次性展开全部帧。
        """
        if video.dim() == 4:
            return self.forward_frames(video)
        b, t = video.shape[0], video.shape[1]
        flat = video.reshape(b * t, *video.shape[2:])
        outs = [self.forward_frames(flat[i : i + chunk]) for i in range(0, flat.size(0), chunk)]
        frame_logits = torch.cat(outs).view(b, t)
        return self.aggregator(frame_logits)

    @staticmethod
    def prob_from_logits(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)


def build_visual_model(
    backbone: str = "efficientnet_b0",
    pretrained: bool = False,
    dropout: float = 0.2,
    temporal: str = "mean",
    device: str | torch.device | None = None,
) -> Tuple[VisualDetector, torch.device]:
    model = VisualDetector(backbone=backbone, pretrained=pretrained, dropout=dropout, temporal=temporal)
    target = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return model.to(target), target
