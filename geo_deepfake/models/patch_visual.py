"""LaDeDa 启发的 patch 级视觉检测器。

对应论文：Real-Time Deepfake Detection in the Real-World (Cavia et al., 2024)

论文要点：
1. 把图像切成 q×q 的小 patch，模型**感受野被限制在 q×q 内**（BagNet 式，1×1 卷积为主），
   迫使网络只依赖局部生成痕迹而不是全局语义；
2. 图像分数 = 所有 patch 分数的**平均池化**，线性可蒸馏；
3. 蒸馏出 Tiny-LaDeDa：4 层卷积，FLOPs 少 375×，可跑在边缘设备。

本实现据此做一个 patch 级检测器：
- 输入 224×224，patch=32、stride=16（7×7=49 个 patch）；
- 每个 patch 过一个 4 层卷积小网（约 6 万参数）得到 patch logit；
- 图像分数默认取 patch 均值（LaDeDa），可选 top-k 均值以突出局部伪影；
- 对外暴露 `forward_frames` 与 `aggregator`，与 `VisualDetector` 接口一致，可直接替换。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .visual import TemporalAggregator


class TinyPatchNet(nn.Module):
    """4 层卷积的 patch 打分器（Tiny-LaDeDa 风格）。"""

    def __init__(self, patch_size: int = 32, channels: Tuple[int, ...] = (32, 64, 64, 64),
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        in_ch = 3
        for i, out_ch in enumerate(channels):
            stride = 2 if i < 3 else 1
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)).flatten(1)).squeeze(-1)


class PatchDetector(nn.Module):
    def __init__(
        self,
        backbone: str = "patch",
        pretrained: bool = False,
        dropout: float = 0.1,
        temporal: str = "mean",
        image_size: int = 224,
        patch_size: int = 32,
        patch_stride: int = 16,
        patch_pool: str = "mean",
        topk_ratio: float = 0.25,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_pool = patch_pool
        self.topk_ratio = topk_ratio
        self.patch_net = TinyPatchNet(patch_size=patch_size, dropout=dropout)
        self.aggregator = TemporalAggregator(temporal)
        self.temporal_mode = temporal

    # ---------------------------------------------------------------- patch 化
    def patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """(B,3,H,W) → (B*P,3,q,q), gh, gw。"""
        b, c, h, w = x.shape
        q, s = self.patch_size, self.patch_stride
        gh = (h - q) // s + 1
        gw = (w - q) // s + 1
        patches = x.unfold(2, q, s).unfold(3, q, s)  # (B,C,gh,gw,q,q)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(b, gh, gw, c, q, q)
        return patches.view(b * gh * gw, c, q, q), gh, gw

    def _aggregate_patches(self, patch_logits: torch.Tensor, b: int, gh: int, gw: int) -> torch.Tensor:
        scores = patch_logits.view(b, gh * gw)  # (B,P)
        if self.patch_pool == "mean":
            return scores.mean(dim=1)
        if self.patch_pool == "topk":
            k = max(1, int(scores.size(1) * self.topk_ratio))
            return scores.topk(k, dim=1).values.mean(dim=1)
        raise ValueError(f"未知 patch 池化方式: {self.patch_pool}")

    # ---------------------------------------------------------------- 前向
    def forward_frames(self, frames: torch.Tensor, return_patch_scores: bool = False):
        """(N,3,H,W) → 图像级 logits (N,)；可选返回 patch 分数用于可视化。"""
        if frames.dim() != 4:
            raise ValueError(f"forward_frames 期望 4 维输入，收到 {tuple(frames.shape)}")
        patches, gh, gw = self.patchify(frames)
        patch_logits = self.patch_net(patches)
        logits = self._aggregate_patches(patch_logits, frames.size(0), gh, gw)
        if return_patch_scores:
            b = frames.size(0)
            return logits, patch_logits.view(b, gh, gw)
        return logits

    def forward(self, video: torch.Tensor, chunk: int = 16) -> torch.Tensor:
        """(B,T,3,H,W) → 视频级 logits；4 维输入时等价于 forward_frames。"""
        if video.dim() == 4:
            return self.forward_frames(video)
        b, t = video.shape[0], video.shape[1]
        flat = video.reshape(b * t, *video.shape[2:])
        outs = [self.forward_frames(flat[i : i + chunk]) for i in range(0, flat.size(0), chunk)]
        return self.aggregator(torch.cat(outs).view(b, t))

    @property
    def num_patches(self) -> int:
        gh = (self.image_size - self.patch_size) // self.patch_stride + 1
        return gh * gh


def build_patch_detector(image_size: int = 224, patch_size: int = 32, patch_stride: int = 16,
                         patch_pool: str = "mean", topk_ratio: float = 0.25,
                         temporal: str = "mean", dropout: float = 0.1) -> PatchDetector:
    return PatchDetector(
        image_size=image_size, patch_size=patch_size, patch_stride=patch_stride,
        patch_pool=patch_pool, topk_ratio=topk_ratio, temporal=temporal, dropout=dropout,
    )
