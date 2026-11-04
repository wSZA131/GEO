"""多模态融合模块：各模态概率 → 综合伪造概率 + GEO 内容可信评分。

说明：开题报告图 1 中"GEO 特征融合模块"一处写作"地理空间 (GEO) 特征提取"，
与本报告正文的 Generative Engine Optimization 不一致。本实现按正文口径处理：
GEO 信号指营销内容的**可引用性/可追溯性/重复度**等可信性元数据，
而非地理空间坐标特征。

融合策略（最小版）：
    fake_prob = Σ w_m * p_m / Σ w_m        （缺失模态自动剔除并归一化）
    trust     = α * (1 - fake_prob) + (1 - α) * (1 - geo_penalty)
其中 geo_penalty 由 provenance（来源凭证是否完备）与 duplication（内容重复度）加权得到。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class FusionResult:
    fake_prob: float
    trust_score: float
    verdict: str
    components: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "fake_prob": round(self.fake_prob, 6),
            "trust_score": round(self.trust_score, 6),
            "verdict": self.verdict,
            "components": {k: round(float(v), 6) for k, v in self.components.items()},
            "notes": self.notes,
        }


def geo_trust_score(
    fake_prob: float,
    meta: Optional[Dict[str, float]] = None,
    geo_weights: Optional[Dict[str, float]] = None,
    alpha: float = 0.7,
) -> Dict[str, float]:
    """把伪造概率与 GEO 可信性元数据合成为可信评分。

    meta 可含：
        provenance: 0~1，来源凭证完备度（C2PA 签名、品牌授权、可追溯首发出处）；
        duplication: 0~1，与其他内容的重复/伪共识程度（越高越可疑）。
    缺失 meta 时退化为 1 - fake_prob。
    """
    geo_weights = geo_weights or {"provenance": 0.5, "duplication": 0.5}
    base = float(np.clip(1.0 - fake_prob, 0.0, 1.0))
    if not meta:
        return {"trust_score": base, "geo_penalty": 0.0, "base": base}

    w_prov = float(geo_weights.get("provenance", 0.5))
    w_dup = float(geo_weights.get("duplication", 0.5))
    total = max(w_prov + w_dup, 1e-6)
    provenance = float(np.clip(meta.get("provenance", 0.0), 0.0, 1.0))
    duplication = float(np.clip(meta.get("duplication", 0.0), 0.0, 1.0))
    geo_penalty = (w_prov * (1.0 - provenance) + w_dup * duplication) / total
    trust = alpha * base + (1.0 - alpha) * (1.0 - geo_penalty)
    return {
        "trust_score": float(np.clip(trust, 0.0, 1.0)),
        "geo_penalty": geo_penalty,
        "base": base,
        "provenance": provenance,
        "duplication": duplication,
    }


class FusionScorer:
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        geo_weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
        alpha: float = 0.7,
    ):
        self.weights = weights or {"visual": 0.8, "audio": 0.2}
        self.geo_weights = geo_weights or {"provenance": 0.5, "duplication": 0.5}
        self.threshold = threshold
        self.alpha = alpha

    def score(
        self,
        visual_prob: Optional[float],
        audio_prob: Optional[float] = None,
        meta: Optional[Dict[str, float]] = None,
    ) -> FusionResult:
        probs = {"visual": visual_prob, "audio": audio_prob}
        used, num, den, notes = {}, 0.0, 0.0, []
        for name, prob in probs.items():
            w = float(self.weights.get(name, 0.0))
            if prob is None:
                if w > 0:
                    notes.append(f"{name} 模态无输出，融合权重重新归一化")
                continue
            used[name] = float(prob)
            num += w * float(prob)
            den += w
        if den <= 0:
            raise ValueError("至少需要一个可用模态的概率输出")
        fake_prob = num / den
        geo = geo_trust_score(fake_prob, meta=meta, geo_weights=self.geo_weights, alpha=self.alpha)
        components = dict(used)
        components["geo_penalty"] = geo["geo_penalty"]
        return FusionResult(
            fake_prob=fake_prob,
            trust_score=geo["trust_score"],
            verdict="fake" if fake_prob >= self.threshold else "real",
            components=components,
            notes=notes,
        )
