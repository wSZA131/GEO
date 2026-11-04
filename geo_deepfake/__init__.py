"""GEO 营销场景下的实时深度伪造检测 —— 最小可跑通实现。

模块与开题报告 3.3 的划分对应：
    预处理模块      geo_deepfake.data.preprocess
    视觉检测模块    geo_deepfake.models.visual
    音频检测模块    geo_deepfake.models.audio   （接口占位）
    多模态融合模块  geo_deepfake.models.fusion  （接口 + 视觉单模态降级）
    实时推理引擎    geo_deepfake.engine.predictor
    评估与接口层    geo_deepfake.engine.metrics / geo_deepfake.api.server
"""

__version__ = "0.1.0"
