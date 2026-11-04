"""ONNX Runtime 推理封装（对应开题报告 3.2 的 ONNX 加速 / INT8 量化）。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


class OnnxClassifier:
    """二分类 logit 模型的 ONNX Runtime 会话。"""

    def __init__(
        self,
        path: str | Path,
        providers: Optional[Sequence[str]] = None,
        intra_threads: Optional[int] = None,
    ):
        import onnxruntime as ort

        options = ort.SessionOptions()
        if intra_threads:
            options.intra_op_num_threads = int(intra_threads)
        self.path = str(path)
        self.session = ort.InferenceSession(
            self.path,
            sess_options=options,
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.providers = self.session.get_providers()

    def logits(self, array: np.ndarray) -> np.ndarray:
        feed = {self.input_name: np.ascontiguousarray(array, dtype=np.float32)}
        return np.asarray(self.session.run([self.output_name], feed)[0])

    def probs(self, array: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logits(array)))
