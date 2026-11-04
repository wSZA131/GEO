"""核心逻辑单测（不依赖 GPU / 不需要训练）。

运行：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geo_deepfake.config import Config  # noqa: E402
from geo_deepfake.data.preprocess import FrameTransform, frame_indices  # noqa: E402
from geo_deepfake.engine.metrics import compute_eer, compute_metrics  # noqa: E402
from geo_deepfake.models.audio import AudioDetector  # noqa: E402
from geo_deepfake.models.fusion import FusionScorer, geo_trust_score  # noqa: E402


class TestPreprocess(unittest.TestCase):
    def test_frame_indices_uniform(self):
        self.assertEqual(frame_indices(10, 10), list(range(10)))
        self.assertEqual(frame_indices(100, 5), [0, 25, 50, 74, 99])
        self.assertEqual(frame_indices(0, 5), [])

    def test_transform_shape_and_range(self):
        tf = FrameTransform(size=112, train=False)
        out = tf(np.full((240, 320, 3), 128, dtype=np.uint8))
        self.assertEqual(out.shape, (3, 112, 112))
        self.assertEqual(out.dtype, np.float32)

    def test_transform_train_augment_keeps_shape(self):
        tf = FrameTransform(size=64, train=True, jpeg_quality=(50, 60))
        rng = np.random.default_rng(0)
        out = tf(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8), rng=rng)
        self.assertEqual(out.shape, (3, 64, 64))


class TestMetrics(unittest.TestCase):
    def test_perfect_separation(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]
        metrics = compute_metrics(y_true, y_score)
        self.assertAlmostEqual(metrics["auc"], 1.0)
        self.assertAlmostEqual(metrics["eer"], 0.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)

    def test_eer_midpoint(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.4, 0.6, 0.4, 0.6]
        eer, threshold = compute_eer(y_true, y_score)
        self.assertAlmostEqual(eer, 0.5)
        self.assertTrue(0.4 <= threshold <= 0.6)

    def test_single_class_returns_nan(self):
        metrics = compute_metrics([0, 0], [0.1, 0.2])
        self.assertTrue(np.isnan(metrics["auc"]))


class TestAudio(unittest.TestCase):
    def test_logmel_shape(self):
        from geo_deepfake.data.audio import LogMel

        feat = LogMel(n_mels=64)(np.random.randn(16000).astype(np.float32))
        self.assertEqual(feat.shape[0], 64)
        self.assertAlmostEqual(feat.shape[1], 1 + (16000 - 400) // 160)

    def test_lfcc_shape_and_dct(self):
        from geo_deepfake.data.audio import LFCC, dct_matrix

        feat = LFCC(n_filters=40, n_ceps=20)(np.random.randn(16000).astype(np.float32))
        self.assertEqual(feat.shape[0], 20)
        self.assertAlmostEqual(feat.shape[1], 1 + (16000 - 400) // 160)
        mat = dct_matrix(20, 40)
        self.assertEqual(mat.shape, (20, 40))

    def test_iter_chunks_short_audio_padded(self):
        from geo_deepfake.data.audio import iter_chunks

        chunks = list(iter_chunks(np.zeros(8000, dtype=np.float32), window_s=1.5, hop_s=0.5))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].size, 24000)

    def test_iter_chunks_long_audio(self):
        from geo_deepfake.data.audio import iter_chunks

        chunks = list(iter_chunks(np.zeros(48000, dtype=np.float32), window_s=1.0, hop_s=0.5))
        self.assertEqual(len(chunks), 5)  # 0,0.5,1,1.5,2 秒起点
        self.assertTrue(all(c.size == 16000 for c in chunks))


class TestPaperImprovements(unittest.TestCase):
    def test_sentry_feature_dim(self):
        from geo_deepfake.data.wavelet_features import SentryFeatures, feature_dim

        feat = SentryFeatures()(np.random.randn(16000).astype(np.float32) * 0.1)
        self.assertEqual(feat.shape[0], feature_dim())
        self.assertTrue(np.isfinite(feat).all())

    def test_patch_detector_forward(self):
        import torch

        from geo_deepfake.models.patch_visual import PatchDetector

        model = PatchDetector(image_size=64, patch_size=16, patch_stride=8, patch_pool="topk")
        out = model.forward_frames(torch.zeros(3, 3, 64, 64))
        self.assertEqual(tuple(out.shape), (3,))
        self.assertEqual(model.num_patches, 49)

    def test_factory_switches_arch(self):
        from geo_deepfake.config import Config
        from geo_deepfake.models.factory import build_visual_detector

        cfg = Config()
        cfg.model.arch = "patch"
        cfg.data.image_size = 64
        self.assertEqual(build_visual_detector(cfg).patch_size, 32)
        cfg.model.arch = "cnn"
        self.assertTrue(hasattr(build_visual_detector(cfg), "backbone"))


class TestFusion(unittest.TestCase):
    def test_audio_missing_renormalizes(self):
        scorer = FusionScorer(weights={"visual": 0.8, "audio": 0.2})
        result = scorer.score(visual_prob=0.9)
        self.assertAlmostEqual(result.fake_prob, 0.9)
        self.assertEqual(result.verdict, "fake")
        self.assertTrue(any("audio" in note for note in result.notes))

    def test_audio_present_weighted(self):
        scorer = FusionScorer(weights={"visual": 0.5, "audio": 0.5})
        result = scorer.score(visual_prob=1.0, audio_prob=0.0)
        self.assertAlmostEqual(result.fake_prob, 0.5)

    def test_geo_penalty_lowers_trust(self):
        high = geo_trust_score(0.1, meta={"provenance": 1.0, "duplication": 0.0})["trust_score"]
        low = geo_trust_score(0.1, meta={"provenance": 0.0, "duplication": 1.0})["trust_score"]
        self.assertGreater(high, low)
        self.assertLessEqual(low, 1.0)

    def test_audio_placeholder_unavailable(self):
        result = AudioDetector().predict_stream([np.zeros(1600, dtype=np.float32)])
        self.assertFalse(result.available)
        self.assertIsNone(result.fake_prob)


class TestConfig(unittest.TestCase):
    def test_roundtrip(self):
        cfg = Config()
        cfg.model.fusion_weights = {"visual": 0.7, "audio": 0.3}
        restored = Config.from_dict(cfg.to_dict())
        self.assertEqual(restored.model.fusion_weights["audio"], 0.3)
        self.assertEqual(restored.data.jpeg_quality, cfg.data.jpeg_quality)

    def test_unknown_field_rejected(self):
        with self.assertRaises(ValueError):
            Config.from_dict({"model": {"not_a_field": 1}})


if __name__ == "__main__":
    unittest.main()
