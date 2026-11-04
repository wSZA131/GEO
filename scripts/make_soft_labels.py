"""用教师模型给训练帧打软标签（LaDeDa 式蒸馏的第一步）。

用法：
    python scripts/make_soft_labels.py --teacher runs/wild_full/best.pt \
        --manifest data/wild/manifest.csv --split train --out data/wild/soft_train.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo_deepfake.config import Config  # noqa: E402
from geo_deepfake.data.dataset import load_manifest_rows  # noqa: E402
from geo_deepfake.data.preprocess import read_image  # noqa: E402
from geo_deepfake.engine.predictor import DeepfakePredictor  # noqa: E402
from geo_deepfake.utils import setup_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="runs/wild_full/best.pt")
    parser.add_argument("--manifest", default="data/wild/manifest.csv")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch

    logger = setup_logger()
    cfg = Config.load("configs/default.yaml")
    predictor = DeepfakePredictor(cfg=cfg, ckpt=args.teacher, device=args.device)
    rows = [r for r in load_manifest_rows(args.manifest) if r["split"] == args.split]
    logger.info("教师=%s，待打标签 %d 帧", args.teacher, len(rows))

    out = Path(args.out or Path(args.manifest).parent / f"soft_{args.split}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    with out.open("w", encoding="utf-8") as fh:
        fh.write("path,prob\n")
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i : i + args.batch_size]
            frames = [read_image(r["path"]) for r in batch]
            tensor = torch.from_numpy(predictor.prepare_frames(frames)).to(predictor.device)
            with torch.inference_mode():
                probs = torch.sigmoid(predictor.model.forward_frames(tensor)).float().cpu().numpy()
            for row, prob in zip(batch, probs):
                fh.write(f"{row['path']},{float(prob):.6f}\n")
            if (i // args.batch_size) % 100 == 0:
                logger.info("  %d/%d（%.1fs）", i + len(batch), len(rows), time.perf_counter() - start)

    logger.info("软标签已写入: %s（用时 %.1fs）", out, time.perf_counter() - start)


if __name__ == "__main__":
    main()
