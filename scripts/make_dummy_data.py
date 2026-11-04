"""生成合成冒烟数据（无数据集/无网络时可跑通全流程）。

用法：python scripts/make_dummy_data.py --out data/dummy
产物：
    data/dummy/frames/{real,fake}/video_xxxx/frame_yy.jpg
    data/dummy/videos/{real,fake}_xxxx.mp4
    data/dummy/manifest.csv   （path,label,video_id,split）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo_deepfake.data.dataset import write_manifest  # noqa: E402


def make_real(rng: np.random.Generator, size: int) -> np.ndarray:
    ys = np.linspace(0, 1, size)[:, None]
    xs = np.linspace(0, 1, size)[None, :]
    c1 = rng.uniform(70, 190, 3)
    c2 = rng.uniform(70, 190, 3)
    img = np.stack([c1[c] * (1 - xs) + c2[c] * xs for c in range(3)], axis=-1)
    img = np.ascontiguousarray(np.broadcast_to(img, (size, size, 3)), dtype=np.float32)
    cv2.ellipse(img, (size // 2, size // 2), (size // 4, size // 3), 0, 0, 360,
                tuple(float(v) for v in c1[::-1]), -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.6)
    img = img + rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_fake(rng: np.random.Generator, size: int) -> np.ndarray:
    left = make_real(rng, size)
    right = make_real(rng, size)
    img = left.copy()
    img[:, size // 2 :] = right[:, size // 2 :]  # 生硬拼接，模拟换脸边界
    grid = ((np.indices((size, size)).sum(axis=0) // 4) % 2).astype(np.float32)
    img = np.clip(img.astype(np.float32) + (grid[..., None] * 36.0 - 18.0), 0, 255)
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.3, 0, 255)  # 色偏
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    cv2.line(img, (size // 2, 0), (size // 2, size), (210, 70, 70), 1)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/dummy")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--videos-per-class", type=int, default=4)
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    rng = np.random.default_rng(args.seed)
    rows = []
    total = args.videos_per_class * 2
    for cls_name, label in (("real", 0), ("fake", 1)):
        for v in range(args.videos_per_class):
            vid = f"{cls_name}_{v:03d}"
            split = "train" if v < max(1, args.videos_per_class - 1) else "val"
            frame_dir = out / "frames" / cls_name / vid
            frame_dir.mkdir(parents=True, exist_ok=True)
            frames = []
            for f in range(args.frames_per_video):
                img = make_real(rng, args.size) if label == 0 else make_fake(rng, args.size)
                frames.append(img)
                path = frame_dir / f"frame_{f:02d}.jpg"
                cv2.imwrite(str(path), img)
                rows.append({"path": str(path), "label": label, "video_id": vid, "split": split})

            video_dir = out / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"{vid}.mp4"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (args.size, args.size)
            )
            for img in frames:
                writer.write(img)
            writer.release()

    manifest = write_manifest(rows, out / "manifest.csv")
    print(f"生成 {total} 个视频 / {len(rows)} 帧，清单: {manifest}")


if __name__ == "__main__":
    main()
