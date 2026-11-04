"""从公开数据集（FaceForensics++ / Celeb-DF / DFDC）构建帧级清单。

支持的目录布局：
    ff++     <root>/<compression>/<method>/<video_id>.mp4   （method=original 视为真实）
    celebdf  <root>/Celeb-real/*.mp4、<root>/YouTube-real/*.mp4 为真实，<root>/Celeb-synthesis/*.mp4 为伪造
    auto     <root>/real/** 与 <root>/fake/**（视频或图片均可）
    frames   <root>/real/** 与 <root>/fake/** 已是帧图片，仅登记不抽帧

用法：
    python scripts/build_manifest.py --root /data/FaceForensics++/c23 --layout ff++ \
        --out data/ffpp/manifest.csv --frames-per-video 8
    python scripts/build_manifest.py --root /data/Celeb-DF-v2 --layout celebdf --out data/celebdf/manifest.csv

注意：抽帧会占用磁盘，默认写入 <out 所在目录>/frames_cache。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo_deepfake.data.dataset import write_manifest  # noqa: E402
from geo_deepfake.data.preprocess import read_video_frames  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

FFPP_FAKE_METHODS = {"Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter", "DeepFakeDetection"}


def split_of(video_id: str) -> str:
    bucket = int(hashlib.md5(video_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "val"
    return "test"


def collect(root: Path, layout: str) -> List[tuple]:
    """返回 [(path, label, video_id)]。"""
    items: List[tuple] = []
    if layout == "ff++":
        for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            label = 0 if method_dir.name.lower() == "original" else 1
            if label == 1 and method_dir.name not in FFPP_FAKE_METHODS:
                continue
            for video in sorted(method_dir.rglob("*")):
                if video.suffix.lower() in VIDEO_SUFFIXES:
                    items.append((video, label, f"{method_dir.name}_{video.stem}"))
    elif layout == "ff++v2":
        # 官方 download_FaceForensics.py 的目录结构：
        #   original_sequences/youtube/c23/videos/<id>.mp4
        #   manipulated_sequences/<Method>/c23/videos/<src>_<tgt>.mp4
        for video in sorted(root.rglob("*")):
            if video.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            parts = video.parts
            if "original_sequences" in parts:
                items.append((video, 0, f"original_{video.stem}"))
            elif "manipulated_sequences" in parts:
                idx = parts.index("manipulated_sequences")
                method = parts[idx + 1] if idx + 1 < len(parts) else "manip"
                items.append((video, 1, f"{method}_{video.stem}"))
    elif layout == "celebdf":
        for sub in ("Celeb-real", "YouTube-real"):
            base = root / sub
            if base.exists():
                for video in sorted(base.rglob("*")):
                    if video.suffix.lower() in VIDEO_SUFFIXES:
                        items.append((video, 0, f"{sub}_{video.stem}"))
        base = root / "Celeb-synthesis"
        if base.exists():
            for video in sorted(base.rglob("*")):
                if video.suffix.lower() in VIDEO_SUFFIXES:
                    items.append((video, 1, f"Synthesis_{video.stem}"))
    elif layout in ("auto", "frames"):
        for cls, label in (("real", 0), ("fake", 1)):
            base = root / cls
            if not base.exists():
                raise SystemExit(f"目录不存在: {base}")
            for f in sorted(base.rglob("*")):
                if f.suffix.lower() in (VIDEO_SUFFIXES | IMAGE_SUFFIXES):
                    # frames 布局按父目录聚合成"视频"，保证视频级评测不被打散
                    vid = f"{cls}_{f.parent.name}" if layout == "frames" else f"{cls}_{f.stem}"
                    items.append((f, label, vid))
    else:
        raise SystemExit(f"未知布局: {layout}")
    if not items:
        raise SystemExit(f"在 {root} 下未找到任何样本，请检查 --layout")
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--layout", default="ff++", choices=["ff++", "ff++v2", "celebdf", "auto", "frames"])
    parser.add_argument("--out", default="data/manifest.csv")
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少个视频（调试用）")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    items = collect(root, args.layout)
    if args.limit:
        items = items[: args.limit]

    rows: List[Dict] = []
    if args.layout == "frames":
        for path, label, vid in items:
            rows.append({"path": str(path), "label": label, "video_id": vid, "split": split_of(vid)})
    else:
        cache = Path(args.cache_dir) if args.cache_dir else out.parent / "frames_cache"
        skipped = 0
        for idx, (path, label, vid) in enumerate(items, 1):
            try:
                frames = read_video_frames(path, args.frames_per_video)
            except Exception as exc:  # 单个视频损坏不应中断整体抽帧
                print(f"[跳过] {path}: {type(exc).__name__}")
                skipped += 1
                continue
            if not frames:
                print(f"[跳过] 无法读取: {path}")
                skipped += 1
                continue
            frame_dir = cache / vid
            frame_dir.mkdir(parents=True, exist_ok=True)
            split = split_of(vid)
            for i, frame in enumerate(frames):
                frame_path = frame_dir / f"frame_{i:02d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                rows.append({"path": str(frame_path), "label": label, "video_id": vid, "split": split})
            if idx % 50 == 0:
                print(f"已处理 {idx}/{len(items)} 个视频")

    manifest = write_manifest(rows, out)
    stats: Dict[str, int] = {}
    for row in rows:
        stats[f"{row['split']}/{row['label']}"] = stats.get(f"{row['split']}/{row['label']}", 0) + 1
    print(f"清单已生成: {manifest}，共 {len(rows)} 帧，分布: {stats}")


if __name__ == "__main__":
    main()
