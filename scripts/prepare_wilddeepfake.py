"""WildDeepfake（HuggingFace: xingjunm/WildDeepfake）→ 帧级清单。

数据集以 webdataset 分片组织：
    deepfake_in_the_wild/{real,fake}_{train,test}/<n>.tar.gz
分片内结构：
    ./<n>/<label>/<seq>/<frame>.png
其中 <seq> 即一段"野生"人脸序列（可视作一个视频）。

相比 mlcroissant 的 parquet 视图（单分片 500MB、总计 9.9GB），直接按需拉取
tar.gz 分片更省流量、可断点续传，且保留原始 PNG 不二次压缩。

用法：
    # 拉 2 个训练分片/类 + 1 个测试分片/类，每序列最多 8 帧
    python scripts/prepare_wilddeepfake.py --out data/wild --train-shards 2 --test-shards 1 \
        --max-seqs 20 --max-frames 8
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List

BASE_URL = "https://huggingface.co/datasets/xingjunm/WildDeepfake/resolve/main/deepfake_in_the_wild"
IMG_SUFFIXES = (".png", ".jpg", ".jpeg")


def list_shards(label_name: str, hf_split: str, timeout: int = 60) -> List[int]:
    """通过 HF API 获取实际存在的分片编号（各 split 编号并不连续）。"""
    import json

    url = f"https://huggingface.co/api/datasets/xingjunm/WildDeepfake/tree/main/deepfake_in_the_wild/{label_name}_{hf_split}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = json.load(resp)
    ids = sorted(
        int(Path(item["path"]).name.split(".")[0])
        for item in payload
        if item["path"].endswith(".tar.gz")
    )
    if not ids:
        raise SystemExit(f"未找到分片: {label_name}_{hf_split}")
    return ids


def _readable_tar(path: Path) -> bool:
    """完整遍历一遍 tar，能走到底才算下载完整。"""
    try:
        with tarfile.open(path, "r:*") as tar:
            for _ in tar:
                pass
        return True
    except Exception:
        return False


def download(url: str, dest: Path, retries: int = 6, timeout: int = 120) -> Path:
    """支持断点续传的下载：代理常在中途断流，因此按已下载字节数续传。"""
    if dest.exists() and _readable_tar(dest):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载 {url}")
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                mode = "ab"
                if have and resp.status != 206:  # 服务端不支持续传，从头写
                    have, mode = 0, "wb"
                with dest.open(mode) as fh:
                    while chunk := resp.read(1 << 20):
                        fh.write(chunk)
        except Exception as exc:  # 网络中断，进入下一轮续传
            print(f"  中断({attempt}/{retries}): {type(exc).__name__}: {exc}")
        if _readable_tar(dest):
            return dest
        size = dest.stat().st_size if dest.exists() else 0
        print(f"  分片仍不完整（{size} 字节），续传重试")
    raise RuntimeError(f"分片下载失败: {url}")


def extract_shard(
    shard: Path,
    frames_root: Path,
    label: int,
    split: str,
    max_seqs: int,
    max_frames: int,
) -> List[Dict]:
    label_name = "real" if label == 0 else "fake"
    rows: List[Dict] = []
    # 注意：这些 .tar.gz 实际多为未压缩的 tar，用 r:* 自动识别压缩方式
    with tarfile.open(shard, "r:*") as tar:
        members = [
            m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(IMG_SUFFIXES)
        ]
        groups: Dict[str, List[tarfile.TarInfo]] = {}
        shard_ids: Dict[str, str] = {}
        for m in members:
            parts = m.name.strip("./").split("/")
            if len(parts) < 4:
                continue
            shard_ids[parts[2]] = parts[0]
            groups.setdefault(parts[2], []).append(m)

        seq_ids = sorted(groups) if max_seqs <= 0 else sorted(groups)[:max_seqs]
        for seq in seq_ids:
            picked = sorted(groups[seq], key=lambda x: x.name)[:max_frames]
            if not picked:
                continue
            video_id = f"{label_name}_{split}_s{shard_ids[seq]}_{seq}"
            seq_dir = frames_root / label_name / video_id
            seq_dir.mkdir(parents=True, exist_ok=True)
            for i, member in enumerate(picked):
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                suffix = Path(member.name).suffix.lower() or ".png"
                frame_path = seq_dir / f"frame_{i:02d}{suffix}"
                frame_path.write_bytes(handle.read())
                rows.append(
                    {"path": str(frame_path), "label": label, "video_id": video_id, "split": split}
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/wild")
    parser.add_argument("--cache", default=None, help="分片缓存目录，默认 <out>/_shards")
    parser.add_argument("--train-shards", type=int, default=2, help="每类训练分片数；<=0 表示全量")
    parser.add_argument("--test-shards", type=int, default=1, help="每类测试分片数（作为 val）；<=0 表示全量")
    parser.add_argument("--workers", type=int, default=8, help="并发下载线程数")
    parser.add_argument("--max-seqs", type=int, default=20, help="每个分片最多取多少条序列；<=0 表示全部")
    parser.add_argument("--max-frames", type=int, default=8, help="每条序列最多取多少帧")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from geo_deepfake.data.dataset import write_manifest

    out = Path(args.out)
    cache = Path(args.cache) if args.cache else out / "_shards"
    frames_root = out / "frames"
    rows: List[Dict] = []

    # 1) 先汇总所有需要的分片（num_shards <= 0 表示全量）
    tasks = []
    for hf_split, num_shards, our_split in (
        ("train", args.train_shards, "train"),
        ("test", args.test_shards, "val"),
    ):
        for label, label_name in ((0, "real"), (1, "fake")):
            available = list_shards(label_name, hf_split)
            picked = available if num_shards <= 0 else available[:num_shards]
            for idx in picked:
                tasks.append(
                    {
                        "label": label,
                        "label_name": label_name,
                        "split": our_split,
                        "idx": idx,
                        "url": f"{BASE_URL}/{label_name}_{hf_split}/{idx}.tar.gz",
                        "dest": cache / f"{label_name}_{hf_split}_{idx}.tar.gz",
                    }
                )
    print(f"待处理分片 {len(tasks)} 个（workers={args.workers}）")

    # 2) 并发下载（每个分片内部支持断点续传）
    from concurrent.futures import ThreadPoolExecutor, as_completed

    done, failed = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, t["url"], t["dest"]): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
            except Exception as exc:  # 单个分片失败不应中断整体
                failed.append(task["url"])
                print(f"  [跳过] {task['dest'].name}: {type(exc).__name__}")
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"  下载进度 {done}/{len(tasks)}（失败 {len(failed)}）")
    if failed:
        print(f"失败分片 {len(failed)} 个，可稍后重跑脚本断点续传")

    # 3) 串行解压成帧（IO 密集，避免并发抢占）
    for i, task in enumerate(tasks, 1):
        if not _readable_tar(task["dest"]):
            continue  # 下载失败的分片跳过
        new_rows = extract_shard(
            task["dest"], frames_root, task["label"], task["split"],
            args.max_seqs, args.max_frames,
        )
        rows.extend(new_rows)
        if i % 50 == 0 or i == len(tasks):
            print(f"  解压进度 {i}/{len(tasks)}，累计 {len(rows)} 帧")

    if not rows:
        raise SystemExit("未提取到任何帧，请检查分片编号")
    manifest = write_manifest(rows, out / "manifest.csv")
    stats: Dict[str, int] = {}
    for row in rows:
        stats[f"{row['split']}/{'fake' if row['label'] else 'real'}"] = (
            stats.get(f"{row['split']}/{'fake' if row['label'] else 'real'}", 0) + 1
        )
    print(f"清单: {manifest}，共 {len(rows)} 帧，分布 {stats}")


if __name__ == "__main__":
    main()
