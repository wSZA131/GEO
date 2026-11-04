"""ASVspoof 2019 → 音频清单（path,label,utt_id,split）。

数据集来源（Kaggle 公开，ODC-By 许可）：
    kagglehub.dataset_download("awsaf49/asvpoof-2019-dataset")

协议文件形如 `ASVspoof2019.LA.cm.train.trn.txt`，每行：
    SPEAKER  UTT_ID  SYSTEM  KEY(bonafide|spoof)

脚本按协议里的 utt_id 反查 flac（先对整棵树建索引），因此不依赖 Kaggle 镜像的具体目录层级。

用法：
    python scripts/prepare_asvspoof.py --subset LA --out data/asvspoof
    python scripts/prepare_asvspoof.py --subset LA --max-per-class 4000   # 快速子集
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

PROTOCOL_GLOB = "ASVspoof2019.{subset}.cm.{phase}.{suffix}.txt"
PHASES = {
    "train": ("train", "trn", "train"),
    "dev": ("dev", "trl", "val"),
    "eval": ("eval", "trl", "test"),
}


def default_root() -> Path | None:
    cache = Path.home() / ".cache" / "kagglehub" / "datasets" / "awsaf49" / "asvpoof-2019-dataset"
    if not cache.exists():
        return None
    dirs = sorted([p for p in cache.iterdir() if p.is_dir()])
    for cand in dirs + [cache]:
        if any(cand.rglob("ASVspoof2019.*.cm.*.txt")):
            return cand
    return dirs[0] if dirs else None


def find_protocols(root: Path, subset: str) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for phase, (name, suffix, _) in PHASES.items():
        pattern = PROTOCOL_GLOB.format(subset=subset, phase=name, suffix=suffix)
        matches = sorted(root.rglob(pattern))
        if matches:
            found[phase] = matches[0]
    return found


def parse_protocol(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            utt, key = parts[1], parts[-1]
            rows.append({"utt_id": utt, "label": 0 if key.lower() == "bonafide" else 1})
    return rows


def index_audio(root: Path, subset: str) -> Dict[str, Path]:
    """建立 utt_id → flac 路径索引。"""
    index: Dict[str, Path] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        if subset not in dirpath:
            continue
        for name in filenames:
            if name.lower().endswith((".flac", ".wav")):
                index[Path(name).stem] = Path(dirpath) / name
    return index


def subsample(rows: List[Dict], max_per_class: int, seed: int) -> List[Dict]:
    if max_per_class <= 0:
        return rows
    rng = random.Random(seed)
    by_label: Dict[int, List[Dict]] = {0: [], 1: []}
    for row in rows:
        by_label[row["label"]].append(row)
    out: List[Dict] = []
    for label, items in by_label.items():
        rng.shuffle(items)
        out.extend(items[:max_per_class])
    rng.shuffle(out)  # 打散类别顺序，避免按清单顺序分批时整批同类
    return out


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from geo_deepfake.data.audio_dataset import write_audio_manifest

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="数据集根目录，默认自动探测 kagglehub 缓存")
    parser.add_argument("--subset", default="LA", choices=["LA", "PA"])
    parser.add_argument("--out", default="data/asvspoof")
    parser.add_argument("--max-per-class", type=int, default=0, help="每个划分每类最多保留多少条")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root) if args.root else default_root()
    if root is None or not root.exists():
        raise SystemExit("未找到数据集根目录，请用 --root 指定")

    protocols = find_protocols(root, args.subset)
    if not protocols:
        raise SystemExit(f"在 {root} 下未找到 ASVspoof2019.{args.subset} 的协议文件")

    index = index_audio(root, args.subset)
    print(f"根目录: {root}\n音频文件索引: {len(index)} 条")

    rows: List[Dict] = []
    for phase, protocol in protocols.items():
        split = PHASES[phase][2]
        entries = parse_protocol(protocol)
        entries = subsample(entries, args.max_per_class, args.seed)
        hit, miss = 0, 0
        for entry in entries:
            path = index.get(entry["utt_id"])
            if path is None:
                miss += 1
                continue
            hit += 1
            rows.append(
                {"path": str(path), "label": entry["label"], "utt_id": entry["utt_id"], "split": split}
            )
        print(f"  {phase:5s} 协议 {len(entries)} 条 → 命中 {hit}，缺失 {miss}")

    if not rows:
        raise SystemExit("未匹配到任何音频文件，请检查 --root / --subset")
    manifest = write_audio_manifest(rows, Path(args.out) / "manifest.csv")
    stats: Dict[str, int] = {}
    for row in rows:
        key = f"{row['split']}/{'spoof' if row['label'] else 'bonafide'}"
        stats[key] = stats.get(key, 0) + 1
    print(f"清单: {manifest}，共 {len(rows)} 条，分布 {stats}")


if __name__ == "__main__":
    main()
