"""FaceForensics++ c23 并发下载（官方 EU2 服务器，支持断点续传）。

说明：官方 `download_FaceForensics.py` 只支持单线程 + 需交互确认 TOS；本脚本复用同样的
URL 规则，改为多线程 + Range 续传 + 失败跳过，便于在集群/长链路上稳定拉取。

用法：
    python scripts/download_ffpp.py --root /home/ddl/data/ffpp --limit 800 --workers 12
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_SERVER = "https://kaldir.vc.in.tum.de/faceforensics/v3"
MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def fetch_pairs(base_url: str) -> list:
    url = f"{base_url}/misc/filelist.json"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def build_tasks(base_url: str, root: Path, methods: list, compression: str, limit: int) -> list:
    pairs = fetch_pairs(base_url)
    orig_ids = sorted({p for pair in pairs for p in pair})
    tasks = []

    def add(dataset_path: str, name: str, label: int):
        url = f"{base_url}/{dataset_path}/{compression}/videos/{name}.mp4"
        dest = root / dataset_path / compression / "videos" / f"{name}.mp4"
        tasks.append({"url": url, "dest": dest, "label": label})

    if "original" in methods:
        for vid in orig_ids[:limit] if limit else orig_ids:
            add("original_sequences/youtube", vid, 0)

    for method in methods:
        if method not in MANIPULATIONS:
            continue
        names = []
        for pair in pairs:
            names.append(f"{pair[0]}_{pair[1]}")
            names.append(f"{pair[1]}_{pair[0]}")
        for name in names[: limit * 2 if limit else None]:
            add(f"manipulated_sequences/{method}", name, 1)
    return tasks


def expected_size(url: str, timeout: int = 30) -> int:
    """HEAD 拿 Content-Length，用于校验本地文件是否被截断。"""
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return int(resp.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def download(task: dict, retries: int = 5, timeout: int = 120) -> bool:
    dest: Path = task["dest"]
    if dest.exists() and dest.stat().st_size > 10_000:
        size = dest.stat().st_size
        exp = expected_size(task["url"])
        if not exp or size == exp:
            return True
        dest.unlink()  # 长度不符：截断文件，重新下载
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            request = urllib.request.Request(task["url"], headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                mode = "ab"
                if have and resp.status != 206:
                    have, mode = 0, "wb"
                with dest.open(mode) as fh:
                    while chunk := resp.read(1 << 20):
                        fh.write(chunk)
            if dest.stat().st_size > 10_000:
                return True
        except Exception:
            continue
    dest.unlink(missing_ok=True)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/ddl/data/ffpp")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--methods", default="original,Deepfakes",
                        help="逗号分隔：original,Deepfakes,Face2Face,FaceSwap,NeuralTextures")
    parser.add_argument("--compression", default="c23", choices=["raw", "c23", "c40"])
    parser.add_argument("--limit", type=int, default=800, help="每类最多多少个源视频")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    root = Path(args.root)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    tasks = build_tasks(args.server, root, methods, args.compression, args.limit)
    print(f"待下载 {len(tasks)} 个视频 -> {root}")

    done, failed = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, t): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            if future.result():
                done += 1
            else:
                failed.append(task["url"])
            if (done + len(failed)) % 100 == 0:
                print(f"  进度 {done + len(failed)}/{len(tasks)}（成功 {done}，失败 {len(failed)}）")
    print(f"完成：成功 {done}，失败 {len(failed)}")


if __name__ == "__main__":
    main()
