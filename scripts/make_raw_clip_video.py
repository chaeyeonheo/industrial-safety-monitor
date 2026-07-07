"""VLM 비교용 원본(오버레이 없는) 영상을 클립 폴더 안에 만든다. CV 파이프라인이
그린 demo_<fall_mode>.mp4와 나란히 비교할 수 있게 outputs/<클립>/raw.mp4로 저장.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"
sys.path.insert(0, str(REPO_ROOT))

from scripts.reencode_to_h264 import reencode  # noqa: E402


def imread_unicode(path: Path):
    """cv2.imread(str(path))는 Windows에서 경로에 non-ASCII(한글) 문자가 있으면
    내부적으로 ANSI 코드페이지 변환을 거치다 파일을 못 찾는 문제가 있다(이
    저장소 경로 자체가 '채연' 폴더를 포함해서 실측으로 재현됨). pathlib으로
    바이트를 직접 읽고 cv2.imdecode로 디코드하면 이 문제를 피할 수 있다."""
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_clip_frames(clip_dir: Path) -> list[Path]:
    info = json.loads((clip_dir / "clip_info.json").read_text(encoding="utf-8"))
    source_dir = Path(info["source_dir"])
    all_frames = sorted(source_dir.glob("*.jpg"))
    names = [p.name for p in all_frames]
    start = names.index(info["first_frame"])
    end = names.index(info["last_frame"])
    return all_frames[start:end + 1]


def make_raw_video(clip_name: str, fps: float = 5.0) -> None:
    clip_dir = OUTPUT_DIR / clip_name
    frames = load_clip_frames(clip_dir)
    first = imread_unicode(frames[0])
    h, w = first.shape[:2]
    compare_dir = clip_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    out_path = compare_dir / "raw.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fp in frames:
        writer.write(imread_unicode(fp))
    writer.release()
    # cv2.VideoWriter(fourcc='mp4v')는 실제로 FMP4 코덱이라 브라우저에서 재생이
    # 안 된다 — 웹앱에서 바로 재생 가능하도록 H.264로 재인코딩.
    reencode(out_path)
    print(f"저장됨: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=str, required=True)
    args = parser.parse_args()
    make_raw_video(args.clip)
