"""AIHub 163 키포인트 라벨 전수 분석.

4개 카테고리(떨어짐/부딪힘/넘어짐/물체에맞음) train 라벨 zip 전체(23,840개 JSON 추정)를
압축 해제 없이 zip 내부에서 직접 읽어 path(원본 영상/시퀀스 ID)별로 그룹핑하고,
프레임 번호 간격 분포를 계산한다. scripts/convert_aihub163_keypoints_to_pyskl.py에서
슬라이딩 윈도우를 어떻게 구성할지 결정하기 위한 사전 분석.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

LABELS_DIR = Path(__file__).resolve().parents[1] / \
    "data/raw/ppe_construction_aihub163/keypoints/train/labels"

CATEGORIES = {
    "1.떨어짐.zip": "falling_from_height",
    "2.부딪힘.zip": "struck_by_collision",
    "3.넘어짐.zip": "trip_and_fall",
    "4.물체에_맞음.zip": "struck_by_object",
}

FRAME_NUM_RE = re.compile(r"(\d+)\.jpg$")


def analyze_category(zip_path: Path) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    n_files = 0

    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".json")]
        n_files = len(names)
        for name in names:
            with z.open(name) as f:
                d = json.load(f)
            path = d["image"]["path"]
            m = FRAME_NUM_RE.search(d["image"]["filename"])
            if m:
                groups[path].append(int(m.group(1)))

    group_sizes = sorted(len(v) for v in groups.values())
    all_gaps = []
    for frames in groups.values():
        frames.sort()
        all_gaps.extend(b - a for a, b in zip(frames, frames[1:]))
    all_gaps.sort()

    return {
        "n_files": n_files,
        "n_groups": len(groups),
        "group_size_min": group_sizes[0] if group_sizes else None,
        "group_size_median": group_sizes[len(group_sizes) // 2] if group_sizes else None,
        "group_size_max": group_sizes[-1] if group_sizes else None,
        "gap_min": all_gaps[0] if all_gaps else None,
        "gap_median": all_gaps[len(all_gaps) // 2] if all_gaps else None,
        "gap_max": all_gaps[-1] if all_gaps else None,
    }


def main() -> None:
    print(f"[analyze] labels dir: {LABELS_DIR}")
    total_files = 0
    for zip_name, label in CATEGORIES.items():
        zip_path = LABELS_DIR / zip_name
        if not zip_path.exists():
            print(f"[analyze] SKIP {zip_name} (파일 없음)")
            continue
        stats = analyze_category(zip_path)
        total_files += stats["n_files"]
        print(f"\n[{label}] ({zip_name})")
        print(f"  files={stats['n_files']}  video_groups={stats['n_groups']}")
        print(f"  group_size min/median/max = {stats['group_size_min']}/{stats['group_size_median']}/{stats['group_size_max']}")
        print(f"  frame_gap  min/median/max = {stats['gap_min']}/{stats['gap_median']}/{stats['gap_max']}")

    print(f"\n[analyze] total labeled frames across 4 categories: {total_files}")


if __name__ == "__main__":
    main()
