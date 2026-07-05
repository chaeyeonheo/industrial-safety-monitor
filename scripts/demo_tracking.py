"""Phase 1 동작 확인용 데모.

1) 정지 이미지 탐지 sanity check (ultralytics 내장 bus.jpg, 사람 4명)
2) 비디오 소스가 주어지면 프레임별 트랙 ID 일관성을 오버레이해 mp4/png로 저장

AIHub 영상 다운로드가 끝나기 전까지는 (1)만 실행 가능하다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics.utils import ASSETS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detection_tracking.tracker import PersonTracker  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "results" / "figures"


def run_image_sanity_check() -> None:
    tracker = PersonTracker(conf_threshold=0.25)
    image_path = ASSETS / "bus.jpg"
    tracks = tracker.detect_image(image_path)

    print(f"[sanity-check] source={image_path}")
    print(f"[sanity-check] detected persons: {len(tracks)}")
    for t in tracks:
        print(f"  id={t.track_id} bbox={tuple(round(v, 1) for v in t.bbox)} conf={t.confidence:.3f}")

    img = cv2.imread(str(image_path))
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"person {t.confidence:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "tracking_demo_bus.png"
    cv2.imwrite(str(out_path), img)
    print(f"[sanity-check] saved overlay to {out_path}")


def run_video_tracking(source: str, max_frames: int | None = None) -> None:
    tracker = PersonTracker(conf_threshold=0.4)
    id_frame_ranges: dict[int, list[int]] = {}

    for frame_idx, tracks in enumerate(tracker.track_stream(source)):
        if max_frames is not None and frame_idx >= max_frames:
            break
        for t in tracks:
            id_frame_ranges.setdefault(t.track_id, []).append(frame_idx)

    print(f"[video] source={source}")
    print(f"[video] unique track ids observed: {len(id_frame_ranges)}")
    for track_id, frames in sorted(id_frame_ranges.items()):
        print(f"  track_id={track_id} first_frame={frames[0]} last_frame={frames[-1]} n_frames={len(frames)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=str, default=None, help="비디오 파일 경로. 생략 시 정지 이미지 sanity check만 실행")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    run_image_sanity_check()

    if args.source:
        run_video_tracking(args.source, max_frames=args.max_frames)
    else:
        print("[video] --source가 지정되지 않아 비디오 트래킹 검증은 건너뜁니다. "
              "AIHub 데이터 다운로드가 끝나면 실제 영상으로 재실행할 것.")
