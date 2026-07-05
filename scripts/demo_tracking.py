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


TRACK_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 0, 255),
    (0, 255, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
]


def _color_for(track_id: int) -> tuple[int, int, int]:
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def run_video_tracking(source: str | Path, max_frames: int | None = None,
                        save_overlay_name: str | None = None) -> None:
    """source가 디렉토리면 그 안의 이미지들을 파일명 정렬 순서로 프레임 시퀀스 취급."""
    src_path = Path(source)
    if src_path.is_dir():
        frame_paths = sorted(src_path.glob("*.jpg"))
        stream_input: str | list[str] = [str(p) for p in frame_paths]
        frame_lookup = frame_paths
    else:
        stream_input = str(source)
        frame_lookup = None

    tracker = PersonTracker(conf_threshold=0.4)
    id_frame_ranges: dict[int, list[int]] = {}

    writer = None
    if save_overlay_name and frame_lookup is not None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        first_img = cv2.imread(str(frame_lookup[0]))
        h, w = first_img.shape[:2]
        writer = cv2.VideoWriter(
            str(FIGURES_DIR / save_overlay_name),
            cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h),
        )

    frame_idx = -1
    for frame_idx, tracks in enumerate(tracker.track_stream(stream_input)):
        if max_frames is not None and frame_idx >= max_frames:
            break
        for t in tracks:
            id_frame_ranges.setdefault(t.track_id, []).append(frame_idx)

        if writer is not None:
            img = cv2.imread(str(frame_lookup[frame_idx]))
            for t in tracks:
                x1, y1, x2, y2 = (int(v) for v in t.bbox)
                color = _color_for(t.track_id)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"id {t.track_id}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            writer.write(img)

    if writer is not None:
        writer.release()
        print(f"[video] overlay saved to {FIGURES_DIR / save_overlay_name}")

    print(f"[video] source={source}")
    print(f"[video] total frames processed: {frame_idx + 1}")
    print(f"[video] unique track ids observed: {len(id_frame_ranges)}")
    for track_id, frames in sorted(id_frame_ranges.items()):
        print(f"  track_id={track_id} first_frame={frames[0]} last_frame={frames[-1]} n_frames={len(frames)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=str, default=None,
                         help="비디오 파일 경로 또는 정렬 가능한 프레임 이미지 디렉토리. 생략 시 정지 이미지 sanity check만 실행")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-overlay", type=str, default="tracking_demo_video.mp4")
    args = parser.parse_args()

    run_image_sanity_check()

    if args.source:
        run_video_tracking(args.source, max_frames=args.max_frames, save_overlay_name=args.save_overlay)
    else:
        print("[video] --source가 지정되지 않아 비디오 트래킹 검증은 건너뜁니다. "
              "AIHub 데이터 다운로드가 끝나면 실제 영상으로 재실행할 것.")
