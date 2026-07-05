"""Stage A(FallHeuristicTrigger) 동작 확인.

Phase 1에서 이미 받아둔 AIHub 낙상 시퀀스 프레임(S2-N6001, 443프레임)에 대해
탐지+추적 결과를 Stage A 휴리스틱에 흘려서 실제로 어떤 트리거가 언제 발생하는지 확인한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detection_tracking.tracker import PersonTracker  # noqa: E402
from src.fall_detection.heuristic_trigger import FallHeuristicTrigger  # noqa: E402

FRAME_DIR = Path(__file__).resolve().parents[1] / \
    "data/raw/ppe_construction_aihub163/keypoints/val/source/_frames_S2N6001"


def main() -> None:
    frame_paths = sorted(FRAME_DIR.glob("*.jpg"))
    print(f"[fall-trigger] {len(frame_paths)} frames from {FRAME_DIR}")

    tracker = PersonTracker(conf_threshold=0.4)
    trigger = FallHeuristicTrigger(fps=5.0, frame_size=(1920, 1080))  # AIHub 라벨 메타데이터 기준 해상도

    all_events = []
    for frame_idx, tracks in enumerate(tracker.track_stream([str(p) for p in frame_paths])):
        events = trigger.update(frame_idx, tracks)
        all_events.extend(events)

    print(f"[fall-trigger] total trigger events: {len(all_events)}")
    for e in all_events:
        print(f"  frame={e.frame_idx} track_id={e.track_id} reason={e.reason.value} "
              f"bbox={tuple(round(v, 1) for v in e.last_known_bbox)} conf_hint={e.confidence_hint:.2f} "
              f"near_edge={e.near_frame_edge}")


if __name__ == "__main__":
    main()
