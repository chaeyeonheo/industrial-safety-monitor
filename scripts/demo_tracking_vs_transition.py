"""비교 데모: "탐지+추적만 썼을 때" vs "keypoint 전환 감지를 함께 썼을 때".

Phase 1에서 확인한 문제(쓰러진 자세를 YOLO11n이 탐지 못해 bbox가 사라짐)를,
AIHub 정답 keypoint 기반 전환 감지(scripts/convert_aihub163_keypoints_to_pyskl_transition.py와
동일한 로직)로 보완하면 어떻게 되는지 같은 영상으로 나란히 보여준다.

위쪽 절반: YOLO11n+ByteTrack bbox만 표시 (탐지 실패 시 빨간 경고 배너)
아래쪽 절반: 정답 keypoint + 종횡비(ratio) 그래프 + 전환 감지 시 초록 배너

주의: 아래쪽은 "우리 파이프라인이 실시간으로 pose를 뽑아 판단한 것"이 아니라
AIHub가 이미 라벨링해둔 정답 keypoint에 Stage A 로직을 적용한 것이다. 즉 이
데모는 "keypoint 신호가 있으면 bbox 유실 구간에서도 낙상을 잡아낼 수 있다"는
근거를 보여주는 것이지, 완성된 실시간 pose 추출기와의 비교는 아니다(아직
pose 추출기 미구현).
"""

from __future__ import annotations

import sys
import zipfile
import json
import re
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection_tracking.tracker import PersonTracker  # noqa: E402
from convert_aihub163_keypoints_to_pyskl_transition import (  # noqa: E402
    compute_ratio_series, find_transition_indices,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAME_DIR = REPO_ROOT / "data/raw/ppe_construction_aihub163/keypoints/val/source/_frames_S2N6001"
LABEL_ZIP = REPO_ROOT / "data/raw/ppe_construction_aihub163/keypoints/val/labels/3.넘어짐.zip"
OUTPUT_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = REPO_ROOT / "results/figures"

STAGE_A_WINDOW_FRAMES = 15
STAGE_A_RATIO_DELTA_THRESHOLD = 0.5
TRANSITION_HIGHLIGHT_MARGIN = 10  # 전환 프레임 앞뒤로 몇 프레임까지 "감지됨" 배너를 유지할지


def load_keypoints_for_frames(frame_paths: list[Path]) -> dict[str, np.ndarray]:
    wanted = {p.stem for p in frame_paths}
    result = {}
    with zipfile.ZipFile(LABEL_ZIP) as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            stem = name[:-5]
            if stem not in wanted:
                continue
            with z.open(name) as f:
                d = json.load(f)
            result[stem] = np.array(d["annotations"][0]["point"], dtype=np.float32)
    return result


def main() -> None:
    frame_paths = sorted(FRAME_DIR.glob("*.jpg"))
    print(f"[compare] {len(frame_paths)} frames")

    keypoints_by_stem = load_keypoints_for_frames(frame_paths)
    run = []
    for p in frame_paths:
        kp = keypoints_by_stem.get(p.stem)
        if kp is None:
            continue
        img = cv2.imread(str(p))
        run.append((p, kp, (img.shape[1], img.shape[0])))
    print(f"[compare] {len(run)} frames with matching keypoint labels")

    ratios = compute_ratio_series([(0, kp, res) for _, kp, res in run])
    transitions = find_transition_indices(ratios, STAGE_A_WINDOW_FRAMES, STAGE_A_RATIO_DELTA_THRESHOLD)
    print(f"[compare] transition frames detected at indices: {transitions}")

    transition_flag = np.zeros(len(run), dtype=bool)
    for t in transitions:
        lo = max(0, t - TRANSITION_HIGHLIGHT_MARGIN)
        hi = min(len(run), t + TRANSITION_HIGHLIGHT_MARGIN)
        transition_flag[lo:hi] = True

    tracker = PersonTracker(conf_threshold=0.4)
    frame_list = [str(p) for p, _, _ in run]
    bbox_by_frame_idx: dict[int, tuple] = {}
    for idx, tracks in enumerate(tracker.track_stream(frame_list)):
        if tracks:
            bbox_by_frame_idx[idx] = tracks[0].bbox

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    first_img = cv2.imread(str(run[0][0]))
    h, w = first_img.shape[:2]
    half_h = h // 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_combined = cv2.VideoWriter(str(OUTPUT_DIR / "tracking_vs_transition.mp4"), fourcc, 5, (w, h))
    writer_tracking_only = cv2.VideoWriter(str(OUTPUT_DIR / "tracking_only.mp4"), fourcc, 5, (w, h))
    writer_transition_only = cv2.VideoWriter(str(OUTPUT_DIR / "transition_only.mp4"), fourcc, 5, (w, h))

    saved_examples = {"miss_but_detected": False}

    for idx, (path, kp, _res) in enumerate(run):
        img = cv2.imread(str(path))

        # keypoint 자체로부터 bbox를 뽑아 위쪽(YOLO bbox)과 같은 방식(사각형)으로
        # 그려서 "박스 vs 점"이 아니라 "박스 vs 박스"로 직접 비교 가능하게 한다.
        visible = kp[kp[:, 2] > 0]
        kp_bbox = None
        if len(visible) > 0:
            kp_bbox = (visible[:, 0].min(), visible[:, 1].min(), visible[:, 0].max(), visible[:, 1].max())

        top = img.copy()
        bbox = bbox_by_frame_idx.get(idx)
        if bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            cv2.rectangle(top, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(top, "tracking(YOLO+ByteTrack): OK", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        else:
            cv2.putText(top, "tracking(YOLO+ByteTrack): DETECTION LOST", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        bottom = img.copy()
        for (x, y, v) in kp:
            if v == 0:
                continue
            cv2.circle(bottom, (int(x), int(y)), 5, (255, 200, 0), -1)
        if kp_bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in kp_bbox)
            cv2.rectangle(bottom, (x1, y1), (x2, y2), (255, 200, 0), 3)
        ratio_text = f"keypoint bbox ratio={ratios[idx]:.2f}"
        cv2.putText(bottom, ratio_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 3)
        if transition_flag[idx]:
            cv2.putText(bottom, "keypoint-based: FALL TRANSITION DETECTED", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        writer_tracking_only.write(top)
        writer_transition_only.write(bottom)

        top_small = cv2.resize(top, (w, half_h))
        bottom_small = cv2.resize(bottom, (w, h - half_h))
        combined = np.vstack([top_small, bottom_small])
        writer_combined.write(combined)

        if bbox is None and transition_flag[idx] and not saved_examples["miss_but_detected"]:
            cv2.imwrite(str(FIGURES_DIR / "compare_tracking_miss_transition_catches.png"), combined)
            saved_examples["miss_but_detected"] = True
            print(f"[compare] frame {idx}: tracking 실패했지만 전환감지는 성공 — 예시 저장")

    writer_combined.release()
    writer_tracking_only.release()
    writer_transition_only.release()
    print(f"[compare] saved: {OUTPUT_DIR / 'tracking_vs_transition.mp4'} (합본), "
          f"{OUTPUT_DIR / 'tracking_only.mp4'}, {OUTPUT_DIR / 'transition_only.mp4'}")

    n_miss = sum(1 for i in range(len(run)) if i not in bbox_by_frame_idx)
    n_miss_but_flagged = sum(1 for i in range(len(run)) if i not in bbox_by_frame_idx and transition_flag[i])
    print(f"[compare] 전체 {len(run)}프레임 중 tracking 실패 {n_miss}건, "
          f"그중 keypoint 전환감지가 대신 잡아낸 프레임 {n_miss_but_flagged}건")


if __name__ == "__main__":
    main()
