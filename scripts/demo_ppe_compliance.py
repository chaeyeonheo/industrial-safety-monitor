"""PPE 미착용 판정 데모: Phase 1(사람 탐지) + Phase 3(보호구 탐지) 결합.

사람 bbox와 보호구 bbox를 간접 연결(src/ppe_detection/indirect_association.py)로
매칭해서, 사람별로 어떤 보호구가 빠졌는지 실제 이미지에 오버레이한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detection_tracking.tracker import PersonTracker  # noqa: E402
from src.ppe_detection.indirect_association import (  # noqa: E402
    PPEDetection, check_ppe_compliance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PPE_WEIGHTS = REPO_ROOT / "outputs/ppe_yolo_runs/train/weights/best.pt"
VAL_IMAGES_DIR = REPO_ROOT / "data/processed/ppe_yolo/images/val"
FIGURES_DIR = REPO_ROOT / "results/figures"

MISSING_COLOR = (0, 0, 230)     # 빨강: 미착용
OK_COLOR = (80, 200, 80)        # 초록: 착용 확인
PERSON_COLOR = (230, 160, 40)   # 사람 bbox


def draw_status(img, status) -> None:
    x1, y1, x2, y2 = (int(v) for v in status.person_bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), PERSON_COLOR, 2)

    for det in status.detected_items.values():
        ix1, iy1, ix2, iy2 = (int(v) for v in det.bbox)
        cv2.rectangle(img, (ix1, iy1), (ix2, iy2), OK_COLOR, 2)

    lines = [f"{'OK' if not status.missing_items else 'MISSING'}"]
    for item in status.missing_items:
        lines.append(f"- {item} 미착용")

    y_text = max(20, y1 - 10 - 20 * (len(lines) - 1))
    color = OK_COLOR if not status.missing_items else MISSING_COLOR
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x1, y_text + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def main() -> None:
    if not PPE_WEIGHTS.exists():
        print(f"[ppe-compliance] PPE 가중치가 없습니다: {PPE_WEIGHTS}. "
              f"먼저 scripts/train_ppe_yolo.py를 실행하세요.")
        return

    person_tracker = PersonTracker(conf_threshold=0.3)
    ppe_model = YOLO(str(PPE_WEIGHTS))

    all_images = sorted(VAL_IMAGES_DIR.glob("*.jpg"))
    max_examples = 8
    n_saved = 0
    print(f"[ppe-compliance] val 이미지 {len(all_images)}개 중 사람이 탐지되는 예시 {max_examples}개를 찾는 중")

    for img_path in all_images:
        if n_saved >= max_examples:
            break
        person_tracks_raw = person_tracker.detect_image(img_path)
        if not person_tracks_raw:
            continue
        n_saved += 1

        ppe_results = ppe_model.predict(source=str(img_path), conf=0.3, verbose=False)[0]
        ppe_detections = []
        for box in ppe_results.boxes:
            cls_id = int(box.cls[0])
            class_name = ppe_results.names[cls_id]
            xyxy = tuple(box.xyxy[0].tolist())
            conf = float(box.conf[0])
            ppe_detections.append(PPEDetection(class_name=class_name, bbox=xyxy, confidence=conf))

        person_tracks = [(t.track_id, t.bbox) for t in person_tracks_raw]
        statuses = check_ppe_compliance(person_tracks, ppe_detections)

        img = cv2.imread(str(img_path))
        for status in statuses:
            draw_status(img, status)
            print(f"  {img_path.name} track={status.track_id} missing={status.missing_items or '없음(완전 착용)'}")

        out_path = FIGURES_DIR / f"ppe_compliance_{img_path.stem}.png"
        cv2.imwrite(str(out_path), img)

    print(f"[ppe-compliance] 결과를 {FIGURES_DIR}에 저장")


if __name__ == "__main__":
    main()
