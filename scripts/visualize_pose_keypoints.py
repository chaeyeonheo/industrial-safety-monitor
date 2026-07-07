"""추론 시 뽑는 keypoint 품질을 직접 눈으로 확인하기 위한 오버레이 영상 생성.

낙상 감지 recall 문제의 원인이 (a) 실시간 pose 추출기 자체의 keypoint 품질
문제인지, (b) COCO17->AIHub16 리매핑 문제인지, (c) 그 이후 로직 문제인지를
구분하려면 먼저 pose 추출기가 실제로 뭘 찍는지 봐야 한다. 이 스크립트는
AIHub16 리매핑 이전의 원본 COCO-17 keypoint를 프레임 위에 그대로 그린다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
KEYPOINT_SOURCE_DIR = REPO_ROOT / "data/raw/ppe_construction_aihub163/keypoints/val/source"
OUTPUT_DIR = REPO_ROOT / "outputs"

COCO_SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),          # 팔
    (5, 11), (6, 12), (11, 12),                        # 몸통
    (11, 13), (13, 15), (12, 14), (14, 16),            # 다리
    (0, 1), (0, 2), (1, 3), (2, 4),                    # 얼굴
]


def draw_pose(img, keypoints, min_conf: float = 0.3) -> None:
    for x, y, c in keypoints:
        if c < min_conf:
            continue
        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 255), -1, cv2.LINE_AA)
    for i, j in COCO_SKELETON:
        xi, yi, ci = keypoints[i]
        xj, yj, cj = keypoints[j]
        if ci < min_conf or cj < min_conf:
            continue
        cv2.line(img, (int(xi), int(yi)), (int(xj), int(yj)), (60, 220, 60), 2, cv2.LINE_AA)


def run(name: str, frame_dir: Path, weights: str, max_frames: int, device: str) -> None:
    frame_paths = sorted(frame_dir.glob("*.jpg"))[:max_frames]
    if not frame_paths:
        print(f"프레임 없음: {frame_dir}")
        return
    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]

    model = YOLO(weights)
    output_video = OUTPUT_DIR / f"pose_debug_{name}.mp4"
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))

    n_frames_with_person = 0
    for frame_path in frame_paths:
        img = cv2.imread(str(frame_path))
        result = model.predict(source=str(frame_path), conf=0.3, device=device, verbose=False)[0]
        if result.keypoints is not None and len(result.boxes) > 0:
            n_frames_with_person += 1
            boxes = result.boxes.xyxy.cpu().numpy()
            kps = result.keypoints.data.cpu().numpy()
            for box, kp in zip(boxes, kps):
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
                draw_pose(img, kp)
        writer.write(img)
    writer.release()
    print(f"[{name}] {len(frame_paths)}프레임 중 사람+keypoint 검출 {n_frames_with_person}프레임 "
          f"({n_frames_with_person/len(frame_paths):.0%}) -> {output_video}")


if __name__ == "__main__":
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="S2-N4601_fall")
    parser.add_argument("--weights", default=str(REPO_ROOT / "weights/yolo11n-pose.pt"))
    parser.add_argument("--max-frames", type=int, default=200)
    args = parser.parse_args()

    frame_dir_map = {
        "S2-N6001_trip": "_frames_S2N6001", "S2-N6301_trip": "_frames_S2N6301",
        "S2-N6401_trip": "_frames_S2N6401", "S2-N4601_fall": "_frames_S2N4601",
        "S2-N4701_fall": "_frames_S2N4701",
    }
    frame_dir = KEYPOINT_SOURCE_DIR / frame_dir_map[args.source]
    device = "0" if torch.cuda.is_available() else "cpu"
    run(args.source, frame_dir, args.weights, args.max_frames, device)
