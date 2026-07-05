"""실시간 pose 추출기: YOLO11n-pose(COCO-17)로 keypoint를 뽑고, HD-GCN이
학습된 AIHub 16-keypoint 근사 레이아웃(docs/keypoint_mapping.md)으로 리매핑한다.

COCO-17: nose, leye, reye, lear, rear, lshoulder, rshoulder, lelbow, relbow,
         lwrist, rwrist, lhip, rhip, lknee, rknee, lankle, rankle
AIHub16(추론 매핑, 확정 아님): nose, neck, spine, rshoulder, lshoulder,
         relbow, lelbow, rwrist, lwrist, pelvis, rhip, lhip, rknee, lknee,
         rankle, lankle

neck/spine/pelvis는 COCO-17에 없는 점이라 좌우 평균으로 근사한다. 이 리매핑
자체가 이중으로 best-effort다(1: keypoint_mapping.md의 AIHub 16점 추정 자체가
2프레임만 검증한 추정치, 2: COCO-17 -> 그 추정 레이아웃으로의 대응도 근사).
HD-GCN 라이브 분류 결과는 이 이중 근사 위에서 나온다는 걸 감안해서 해석할 것.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

# AIHub16 인덱스 순서대로, 이걸 만드는 데 필요한 COCO-17 인덱스(단일 또는 평균할 쌍)
_NOSE, _LEYE, _REYE, _LEAR, _REAR = 0, 1, 2, 3, 4
_LSHOULDER, _RSHOULDER = 5, 6
_LELBOW, _RELBOW = 7, 8
_LWRIST, _RWRIST = 9, 10
_LHIP, _RHIP = 11, 12
_LKNEE, _RKNEE = 13, 14
_LANKLE, _RANKLE = 15, 16


def _single(coco_kp: np.ndarray, idx: int) -> np.ndarray:
    return coco_kp[idx]


def _mid(coco_kp: np.ndarray, i: int, j: int) -> np.ndarray:
    a, b = coco_kp[i], coco_kp[j]
    return np.array([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, min(a[2], b[2])])


def remap_coco17_to_aihub16(coco_kp: np.ndarray) -> np.ndarray:
    """coco_kp: (17, 3) [x, y, conf] -> (16, 3) AIHub16 근사 레이아웃."""
    neck = _mid(coco_kp, _LSHOULDER, _RSHOULDER)
    pelvis = _mid(coco_kp, _LHIP, _RHIP)
    spine = np.array([(neck[0] + pelvis[0]) / 2, (neck[1] + pelvis[1]) / 2, min(neck[2], pelvis[2])])

    return np.stack([
        _single(coco_kp, _NOSE),          # 0 nose
        neck,                              # 1 neck
        spine,                             # 2 spine
        _single(coco_kp, _RSHOULDER),      # 3 shoulder R
        _single(coco_kp, _LSHOULDER),      # 4 shoulder L
        _single(coco_kp, _RELBOW),         # 5 elbow R
        _single(coco_kp, _LELBOW),         # 6 elbow L
        _single(coco_kp, _RWRIST),         # 7 wrist R
        _single(coco_kp, _LWRIST),         # 8 wrist L
        pelvis,                            # 9 pelvis center
        _single(coco_kp, _RHIP),           # 10 hip R
        _single(coco_kp, _LHIP),           # 11 hip L
        _single(coco_kp, _RKNEE),          # 12 knee R
        _single(coco_kp, _LKNEE),          # 13 knee L
        _single(coco_kp, _RANKLE),         # 14 ankle R
        _single(coco_kp, _LANKLE),         # 15 ankle L
    ])


class PoseExtractor:
    def __init__(self, weights: str = "yolo11n-pose.pt", conf_threshold: float = 0.4,
                 device: str = "cpu"):
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold
        self.device = device

    def extract(self, frame_path: str) -> list[tuple[tuple[float, float, float, float], np.ndarray]]:
        """반환: [(person_bbox, aihub16_keypoints), ...]. bbox는 그대로 매칭용으로 반환
        (track_id와의 연결은 호출부에서 IoU로 매칭)."""
        result = self.model.predict(
            source=frame_path, conf=self.conf_threshold, device=self.device, verbose=False)[0]
        detections = []
        if result.keypoints is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            kps = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
            for box, kp in zip(boxes, kps):
                remapped = remap_coco17_to_aihub16(kp)
                detections.append((tuple(box.tolist()), remapped))
        return detections
