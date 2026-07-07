"""실시간 pose 추출기: COCO-17 keypoint를 뽑고, HD-GCN이 학습된 AIHub 16-keypoint
근사 레이아웃(docs/keypoint_mapping.md)으로 리매핑한다.

두 백엔드를 제공한다:
- YoloPoseExtractor: YOLO11n-pose(Ultralytics). 가볍고 빠르지만 실측 결과
  사람이 완전히 쓰러진/누운 자세에서 keypoint를 거의 못 뽑는다(프레임 위에
  직접 그려서 확인함, docs/ablation_studies.md 참고). 원인은 COCO 학습 데이터
  자체가 서 있는/걷는 사람 위주라 눕거나 뒤집힌 자세가 거의 없기 때문으로 보임.
- RTMPoseExtractor: rtmlib(RTMPose, OpenMMLab) 기반. 같은 프레임에서 쓰러진
  사람을 실제로 더 잘 잡는 것을 실측으로 확인(스크린샷 outputs/rtmpose_*_frame_*.png,
  YOLO11n-pose는 완전히 놓친 쓰러진 사람을 RTMPose balanced 모드는 검출).
  기본 백엔드로 채택.

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


class YoloPoseExtractor:
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
            kps = result.keypoints.data.cpu().numpy()  # 원본 COCO17이면 (N,17,3), AIHub163으로
            for box, kp in zip(boxes, kps):              # fine-tuning한 모델이면 이미 (N,16,3)
                # fine-tuning된 모델은 AIHub163 GT keypoint(16점) 레이아웃을 그대로
                # 예측하도록 학습했으므로 리매핑이 필요 없다 — COCO17(17점) 원본
                # 모델일 때만 리매핑한다.
                remapped = kp if kp.shape[0] == 16 else remap_coco17_to_aihub16(kp)
                detections.append((tuple(box.tolist()), remapped))
        return detections


class RTMPoseExtractor:
    """rtmlib(RTMPose) 기반 pose 추출기. 내부적으로 Body가 쓰는 det_model(YOLOX)
    + pose_model(RTMPose)을 직접 호출해서 keypoint뿐 아니라 bbox도 얻는다
    (Body.__call__은 keypoint/score만 반환하고 bbox는 버림 — track_id 매칭에
    bbox가 필요해서 직접 접근)."""

    def __init__(self, mode: str = "balanced", device: str = "cpu"):
        from rtmlib import Body
        # GPU(onnxruntime CUDA EP)는 이 환경에서 cuDNN8 의존 dll(zlibwapi.dll)이
        # 없어 로드에 실패해 CPU로 자동 폴백된다(확인됨). 'balanced' 모드는
        # CPU에서도 289ms/프레임으로 쓸만하고, 'performance'와 동일하게 쓰러진
        # 사람을 검출했다(실측 비교, outputs/rtmpose_balanced_frame_171.png).
        self.body = Body(mode=mode, backend="onnxruntime", device=device)

    def extract(self, frame_path: str) -> list[tuple[tuple[float, float, float, float], np.ndarray]]:
        import cv2
        img = cv2.imread(frame_path)
        bboxes = self.body.det_model(img)
        keypoints, scores = self.body.pose_model(img, bboxes=bboxes)
        detections = []
        for bbox, kp, score in zip(bboxes, keypoints, scores):
            coco_kp = np.concatenate([kp, score[:, None]], axis=-1)  # (17, 3)
            remapped = remap_coco17_to_aihub16(coco_kp)
            detections.append((tuple(float(v) for v in bbox), remapped))
        return detections
