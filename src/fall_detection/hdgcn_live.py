"""실시간 HD-GCN 낙상 분류. track_id별로 최근 WINDOW(=30)프레임의 keypoint를
버퍼에 쌓고, 버퍼가 차면 학습된 HD-GCN으로 5-way(낙상 4종 + 정상) 분류한다.

주의: 학습 데이터(v2, data/processed/fall_keypoints_transition)는 AIHub 정답
keypoint였고, 여기서는 pose_extractor.py의 YOLO11n-pose + COCO17->AIHub16
근사 리매핑으로 나온 keypoint를 쓴다. 즉 학습/추론의 keypoint 출처가 달라
정확도가 오프라인 평가(81.8%)보다 낮게 나올 수 있다 — 실측으로 확인할 것.
"""

from __future__ import annotations

import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "HD-GCN"))

from model.HDGCN import Model  # noqa: E402
import src.fall_detection.hdgcn_graph  # noqa: E402,F401  (import_class가 getattr로 찾음)

WINDOW = 30
NUM_CLASSES = 5
CLASS_NAMES = ["falling_from_height", "struck_by_collision", "trip_and_fall",
               "struck_by_object", "normal"]
NORMAL_CLASS_INDEX = 4


class HDGCNLiveClassifier:
    def __init__(self, weights_path: str, frame_size: tuple[int, int], device: str = "cpu"):
        self.device = device
        self.frame_w, self.frame_h = frame_size
        self.buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=WINDOW))
        self.model = self._load_model(weights_path)

    def _load_model(self, weights_path: str):
        model = Model(
            num_class=NUM_CLASSES, num_point=16, num_person=1,
            graph="src.fall_detection.hdgcn_graph.Graph",
            graph_args={"labeling_mode": "spatial"},
            in_channels=3,
        ).to(self.device)
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model.eval()
        return model

    def update(self, track_id: int, keypoints_16x3: np.ndarray) -> tuple[str, float] | None:
        """이번 프레임의 keypoint를 버퍼에 추가하고, 버퍼가 꽉 찼으면 분류 결과를
        (클래스이름, confidence) 형태로 반환. 아직 안 찼으면 None."""
        buf = self.buffers[track_id]
        buf.append(keypoints_16x3)
        if len(buf) < WINDOW:
            return None

        arr = np.stack(buf).astype(np.float32)  # (30, 16, 3)
        xy = arr[..., :2].copy()
        xy[..., 0] /= max(self.frame_w, 1)
        xy[..., 1] /= max(self.frame_h, 1)
        conf = arr[..., 2]
        data = np.concatenate([xy, conf[..., None]], axis=-1)  # (30, 16, 3)
        data = data.transpose(2, 0, 1)[..., None]  # (3, 30, 16, 1)
        tensor = torch.from_numpy(data).float().unsqueeze(0).to(self.device)  # (1,3,30,16,1)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax().item())
        return CLASS_NAMES[pred_idx], float(probs[pred_idx].item())

    def drop_track(self, track_id: int) -> None:
        self.buffers.pop(track_id, None)
