"""사람 탐지 + 추적 공유 백본.

YOLO11(person class)로 탐지하고 Ultralytics에 내장된 ByteTrack으로 트랙 ID를 부여한다.
낙상 감지(fall_detection)와 PPE/구역침입(ppe_detection, zone_intrusion) 두 시나리오가
동일한 트랙 스트림을 입력으로 공유한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from ultralytics import YOLO

COCO_PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class Track:
    frame_idx: int
    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel 좌표)
    confidence: float

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class PersonTracker:
    """YOLO + ByteTrack 기반 사람 탐지·추적기."""

    DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "yolo11n.pt"

    def __init__(
        self,
        model_path: str | Path | None = None,
        tracker_cfg: str = "bytetrack.yaml",
        conf_threshold: float = 0.4,
        device: str | None = None,
    ) -> None:
        resolved = str(model_path) if model_path is not None else (
            str(self.DEFAULT_WEIGHTS) if self.DEFAULT_WEIGHTS.exists() else "yolo11n.pt"
        )
        self.model = YOLO(resolved)
        self.tracker_cfg = tracker_cfg
        self.conf_threshold = conf_threshold
        # 명시적으로 CUDA를 강제한다 — None을 넘기면 ultralytics가 "자동 선택"하지만
        # 드라이버 상태에 따라 조용히 CPU로 떨어질 수 있어 항상 확인 가능하게 함.
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "0"
        else:
            self.device = "cpu"
        print(f"[PersonTracker] using device={self.device} "
              f"({torch.cuda.get_device_name(0) if self.device not in ('cpu',) and torch.cuda.is_available() else 'CPU'})")

    def track_stream(self, source: str | Path | int | list[str]) -> Iterator[list[Track]]:
        """비디오 파일 / 카메라 인덱스 / 정렬된 이미지 경로 리스트(프레임 시퀀스)를 받아
        프레임별 Track 리스트를 yield한다. 리스트를 넘기면 그 순서를 프레임 순서로 취급한다."""
        if isinstance(source, list):
            resolved_source = [str(p) for p in source]
        elif isinstance(source, int):
            resolved_source = source
        else:
            resolved_source = str(source)
        results = self.model.track(
            source=resolved_source,
            classes=[COCO_PERSON_CLASS_ID],
            conf=self.conf_threshold,
            tracker=self.tracker_cfg,
            device=self.device,
            stream=True,
            persist=True,
            verbose=False,
        )
        for frame_idx, result in enumerate(results):
            yield self._to_tracks(frame_idx, result)

    def detect_image(self, image_path: str | Path) -> list[Track]:
        """단일 이미지에 대한 탐지 결과(트랙 ID 없이 detection만) 반환. frame_idx=0, track_id는 detection 순번."""
        results = self.model.predict(
            source=str(image_path),
            classes=[COCO_PERSON_CLASS_ID],
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        return self._to_tracks(0, results[0], fallback_sequential_id=True)

    @staticmethod
    def _to_tracks(frame_idx: int, result, fallback_sequential_id: bool = False) -> list[Track]:
        tracks: list[Track] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return tracks

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        elif fallback_sequential_id:
            ids = list(range(len(xyxy)))
        else:
            return tracks  # 트래커가 아직 ID를 배정하지 못한 프레임

        for box, track_id, conf in zip(xyxy, ids, confs):
            tracks.append(
                Track(
                    frame_idx=frame_idx,
                    track_id=int(track_id),
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=float(conf),
                )
            )
        return tracks
