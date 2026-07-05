"""Stage A: 경량 휴리스틱 낙상 후보 트리거.

ST-GCN/CTR-GCN 같은 무거운 포즈 분류기(Stage B)를 매 프레임 돌리는 대신, 사람
탐지+추적 결과(bbox 시계열)만으로 "낙상일 수도 있다"는 후보를 먼저 걸러낸다.
Stage B는 이 트리거가 발생한 track에 대해서만 실행해 연산량을 줄인다.

세 가지 독립적인 트리거 신호를 사용한다:

1. ASPECT_RATIO_SPIKE: bbox의 (width/height) 비율이 짧은 시간에 급격히 커짐.
   서 있는 사람은 세로로 길쭉해 비율이 작고(예: 0.3~0.5), 쓰러지면 가로로
   납작해져 비율이 커진다(예: 1.5 이상).
2. VERTICAL_VELOCITY_SPIKE: bbox 중심의 수직 하강 속도(px/s)가 임계값을 넘음.
3. TRACK_LOST: 일정 프레임 이상 안정적으로 추적되던 track이 갑자기 사라짐.

   2026-07-05 실측(scripts/demo_tracking.py, results/RESULTS.md 참고): AIHub
   낙상 시퀀스에서 사람이 안전매트에 엎드려 쓰러진 프레임을 YOLO11n이 전혀
   탐지하지 못하는 사례를 확인했다. 이 경우 bbox 자체가 없으므로 1번, 2번
   신호는 애초에 계산이 불가능하다. 따라서 "탐지 유실"을 세 번째 독립 신호로
   둔다. 이 트리거는 다른 둘과 달리 Stage B에 넘길 살아있는 bbox가 없으므로,
   호출부(pipeline.py)에서 마지막으로 확인된 위치에서 best-effort로 포즈
   추출을 시도하고, 그마저 실패하면 confidence_hint가 낮은 "낙상 의심(확인
   불가)" 이벤트를 바로 내보내는 방식으로 처리해야 한다(TRACK_LOST는 오탐이
   섞일 수 있음 — 예: 사람이 화면 밖으로 걸어나간 경우도 이 신호를 만족함).

   짧은 가려짐(occlusion)은 대부분 ByteTrack 자체의 lost-track 버퍼(기본 약
   30프레임)가 재매칭으로 흡수하므로 우리 Track 출력까지 도달하지 않는 경우가
   많다. 그 버퍼를 넘어서는 긴 가려짐과 "화면 밖으로 나감"은 구분이 필요한데,
   여기서는 간단히 마지막 bbox 중심이 프레임 가장자리 근처(`frame_edge_margin_ratio`)
   인지로 1차 필터링한다: 가장자리 근처에서 사라지면 화면 이탈 가능성이 높다고
   보고 confidence_hint를 낮춘다(완전히 버리지는 않음 — Stage B/사람 확인으로
   최종 판단). frame_size가 주어지지 않으면 이 필터는 건너뛴다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque

from src.detection_tracking.tracker import Track


class TriggerReason(Enum):
    ASPECT_RATIO_SPIKE = "aspect_ratio_spike"
    VERTICAL_VELOCITY_SPIKE = "vertical_velocity_spike"
    TRACK_LOST = "track_lost"


@dataclass(frozen=True)
class TriggerEvent:
    track_id: int
    frame_idx: int
    reason: TriggerReason
    last_known_bbox: tuple[float, float, float, float]
    confidence_hint: float  # Stage B 우선순위 참고용. TRACK_LOST는 확인 전이라 항상 중간값 고정
    near_frame_edge: bool = False  # True면 화면 이탈 가능성 높음(TRACK_LOST 오탐 후보)


@dataclass
class _TrackHistory:
    frames: Deque[int] = field(default_factory=deque)
    bboxes: Deque[tuple[float, float, float, float]] = field(default_factory=deque)
    consecutive_missing: int = 0
    already_triggered_track_loss: bool = False


class FallHeuristicTrigger:
    """프레임별 Track 리스트를 순서대로 update()에 넣으면 트리거 이벤트를 반환한다."""

    def __init__(
        self,
        fps: float = 30.0,
        window_frames: int = 15,
        aspect_ratio_delta_threshold: float = 0.5,
        vertical_velocity_threshold_px_per_s: float = 200.0,
        track_loss_min_history_frames: int = 10,
        track_loss_grace_frames: int = 2,
        frame_size: tuple[int, int] | None = None,  # (width, height), 있으면 가장자리 필터 적용
        frame_edge_margin_ratio: float = 0.08,
    ) -> None:
        self.fps = fps
        self.window_frames = window_frames
        self.aspect_ratio_delta_threshold = aspect_ratio_delta_threshold
        self.vertical_velocity_threshold = vertical_velocity_threshold_px_per_s
        self.track_loss_min_history_frames = track_loss_min_history_frames
        self.track_loss_grace_frames = track_loss_grace_frames
        self.frame_size = frame_size
        self.frame_edge_margin_ratio = frame_edge_margin_ratio
        self._history: dict[int, _TrackHistory] = {}

    def _is_near_frame_edge(self, bbox: tuple[float, float, float, float]) -> bool:
        if self.frame_size is None:
            return False
        width, height = self.frame_size
        margin_x = width * self.frame_edge_margin_ratio
        margin_y = height * self.frame_edge_margin_ratio
        x1, y1, x2, y2 = bbox
        return x1 <= margin_x or y1 <= margin_y or x2 >= width - margin_x or y2 >= height - margin_y

    def update(self, frame_idx: int, tracks: list[Track]) -> list[TriggerEvent]:
        events: list[TriggerEvent] = []
        seen_ids = set()

        for t in tracks:
            seen_ids.add(t.track_id)
            hist = self._history.setdefault(t.track_id, _TrackHistory())
            hist.consecutive_missing = 0
            hist.frames.append(frame_idx)
            hist.bboxes.append(t.bbox)
            while hist.frames and frame_idx - hist.frames[0] > self.window_frames:
                hist.frames.popleft()
                hist.bboxes.popleft()

            event = self._check_motion_triggers(t.track_id, frame_idx, hist)
            if event is not None:
                events.append(event)

        for track_id, hist in self._history.items():
            if track_id in seen_ids:
                continue
            hist.consecutive_missing += 1
            if (
                not hist.already_triggered_track_loss
                and len(hist.frames) >= self.track_loss_min_history_frames
                and hist.consecutive_missing == self.track_loss_grace_frames
            ):
                hist.already_triggered_track_loss = True
                last_bbox = hist.bboxes[-1]
                near_edge = self._is_near_frame_edge(last_bbox)
                events.append(TriggerEvent(
                    track_id=track_id,
                    frame_idx=frame_idx,
                    reason=TriggerReason.TRACK_LOST,
                    last_known_bbox=last_bbox,
                    confidence_hint=0.2 if near_edge else 0.5,
                    near_frame_edge=near_edge,
                ))

        return events

    def _check_motion_triggers(self, track_id: int, frame_idx: int, hist: _TrackHistory) -> TriggerEvent | None:
        if len(hist.frames) < 2:
            return None
        f0, f1 = hist.frames[0], hist.frames[-1]
        b0, b1 = hist.bboxes[0], hist.bboxes[-1]
        dt = (f1 - f0) / self.fps
        if dt <= 0:
            return None

        ratio0 = (b0[2] - b0[0]) / max(b0[3] - b0[1], 1e-6)
        ratio1 = (b1[2] - b1[0]) / max(b1[3] - b1[1], 1e-6)
        ratio_delta = ratio1 - ratio0
        if ratio_delta >= self.aspect_ratio_delta_threshold:
            return TriggerEvent(
                track_id, frame_idx, TriggerReason.ASPECT_RATIO_SPIKE, b1,
                confidence_hint=min(1.0, ratio_delta),
            )

        cy0 = (b0[1] + b0[3]) / 2
        cy1 = (b1[1] + b1[3]) / 2
        vertical_velocity = (cy1 - cy0) / dt  # 이미지 좌표계: 아래 방향이 +y
        if vertical_velocity >= self.vertical_velocity_threshold:
            return TriggerEvent(
                track_id, frame_idx, TriggerReason.VERTICAL_VELOCITY_SPIKE, b1,
                confidence_hint=min(1.0, vertical_velocity / 1000),
            )

        return None
