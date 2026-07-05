"""여러 시나리오(낙상, PPE 미착용)에서 나온 이벤트를 하나로 모아 중복 알람을
누른다. 사람 추적(track_id)이 핵심 키 — 같은 사람의 같은 이벤트가 반복되면
`cooldown_seconds` 동안 다시 알리지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EventType(Enum):
    FALL_SUSPECTED = "fall_suspected"      # Stage A만 트리거된 상태(미확인)
    PPE_MISSING = "ppe_missing"


@dataclass(frozen=True)
class SafetyEvent:
    track_id: int
    event_type: EventType
    frame_idx: int
    detail: str  # 예: "helmet" (PPE_MISSING일 때 어떤 항목인지)
    confidence: float


@dataclass
class _CooldownState:
    last_frame_idx: int


class EventAggregator:
    def __init__(self, cooldown_frames: int = 150) -> None:
        # pipeline.yaml의 cooldown_seconds * fps로 호출부에서 변환해 넘겨줄 것
        self.cooldown_frames = cooldown_frames
        self._last_alerted: dict[tuple[int, EventType, str], _CooldownState] = {}

    def submit(self, events: list[SafetyEvent]) -> list[SafetyEvent]:
        """이번 프레임의 후보 이벤트들 중, 쿨다운에 걸리지 않은 것만 반환(=실제 알람 대상)."""
        alerts = []
        for event in events:
            key = (event.track_id, event.event_type, event.detail)
            state = self._last_alerted.get(key)
            if state is not None and event.frame_idx - state.last_frame_idx < self.cooldown_frames:
                continue
            self._last_alerted[key] = _CooldownState(last_frame_idx=event.frame_idx)
            alerts.append(event)
        return alerts
