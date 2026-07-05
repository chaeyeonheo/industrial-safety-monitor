"""FrameResult 목록을 실제 초 단위 타임스탬프가 있는 이벤트 타임라인으로 변환.

VQA(Gemini)가 "낙상이 몇 초에 발생했나요?" 같은 질문에 답할 때 근거로 쓸
구조화된 데이터를 만드는 용도. frame_idx를 fps로 나눠 실제 경과 시간을 계산한다.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.pipeline import FrameResult


def build_event_timeline(results: list[FrameResult], fps: float) -> list[dict]:
    timeline = []
    for result in results:
        for event in result.alerts:
            timeline.append({
                "timestamp_sec": round(event.frame_idx / fps, 2),
                "frame_idx": event.frame_idx,
                "track_id": event.track_id,
                "event_type": event.event_type.value,
                "detail": event.detail,
                "confidence": round(event.confidence, 2),
            })
    return timeline


def save_event_timeline(results: list[FrameResult], fps: float, path: str | Path) -> None:
    timeline = build_event_timeline(results, fps)
    Path(path).write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
