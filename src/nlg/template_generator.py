"""템플릿 기반 알람 문장 생성기(Jinja2). 실시간성이 중요해서 LLM 호출 없이
구조화된 이벤트를 바로 한국어 문장으로 렌더링한다. Gemini API/Ollama 버전은
별도 모듈로 추가 예정(NLG 3-way 비교)."""

from __future__ import annotations

from jinja2 import Template

from src.event_aggregator import EventType, SafetyEvent

_TEMPLATES: dict[EventType, Template] = {
    EventType.FALL_SUSPECTED: Template(
        "[경고] {{ track_id }}번 작업자 낙상 의심 (신뢰도 {{ '%.0f'|format(confidence*100) }}%)"
    ),
    EventType.PPE_MISSING: Template(
        "[주의] {{ track_id }}번 작업자 {{ detail_kr }} 미착용"
    ),
}

_ITEM_NAME_KR = {
    "helmet": "안전모",
    "vest": "안전조끼",
    "harness": "안전벨트",
    "safety_shoes": "안전화",
}


def generate_alarm_text(event: SafetyEvent) -> str:
    template = _TEMPLATES[event.event_type]
    return template.render(
        track_id=event.track_id,
        confidence=event.confidence,
        detail_kr=_ITEM_NAME_KR.get(event.detail, event.detail),
    )
