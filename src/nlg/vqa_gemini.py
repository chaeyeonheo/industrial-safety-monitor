"""Gemini API 기반 VQA(Visual Question Answering, 사실은 구조화된 이벤트
타임라인에 대한 자연어 QA). 실제 감지된 이벤트만 근거로 답하도록 강하게 제약한다.

API 키는 사용자가 직접 발급해서 환경변수 GEMINI_API_KEY로 설정해야 한다
(코드/문서에 키를 절대 기록하지 않음).
"""

from __future__ import annotations

import json

from google import genai

SYSTEM_PROMPT = """당신은 산업 현장 안전 모니터링 시스템의 어시스턴트입니다.
아래는 실제 영상 분석으로 감지된 이벤트 타임라인입니다(초 단위 타임스탬프, 인물
track_id, 이벤트 종류, 세부사항 포함).

규칙:
- 반드시 이 데이터에만 근거해서 답하세요. 타임라인에 없는 내용을 지어내지 마세요.
- 데이터에 없는 질문이면 "기록된 이벤트에서 확인할 수 없습니다"라고 답하세요.
- event_type이 "fall_suspected"면 낙상(넘어짐) 의심, "ppe_missing"이면 보호구
  미착용입니다. detail 필드에 어떤 보호구인지(helmet=안전모, vest=안전조끼,
  harness=안전벨트, safety_shoes=안전화) 나옵니다.
- 간결하고 명확하게 한국어로, 초 단위/track_id를 정확히 인용해서 답하세요.
"""


class SafetyVQA:
    def __init__(self, api_key: str | None = None, model: str = "gemini-flash-latest"):
        # api_key=None이면 GEMINI_API_KEY/GOOGLE_API_KEY 환경변수를 genai.Client()가
        # 자동으로 사용한다.
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model

    def ask(self, event_timeline: list[dict], question: str) -> str:
        context = json.dumps(event_timeline, ensure_ascii=False, indent=2)
        prompt = f"{SYSTEM_PROMPT}\n\n[이벤트 타임라인]\n{context}\n\n[질문]\n{question}"
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return response.text
