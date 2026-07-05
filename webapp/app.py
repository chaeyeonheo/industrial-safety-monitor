"""안전 모니터링 VQA 데모 웹앱.

왼쪽에 오버레이된 데모 영상, 오른쪽에 그 영상의 실제 이벤트 타임라인을 근거로
Gemini가 답하는 채팅형 QA 패널을 보여준다. `scripts/demo_full_pipeline.py`가
만든 outputs/full_pipeline_demo_<source>.mp4 와 outputs/event_timeline_<source>.json을
그대로 사용한다 — 별도 재추론 없음(이미 계산된 결과에 대해서만 질문).

실행 전 `GEMINI_API_KEY` 환경변수를 설정해야 한다(사용자가 직접 발급).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.nlg.vqa_gemini import SafetyVQA  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "outputs"

app = Flask(__name__)
_vqa: SafetyVQA | None = None


def get_vqa() -> SafetyVQA:
    global _vqa
    if _vqa is None:
        _vqa = SafetyVQA()
    return _vqa


def discover_sources() -> list[str]:
    return sorted(
        p.stem.removeprefix("event_timeline_")
        for p in OUTPUT_DIR.glob("event_timeline_*.json")
    )


@app.route("/")
def index():
    return render_template("index.html", sources=discover_sources())


@app.route("/video/<source>")
def video(source: str):
    filename = f"full_pipeline_demo_{source}.mp4"
    if not (OUTPUT_DIR / filename).exists():
        return f"영상을 찾을 수 없습니다: {filename}", 404
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/api/ask", methods=["POST"])
def ask():
    payload = request.get_json(force=True)
    source = payload.get("source")
    question = (payload.get("question") or "").strip()
    if not source or not question:
        return jsonify({"error": "source와 question이 필요합니다"}), 400

    timeline_path = OUTPUT_DIR / f"event_timeline_{source}.json"
    if not timeline_path.exists():
        return jsonify({"error": f"이벤트 타임라인이 없습니다: {timeline_path.name}"}), 404

    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    try:
        answer = get_vqa().ask(timeline, question)
    except Exception as e:  # Gemini API 키 미설정 등
        return jsonify({"error": f"VQA 호출 실패: {e}"}), 500
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True, port=5050)
