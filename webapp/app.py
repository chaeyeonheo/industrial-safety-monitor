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
    """outputs/<클립이름>/event_timeline_<fall_mode>.json 계층 구조를 스캔해서
    "<클립이름>::<fall_mode>" 형태의 식별자 목록을 만든다(demo_full_pipeline.py가
    이 구조로 저장함 — 클립 하나에 fall_mode별로 여러 산출물이 들어있음).

    VLM vs CV 비교 데모용 웹앱이라, compare/vlm_result.json이 있는(=VLM 비교까지
    끝난) 클립만 노출한다 — 나머지 수십 개 클립은 비교 데이터가 없어 드롭다운만
    어지럽히므로 제외."""
    sources = []
    for clip_dir in sorted(OUTPUT_DIR.iterdir()):
        if not clip_dir.is_dir():
            continue
        if not (clip_dir / "compare" / "vlm_result.json").exists():
            continue
        for timeline_path in sorted(clip_dir.glob("event_timeline_*.json")):
            fall_mode = timeline_path.stem.removeprefix("event_timeline_")
            sources.append(f"{clip_dir.name}::{fall_mode}")
    return sources


def _resolve_source(source: str) -> tuple[Path, str]:
    """"<클립이름>::<fall_mode>" 식별자를 (클립 폴더, fall_mode)로 분해."""
    clip_name, _, fall_mode = source.partition("::")
    return OUTPUT_DIR / clip_name, fall_mode


def load_timeline_from_payload(payload: dict) -> list[dict] | None:
    timeline = payload.get("timeline")
    if isinstance(timeline, list):
        return timeline
    return None


@app.route("/")
def index():
    return render_template("index.html", sources=discover_sources())


@app.route("/video/<path:source>")
def video(source: str):
    clip_dir, fall_mode = _resolve_source(source)
    filename = f"demo_{fall_mode}.mp4"
    if not (clip_dir / filename).exists():
        return f"영상을 찾을 수 없습니다: {clip_dir.name}/{filename}", 404
    return send_from_directory(clip_dir, filename)


@app.route("/video_raw/<path:source>")
def video_raw(source: str):
    """VLM 비교용 원본(오버레이 없는) 영상 — scripts/make_raw_clip_video.py가
    outputs/<클립>/compare/raw.mp4로 만들어둔 것을 그대로 서빙."""
    clip_dir, _ = _resolve_source(source)
    compare_dir = clip_dir / "compare"
    if not (compare_dir / "raw.mp4").exists():
        return "원본 비교 영상이 없습니다", 404
    return send_from_directory(compare_dir, "raw.mp4")


@app.route("/api/vlm/<path:source>")
def vlm_result(source: str):
    """scripts/vlm_safety_check.py가 outputs/<클립>/compare/vlm_result.json으로
    저장해둔 VLM vs CV 파이프라인 비교 결과."""
    clip_dir, _ = _resolve_source(source)
    result_path = clip_dir / "compare" / "vlm_result.json"
    if not result_path.exists():
        return jsonify({"available": False}), 404
    data = json.loads(result_path.read_text(encoding="utf-8"))
    data["available"] = True
    return jsonify(data)


@app.route("/api/ask", methods=["POST"])
def ask():
    payload = request.get_json(force=True)
    source = payload.get("source")
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question이 필요합니다"}), 400

    timeline = load_timeline_from_payload(payload)
    if timeline is None:
        if not source:
            return jsonify({"error": "source 또는 timeline이 필요합니다"}), 400

        clip_dir, fall_mode = _resolve_source(source)
        timeline_path = clip_dir / f"event_timeline_{fall_mode}.json"
        if not timeline_path.exists():
            return jsonify({"error": (
                f"이벤트 타임라인이 없습니다: {clip_dir.name}/{timeline_path.name}. "
                "데모 결과를 먼저 생성하거나 mp4와 json을 함께 드래그앤드랍하세요."
            )}), 404
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))

    try:
        answer = get_vqa().ask(timeline, question)
    except Exception as e:  # Gemini API 키 미설정 등
        return jsonify({"error": f"VQA 호출 실패: {e}"}), 500
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True, port=5050)
