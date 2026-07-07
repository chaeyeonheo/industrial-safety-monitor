"""버전 2: 기존 CV 파이프라인(탐지+추적 → PPE 모델 → pose 모델 →
휴리스틱/HD-GCN → 이벤트 통합 → NLG, 여러 단계·여러 모델·학습 필요)과 대비되는
VLM 버전 — 프레임 몇 장을 Gemini에 한 번에 넣고 "낙상/보호구 미착용/통제구역
무단진입"을 바로 물어본다. 학습 데이터 준비나 모델 학습이 전혀 필요 없다는 게
핵심 비교 포인트.

통제구역은 CV 파이프라인(demo_full_pipeline.py)과 똑같은 위치(DEFAULT_ZONE_RATIO)에
노란 사각형을 프레임 위에 직접 그려서 Gemini에 전달한다 — VLM은 좌표 개념이
없으므로 "이 노란 박스가 통제구역"이라고 시각적으로 알려줘야 판단이 가능하다.

같은 데모 클립(outputs/<클립이름>/)에 대해 CV 파이프라인이 이미 만들어둔
event_timeline_*.json과 비교해서 (1) 판단 일치 여부, (2) 소요 시간을 함께
기록한다. API 키는 환경변수 GEMINI_API_KEY로만 받는다(코드/문서에 절대 기록 안 함).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.zone_intrusion.zone_intrusion import DEFAULT_ZONE_RATIO, zone_from_ratio  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"
MAX_FRAMES_TO_SEND = 12  # Gemini 요청 크기/비용을 적당히 유지하기 위해 균등 샘플링
ZONE_COLOR_BGR = (0, 210, 255)  # demo_full_pipeline.py의 COLOR_ZONE과 동일(선명한 노랑)

PROMPT = """당신은 산업 현장 CCTV 영상(프레임 몇 장, 시간 순서대로 제공됨)을 보고
안전 사고를 판단하는 안전 모니터링 시스템입니다. 각 프레임에는 노란 사각형으로
표시된 "통제구역"이 그려져 있습니다.

다음을 확인하세요:
1. 사람이 넘어지거나 떨어지는 등 낙상(사고)이 보이는가? 보인다면 서로 다른
   시점/다른 사람의 낙상을 구분해서 몇 번 발생했는지 세세요.
   **중요: 절대 추측하거나 지어내지 마세요.** 프레임에서 명확히 확인되는
   낙상만 세고, 애매하면(같은 동작이 이어지는 건지 별개 사건인지 불확실하면)
   더 적게(보수적으로) 세세요. 근거 없이 숫자를 부풀리면 안 됩니다.
2. 아래 4가지 보호구 각각에 대해 미착용이 명확히 보이는 사람이 있는지
   개별적으로 확인하세요: 안전모(헬멧), 안전조끼(형광 조끼), 안전벨트(하네스),
   안전화. 이것도 실제로 화면에서 확인한 것만 true로 표시하고, 안 보이거나
   판단이 안 서면 false로 하세요.
3. 노란 사각형(통제구역) **안에 사람이 실제로 서 있거나 들어가 있는** 프레임이
   있는지 확인하세요. 사람의 발/몸 대부분이 노란 박스 경계 안에 있어야 진입으로
   인정하고, 박스 밖에 있거나 경계에 살짝 걸친 정도는 진입으로 세지 마세요.
   여기서도 명확한 경우만 세고, 애매하면 진입으로 세지 마세요.

반드시 아래 JSON 형식으로만 답하세요(다른 텍스트 없이):
{
  "fall_detected": true/false,
  "fall_count": 0,
  "fall_evidence": "낙상이 보이면 각 발생마다 어느 프레임/어떤 모습인지 설명, 없으면 빈 문자열",
  "ppe_violations": {
    "helmet": {"violation": true/false, "evidence": "근거 또는 빈 문자열"},
    "vest": {"violation": true/false, "evidence": "근거 또는 빈 문자열"},
    "harness": {"violation": true/false, "evidence": "근거 또는 빈 문자열"},
    "safety_shoes": {"violation": true/false, "evidence": "근거 또는 빈 문자열"}
  },
  "zone_intrusion": {"detected": true/false, "evidence": "근거 또는 빈 문자열"},
  "scene_summary": "장면에 대한 1문장 요약"
}
"""


def imread_unicode(path: Path):
    """cv2.imread(str(path))는 Windows에서 경로에 non-ASCII(한글) 문자가 있으면
    ANSI 코드페이지 변환 문제로 파일을 못 찾는다(이 저장소 경로 자체가 '채연'
    폴더를 포함해서 실측 재현됨) — pathlib으로 바이트를 직접 읽어 우회."""
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def draw_zone(img, zone: tuple[float, float, float, float]) -> None:
    x1, y1, x2, y2 = (int(v) for v in zone)
    cv2.rectangle(img, (x1, y1), (x2, y2), ZONE_COLOR_BGR, 6)
    cv2.putText(img, "RESTRICTED ZONE", (x1 + 8, y1 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, ZONE_COLOR_BGR, 2, cv2.LINE_AA)


def load_clip_frames(clip_dir: Path) -> list[Path]:
    """clip_info.json의 source_dir/first_frame/last_frame으로 원본(오버레이 없는)
    프레임 경로 목록을 복원한다."""
    info = json.loads((clip_dir / "clip_info.json").read_text(encoding="utf-8"))
    source_dir = Path(info["source_dir"])
    all_frames = sorted(source_dir.glob("*.jpg"))
    names = [p.name for p in all_frames]
    start = names.index(info["first_frame"])
    end = names.index(info["last_frame"])
    return all_frames[start:end + 1]


def sample_frames(frames: list[Path], max_frames: int) -> list[Path]:
    if len(frames) <= max_frames:
        return frames
    step = len(frames) / max_frames
    return [frames[int(i * step)] for i in range(max_frames)]


def run_vlm_check(clip_dir: Path, model: str = "gemini-flash-latest") -> dict:
    frames = load_clip_frames(clip_dir)
    sampled = sample_frames(frames, MAX_FRAMES_TO_SEND)

    first = imread_unicode(sampled[0])
    h, w = first.shape[:2]
    zone = zone_from_ratio(DEFAULT_ZONE_RATIO, w, h)

    image_parts = []
    for p in sampled:
        img = imread_unicode(p)
        draw_zone(img, zone)
        ok, buf = cv2.imencode(".jpg", img)
        image_parts.append(types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"))

    parts = [*image_parts, types.Part.from_text(text=PROMPT)]

    client = genai.Client()
    t0 = time.time()
    response = client.models.generate_content(model=model, contents=parts)
    elapsed = time.time() - t0

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {"raw_response": text, "parse_error": True}

    return {
        "clip": clip_dir.name,
        "n_frames_total": len(frames),
        "n_frames_sent": len(sampled),
        "elapsed_sec": round(elapsed, 2),
        "vlm_result": parsed,
    }


def load_cv_pipeline_result(clip_dir: Path) -> dict:
    timeline_path = clip_dir / "event_timeline_keypoint_heuristic.json"
    if not timeline_path.exists():
        return {"available": False}
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    falls = [e for e in timeline if e["event_type"] == "fall_suspected"]
    ppe = [e for e in timeline if e["event_type"] == "ppe_missing"]
    zone = [e for e in timeline if e["event_type"] == "zone_intrusion"]
    return {
        "available": True,
        "fall_detected": len(falls) > 0,
        "n_fall_events": len(falls),
        "n_ppe_events": len(ppe),
        "zone_intrusion_detected": len(zone) > 0,
        "n_zone_events": len(zone),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=str, required=True, help="outputs/ 밑의 클립 폴더 이름")
    args = parser.parse_args()

    clip_dir = OUTPUT_DIR / args.clip
    if not clip_dir.exists():
        print(f"클립을 찾을 수 없습니다: {clip_dir}")
        return

    print(f"[vlm-check] {args.clip}: VLM 호출 중...")
    vlm_result = run_vlm_check(clip_dir)
    cv_result = load_cv_pipeline_result(clip_dir)
    comparison = {"vlm": vlm_result, "cv_pipeline": cv_result}

    print(json.dumps(comparison, ensure_ascii=False, indent=2))

    # CV 파이프라인 산출물(event_timeline_*.json, demo_*.mp4)과 섞이지 않게
    # 클립 폴더 밑에 compare/ 하위 폴더를 따로 둬서 저장.
    compare_dir = clip_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    out_path = compare_dir / "vlm_result.json"
    out_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[vlm-check] 저장됨: {out_path}")


if __name__ == "__main__":
    main()
