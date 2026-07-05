"""전체 통합 파이프라인 데모: 탐지+추적 → 낙상 Stage A → PPE 미착용 판정
→ 이벤트 통합 → NLG 알람까지 한 영상에 오버레이한다.

시각화는 절제된 팔레트(짙은 남색/앰버/레드 3색 위주)로 정리했다. 레이아웃:
- 사람 bbox + 매칭된 보호구 bbox: 프레임 위에 그대로(위치 확인용)
- 화면 하단: 현재 보이는 모든 사람의 미착용 현황을 상시 표시(깜빡이지 않음,
  PPE 판정은 track당 한 번만 확정되므로 프레임마다 안 바뀜)
- 낙상 의심은 별도로 크고 눈에 띄게(화면 중앙 상단 배너), 발생 후 잠시 유지
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import SafetyMonitorPipeline  # noqa: E402
from src.nlg.template_generator import ITEM_NAME_KR  # noqa: E402
from src.event_timeline import save_event_timeline  # noqa: E402

DEMO_FPS = 5.0

REPO_ROOT = Path(__file__).resolve().parents[1]
PPE_WEIGHTS = REPO_ROOT / "outputs/ppe_yolo_runs/train/weights/best.pt"
KEYPOINT_SOURCE_DIR = REPO_ROOT / "data/raw/ppe_construction_aihub163/keypoints/val/source"
FIGURES_DIR = REPO_ROOT / "results/figures"
OUTPUT_DIR = REPO_ROOT / "outputs"

# 여러 데모 샘플(사용자 요청, 다양한 샘플로 확장) — 부딪힘/물체에맞음은 아직
# 다운로드 안 해서 제외, 대신 이미 받아둔 떨어짐/넘어짐 안에서 서로 다른 영상
# 그룹 5개 사용.
DEMO_SOURCES = {
    "S2-N6001_trip": KEYPOINT_SOURCE_DIR / "_frames_S2N6001",   # 넘어짐
    "S2-N6301_trip": KEYPOINT_SOURCE_DIR / "_frames_S2N6301",   # 넘어짐(다른 그룹)
    "S2-N6401_trip": KEYPOINT_SOURCE_DIR / "_frames_S2N6401",   # 넘어짐(다른 그룹)
    "S2-N4601_fall": KEYPOINT_SOURCE_DIR / "_frames_S2N4601",   # 떨어짐
    "S2-N4701_fall": KEYPOINT_SOURCE_DIR / "_frames_S2N4701",   # 떨어짐(다른 그룹)
}

FALL_BANNER_PERSIST_FRAMES = 15  # 낙상 배너를 최소 이만큼의 프레임 동안 유지(너무 빨리 사라지지 않게)

# 절제된 시맨틱 팔레트 (BGR)
COLOR_NEUTRAL = (210, 200, 190)   # 옅은 슬레이트 그레이 — 사람 bbox 기본선
COLOR_OK = (150, 190, 110)        # 차분한 세이지 그린 — 보호구 착용 확인
COLOR_WARNING = (60, 170, 235)    # 앰버 — 보호구 미착용
COLOR_ALERT = (70, 70, 220)       # 톤 다운된 레드 — 낙상 의심
COLOR_PANEL_BG = (35, 30, 28)     # 알람 패널 배경(반투명 처리)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_rounded_panel(img, x1, y1, x2, y2, color, alpha=0.72):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def draw_frame(img, result, active_fall_banners: dict[int, int]) -> None:
    h, w = img.shape[:2]

    # 1) 사람 bbox + 매칭된 보호구 bbox
    for status in result.ppe_statuses:
        missing = result.track_missing_items.get(status.track_id, [])
        x1, y1, x2, y2 = (int(v) for v in status.person_bbox)
        box_color = COLOR_ALERT if missing else COLOR_NEUTRAL
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)

        for item_name, det in status.detected_items.items():
            ix1, iy1, ix2, iy2 = (int(v) for v in det.bbox)
            cv2.rectangle(img, (ix1, iy1), (ix2, iy2), COLOR_OK, 2)
            item_label = f"{ITEM_NAME_KR.get(item_name, item_name)} {det.confidence:.0%}"
            cv2.putText(img, item_label, (ix1, max(0, iy1 - 8)), FONT, 0.75, COLOR_OK, 2, cv2.LINE_AA)

        label = f"ID {status.track_id}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 1.05, 2)
        draw_rounded_panel(img, x1, y1 - th - 18, x1 + tw + 18, y1, COLOR_PANEL_BG, alpha=0.75)
        cv2.putText(img, label, (x1 + 9, y1 - 11), FONT, 1.05, (245, 245, 245), 2, cv2.LINE_AA)

    # 2) 하단 고정 패널: 현재 보이는 사람들의 미착용 현황(상시 표시, 깜빡임 없음).
    # 화면 전체 폭을 덮는 바 대신, 줄마다 텍스트 길이에 딱 맞는 작은 박스로
    # 좌측 하단에 쌓는다(화면을 너무 많이 가리지 않게).
    visible_ids = [s.track_id for s in result.ppe_statuses]
    lines = []
    for track_id in visible_ids:
        missing = result.track_missing_items.get(track_id, [])
        if missing:
            items_kr = [ITEM_NAME_KR.get(item, item) for item in missing]
            lines.append(f"ID {track_id}  미착용: {', '.join(items_kr)}")

    line_h = 42
    margin = 14
    font_scale = 0.85
    for i, text in enumerate(reversed(lines)):
        (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, 2)
        y_bottom = h - margin - i * (line_h + 8)
        y_top = y_bottom - line_h
        draw_rounded_panel(img, margin, y_top, margin + tw + 24, y_bottom, COLOR_PANEL_BG, alpha=0.82)
        cv2.putText(img, text, (margin + 12, y_bottom - 12), FONT, font_scale, COLOR_WARNING, 2, cv2.LINE_AA)

    # 3) 낙상 배너: 이번 프레임에 새로 발생했으면 등록, 발생 안 했어도 최근
    # 발생분은 잠시 더 유지(너무 빨리 사라지지 않게)
    for event in result.fall_events:
        active_fall_banners[event.track_id] = FALL_BANNER_PERSIST_FRAMES
    expired = [tid for tid, remaining in active_fall_banners.items() if remaining <= 0]
    for tid in expired:
        del active_fall_banners[tid]
    for tid in active_fall_banners:
        active_fall_banners[tid] -= 1

    if active_fall_banners:
        banner_h = 84
        draw_rounded_panel(img, 0, 0, w, banner_h, COLOR_ALERT, alpha=0.90)
        text = "낙상 발생: " + ", ".join(f"ID {tid}번" for tid in active_fall_banners)
        (tw, th), _ = cv2.getTextSize(text, FONT, 1.4, 4)
        cv2.putText(img, text, ((w - tw) // 2, banner_h // 2 + th // 2),
                    FONT, 1.4, (255, 255, 255), 4, cv2.LINE_AA)


def run_one(name: str, frame_dir: Path, max_frames: int) -> None:
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    if max_frames:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        print(f"[full-pipeline] {name}: 프레임을 찾을 수 없습니다({frame_dir})")
        return
    first_img = cv2.imread(str(frame_paths[0]))
    h, w = first_img.shape[:2]

    print(f"\n=== [{name}] {len(frame_paths)}프레임 ===")
    pipeline = SafetyMonitorPipeline(
        ppe_weights=str(PPE_WEIGHTS), fps=DEMO_FPS, frame_size=(w, h), cooldown_seconds=10.0,
        ppe_decision_window_frames=6,
    )

    # 추적 모델 -> PPE 모델 순으로 완전히 분리 실행(둘을 동시에 GPU에 띄우지
    # 않음). 결과가 다 나온 뒤에는 GPU 없이 오버레이만 그린다.
    results = pipeline.run_offline([str(p) for p in frame_paths])

    # VQA 웹앱이 쓸 이벤트 타임라인(초 단위 타임스탬프 포함) JSON 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_event_timeline(results, DEMO_FPS, OUTPUT_DIR / f"event_timeline_{name}.json")

    output_video = OUTPUT_DIR / f"full_pipeline_demo_{name}.mp4"
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_example = False
    saved_fall_example = False
    total_alerts = 0
    active_fall_banners: dict[int, int] = {}
    for frame_idx, result in enumerate(results):
        img = cv2.imread(str(frame_paths[frame_idx]))
        total_alerts += len(result.alerts)
        for text in result.alarm_texts:
            print(f"  frame={frame_idx} {text}")

        draw_frame(img, result, active_fall_banners)
        writer.write(img)

        if result.alarm_texts and not saved_example:
            cv2.imwrite(str(FIGURES_DIR / f"full_pipeline_demo_{name}_frame.png"), img)
            saved_example = True
        if result.fall_events and not saved_fall_example:
            cv2.imwrite(str(FIGURES_DIR / f"full_pipeline_demo_{name}_fall_frame.png"), img)
            saved_fall_example = True

    writer.release()
    print(f"[full-pipeline] [{name}] 총 {len(frame_paths)}프레임 처리, 알람 {total_alerts}건 발생")
    print(f"[full-pipeline] [{name}] 영상 저장: {output_video}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=200,
                         help="443프레임 전체는 반복 크래시 이력이 있어 기본 200으로 제한. "
                              "0을 주면 전체 사용")
    parser.add_argument("--source", type=str, default=None, choices=list(DEMO_SOURCES),
                         help="생략하면 DEMO_SOURCES 전체(여러 샘플)를 순차 실행")
    args = parser.parse_args()

    if not PPE_WEIGHTS.exists():
        print(f"[full-pipeline] PPE 가중치가 없습니다: {PPE_WEIGHTS}")
        return

    sources = {args.source: DEMO_SOURCES[args.source]} if args.source else DEMO_SOURCES
    for name, frame_dir in sources.items():
        run_one(name, frame_dir, args.max_frames)


if __name__ == "__main__":
    main()
