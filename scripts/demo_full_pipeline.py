"""전체 통합 파이프라인 데모: 탐지+추적 → 낙상 Stage A → PPE 미착용 판정
→ 이벤트 통합 → NLG 알람까지 한 영상에 오버레이한다.

시각화는 절제된 팔레트(짙은 남색/앰버/레드 3색 위주)로 정리했다 — 트랙마다
다른 원색을 쓰는 대신, "정상=중립색, 주의=앰버, 경보=레드" 의미 기반 색상만
사용해서 실제 모니터링 화면처럼 보이게 했다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import SafetyMonitorPipeline  # noqa: E402
from src.nlg.template_generator import ITEM_NAME_KR  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PPE_WEIGHTS = REPO_ROOT / "outputs/ppe_yolo_runs/train/weights/best.pt"
FRAME_DIR = REPO_ROOT / "data/raw/ppe_construction_aihub163/keypoints/val/source/_frames_S2N6001"
OUTPUT_VIDEO = REPO_ROOT / "outputs" / "full_pipeline_demo.mp4"
FIGURES_DIR = REPO_ROOT / "results/figures"

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


def draw_frame(img, result) -> None:
    for status in result.ppe_statuses:
        missing = result.stable_missing.get(status.track_id, [])
        x1, y1, x2, y2 = (int(v) for v in status.person_bbox)
        box_color = COLOR_ALERT if missing else COLOR_NEUTRAL
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)

        # 이번 프레임에 실제로 매칭된 보호구는 개별 bbox로 표시(어떤 걸
        # 찾았는지 검증할 수 있게)
        for item_name, det in status.detected_items.items():
            ix1, iy1, ix2, iy2 = (int(v) for v in det.bbox)
            cv2.rectangle(img, (ix1, iy1), (ix2, iy2), COLOR_OK, 2)
            item_label = f"{ITEM_NAME_KR.get(item_name, item_name)} {det.confidence:.0%}"
            cv2.putText(img, item_label, (ix1, max(0, iy1 - 6)), FONT, 0.6, COLOR_OK, 2, cv2.LINE_AA)

        label = f"ID {status.track_id}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.9, 2)
        draw_rounded_panel(img, x1, y1 - th - 16, x1 + tw + 16, y1, COLOR_PANEL_BG, alpha=0.75)
        cv2.putText(img, label, (x1 + 8, y1 - 10), FONT, 0.9, (245, 245, 245), 2, cv2.LINE_AA)

        if missing:
            items_kr = [ITEM_NAME_KR.get(item, item) for item in missing]
            text = "미착용: " + ", ".join(items_kr)
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.75, 2)
            draw_rounded_panel(img, x1, y2 + 6, x1 + tw + 16, y2 + th + 22, COLOR_PANEL_BG, alpha=0.75)
            cv2.putText(img, text, (x1 + 8, y2 + th + 14), FONT, 0.75, COLOR_WARNING, 2, cv2.LINE_AA)

    if result.alarm_texts:
        h, w = img.shape[:2]
        panel_h = 40 + 34 * len(result.alarm_texts)
        draw_rounded_panel(img, 0, 0, w, panel_h, COLOR_PANEL_BG, alpha=0.85)
        cv2.putText(img, "SAFETY MONITOR", (16, 28), FONT, 0.85, (245, 245, 245), 2, cv2.LINE_AA)
        for i, text in enumerate(result.alarm_texts):
            color = COLOR_ALERT if "낙상" in text else COLOR_WARNING
            cv2.putText(img, text, (16, 62 + 34 * i), FONT, 0.8, color, 2, cv2.LINE_AA)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=200,
                         help="443프레임 전체는 반복 크래시 이력이 있어 기본 200으로 제한. "
                              "0을 주면 전체 사용")
    args = parser.parse_args()

    if not PPE_WEIGHTS.exists():
        print(f"[full-pipeline] PPE 가중치가 없습니다: {PPE_WEIGHTS}")
        return

    frame_paths = sorted(FRAME_DIR.glob("*.jpg"))
    if args.max_frames:
        frame_paths = frame_paths[:args.max_frames]
    first_img = cv2.imread(str(frame_paths[0]))
    h, w = first_img.shape[:2]

    pipeline = SafetyMonitorPipeline(
        ppe_weights=str(PPE_WEIGHTS), fps=5.0, frame_size=(w, h), cooldown_seconds=10.0,
        # Phase 1에서 실측했듯 이 영상은 track이 자주 끊긴다(443프레임 중 탐지 실패
        # 50.6%). 원래 기본값(track_loss_min_history_frames=10)은 10프레임 연속
        # 추적을 요구하는데 이 영상엔 그만큼 이어지는 track이 드물어 거의 트리거가
        # 안 됐다. 이 영상 특성에 맞춰 완화(다른 영상엔 오탐이 늘 수 있어 기본값이
        # 아니라 여기서만 조정).
        fall_trigger_kwargs=dict(
            aspect_ratio_delta_threshold=0.35,
            track_loss_min_history_frames=4,
            track_loss_grace_frames=1,
        ),
    )

    # 추적 모델 -> PPE 모델 순으로 완전히 분리 실행(둘을 동시에 GPU에 띄우지
    # 않음). 결과가 다 나온 뒤에는 GPU 없이 오버레이만 그린다.
    results = pipeline.run_offline([str(p) for p in frame_paths])

    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    saved_example = False
    total_alerts = 0
    for frame_idx, result in enumerate(results):
        img = cv2.imread(str(frame_paths[frame_idx]))
        total_alerts += len(result.alerts)
        for text in result.alarm_texts:
            print(f"  frame={frame_idx} {text}")

        draw_frame(img, result)
        writer.write(img)

        if result.alarm_texts and not saved_example:
            cv2.imwrite(str(FIGURES_DIR / "full_pipeline_demo_frame.png"), img)
            saved_example = True

    writer.release()
    print(f"[full-pipeline] 총 {len(frame_paths)}프레임 처리, 알람 {total_alerts}건 발생")
    print(f"[full-pipeline] 영상 저장: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
