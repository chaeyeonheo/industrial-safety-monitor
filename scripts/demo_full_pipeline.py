"""전체 통합 파이프라인 데모: 탐지+추적 → 낙상(3가지 fall_mode 중 택1) → PPE
미착용 판정 → 이벤트 통합 → NLG 알람까지 한 영상에 오버레이한다.

AIHub163 keypoint 라벨 폴더 하나(예: _frames_S2N6001)를 통째로 "영상 1개"로
취급하지 않는다 — 프레임 번호 간격이 크게 벌어지는 지점은 실제로 다른 촬영
컷/불연속일 가능성이 커서, `src.frame_clips.discover_clips()`로 간격이 큰
지점마다 별도 클립으로 쪼갠다(짧은 클립이 나와도 그대로 둔다). 클립마다
`outputs/<클립이름>/` 폴더를 만들고 그 클립에 대한 모든 산출물(영상, 이벤트
타임라인, 대표 프레임)을 그 폴더 안에 계층적으로 저장한다 — 예전처럼
outputs/ 바로 밑에 이름을 이어붙여 평평하게 저장하지 않는다.

시각화는 절제된 팔레트(짙은 남색/앰버/레드 3색 위주)로 정리했다. 레이아웃:
- 사람 bbox + 매칭된 보호구 bbox: 프레임 위에 그대로(위치 확인용)
- 화면 하단: 현재 보이는 모든 사람의 미착용 현황을 상시 표시(깜빡이지 않음,
  PPE 판정은 track당 한 번만 확정되므로 프레임마다 안 바뀜)
- 낙상 의심은 별도로 크고 눈에 띄게(화면 중앙 상단 배너), 발생 후 잠시 유지
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import SafetyMonitorPipeline  # noqa: E402
from src.nlg.template_generator import ITEM_NAME_KR  # noqa: E402
from src.event_timeline import save_event_timeline  # noqa: E402
from src.frame_clips import discover_clips  # noqa: E402
from src.zone_intrusion.zone_intrusion import DEFAULT_ZONE_RATIO, zone_from_ratio  # noqa: E402
from scripts.reencode_to_h264 import reencode  # noqa: E402

DEMO_FPS = 5.0

REPO_ROOT = Path(__file__).resolve().parents[1]
PPE_WEIGHTS = REPO_ROOT / "outputs/ppe_yolo_runs/train/weights/best.pt"
POSE_WEIGHTS = REPO_ROOT / "weights/yolo11n-pose.pt"
HDGCN_WEIGHTS = REPO_ROOT / "outputs/hdgcn_runs/hdgcn_fall_v2.pt"
KEYPOINT_SOURCE_DIR = REPO_ROOT / "data/raw/ppe_construction_aihub163/keypoints/val/source"
OUTPUT_DIR = REPO_ROOT / "outputs"

# 여러 데모 소스(원본 AIHub 프레임 폴더). 각 폴더는 내부적으로 여러 클립으로
# 쪼개진다(discover_clips) — 부딪힘/물체에맞음은 아직 다운로드 안 해서 제외,
# 대신 이미 받아둔 떨어짐/넘어짐 안에서 서로 다른 영상 그룹 5개 사용.
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
COLOR_ZONE = (0, 210, 255)        # 선명한 노랑 — 통제구역 표시(빨강보다 눈에 잘 띔)
COLOR_PANEL_BG = (35, 30, 28)     # 알람 패널 배경(반투명 처리)
COLOR_KEYPOINT = (0, 255, 255)    # 노란색 — pose keypoint(keypoint_heuristic/hdgcn 모드일 때만)
COLOR_SKELETON = (60, 220, 60)    # 초록색 — keypoint 연결선
FONT = cv2.FONT_HERSHEY_SIMPLEX

# AIHub16 관절 순서(src/fall_detection/pose_extractor.py 참고): 0 nose, 1 neck,
# 2 spine, 3 shoulder R, 4 shoulder L, 5 elbow R, 6 elbow L, 7 wrist R,
# 8 wrist L, 9 pelvis, 10 hip R, 11 hip L, 12 knee R, 13 knee L, 14 ankle R,
# 15 ankle L
AIHUB16_SKELETON = [
    (0, 1), (1, 2), (2, 9),
    (1, 3), (3, 5), (5, 7),
    (1, 4), (4, 6), (6, 8),
    (9, 10), (10, 12), (12, 14),
    (9, 11), (11, 13), (13, 15),
]


def draw_rounded_panel(img, x1, y1, x2, y2, color, alpha=0.72):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def draw_keypoints(img, keypoints, min_conf: float = 0.3) -> None:
    """AIHub16 keypoint(16,3)를 스켈레톤과 함께 그린다."""
    for x, y, c in keypoints:
        if c >= min_conf:
            cv2.circle(img, (int(x), int(y)), 4, COLOR_KEYPOINT, -1, cv2.LINE_AA)
    for i, j in AIHUB16_SKELETON:
        xi, yi, ci = keypoints[i]
        xj, yj, cj = keypoints[j]
        if ci >= min_conf and cj >= min_conf:
            cv2.line(img, (int(xi), int(yi)), (int(xj), int(yj)), COLOR_SKELETON, 2, cv2.LINE_AA)


def draw_frame(img, result, active_fall_banners: dict[int, int],
               active_zone_banners: dict[int, int], zone: tuple[float, float, float, float]) -> None:
    h, w = img.shape[:2]

    # 0) 통제구역: 항상 굵은 노란 박스로 표시(진입 여부와 무관하게 상시 표시)
    zx1, zy1, zx2, zy2 = (int(v) for v in zone)
    overlay = img.copy()
    cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), COLOR_ZONE, -1)
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, dst=img)
    cv2.rectangle(img, (zx1, zy1), (zx2, zy2), COLOR_ZONE, 6)
    cv2.putText(img, "통제구역", (zx1 + 8, zy1 + 32), FONT, 0.9, COLOR_ZONE, 2, cv2.LINE_AA)

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

        keypoints = result.keypoints.get(status.track_id)
        if keypoints is not None:
            draw_keypoints(img, keypoints)

        label = f"ID {status.track_id}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 1.05, 2)
        draw_rounded_panel(img, x1, y1 - th - 18, x1 + tw + 18, y1, COLOR_PANEL_BG, alpha=0.75)
        cv2.putText(img, label, (x1 + 9, y1 - 11), FONT, 1.05, (245, 245, 245), 2, cv2.LINE_AA)

    # 2) 하단 고정 패널: 현재 보이는 사람들의 미착용 현황(상시 표시, 깜빡임 없음).
    # 화면 전체 폭을 덮는 바 대신, 줄마다 텍스트 길이에 딱 맞는 작은 박스로
    # 좌측 하단에 쌓는다(화면을 너무 많이 가리지 않게).
    # 구역진입 배너(전체 폭, 화면 맨 아래)가 뜰 예정이면 그만큼 위로 띄워서 안 겹치게 함
    # — 배너 상태 갱신을 먼저 해서 이번 프레임에 배너가 뜰지 미리 안다.
    for event in result.zone_events:
        active_zone_banners[event.track_id] = FALL_BANNER_PERSIST_FRAMES
    zone_banner_h = 70 if active_zone_banners else 0

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
        y_bottom = h - margin - zone_banner_h - i * (line_h + 8)
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

    # 4) 구역진입 배너 그리기(상태 갱신은 위 2번에서 이미 함 — PPE 패널이
    # 배너 높이만큼 띄워질지 미리 알아야 해서 순서를 앞당김)
    expired_zone = [tid for tid, remaining in active_zone_banners.items() if remaining <= 0]
    for tid in expired_zone:
        del active_zone_banners[tid]
    for tid in active_zone_banners:
        active_zone_banners[tid] -= 1

    if active_zone_banners:
        banner_h = 70
        y1, y2 = h - banner_h, h
        draw_rounded_panel(img, 0, y1, w, y2, COLOR_ZONE, alpha=0.90)
        text = "통제구역 진입: " + ", ".join(f"ID {tid}번" for tid in active_zone_banners)
        (tw, th), _ = cv2.getTextSize(text, FONT, 1.2, 3)
        # 노란 배경엔 흰 글씨가 잘 안 보여서 짙은 남색으로(대비 확보)
        cv2.putText(img, text, ((w - tw) // 2, y1 + banner_h // 2 + th // 2),
                    FONT, 1.2, (30, 20, 20), 3, cv2.LINE_AA)


def run_one_clip(clip_name: str, frame_paths: list[Path], clip_dir: Path, fall_mode: str) -> None:
    first_img = cv2.imread(str(frame_paths[0]))
    h, w = first_img.shape[:2]
    clip_dir.mkdir(parents=True, exist_ok=True)

    zone = zone_from_ratio(DEFAULT_ZONE_RATIO, w, h)

    print(f"\n=== [{clip_name}] {len(frame_paths)}프레임 (fall_mode={fall_mode}) ===")
    pipeline = SafetyMonitorPipeline(
        ppe_weights=str(PPE_WEIGHTS), fps=DEMO_FPS, frame_size=(w, h), cooldown_seconds=10.0,
        ppe_decision_window_frames=6, fall_mode=fall_mode,
        pose_weights=str(POSE_WEIGHTS), hdgcn_weights=str(HDGCN_WEIGHTS),
        zone=zone,
    )

    # 추적 모델 -> PPE 모델 -> (필요시) pose 모델 순으로 완전히 분리 실행(모델을
    # 동시에 GPU에 띄우지 않음). 결과가 다 나온 뒤에는 GPU 없이 오버레이만 그린다.
    results = pipeline.run_offline([str(p) for p in frame_paths])

    save_event_timeline(results, DEMO_FPS, clip_dir / f"event_timeline_{fall_mode}.json")

    output_video = clip_dir / f"demo_{fall_mode}.mp4"
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))

    saved_example = False
    saved_fall_example = False
    total_alerts = 0
    active_fall_banners: dict[int, int] = {}
    active_zone_banners: dict[int, int] = {}
    for frame_idx, result in enumerate(results):
        img = cv2.imread(str(frame_paths[frame_idx]))
        total_alerts += len(result.alerts)
        for text in result.alarm_texts:
            print(f"  frame={frame_idx} {text}")

        draw_frame(img, result, active_fall_banners, active_zone_banners, zone)
        writer.write(img)

        if result.alarm_texts and not saved_example:
            cv2.imwrite(str(clip_dir / f"frame_{fall_mode}_alarm.png"), img)
            saved_example = True
        if result.fall_events and not saved_fall_example:
            cv2.imwrite(str(clip_dir / f"frame_{fall_mode}_fall.png"), img)
            saved_fall_example = True

    writer.release()
    # cv2.VideoWriter(fourcc='mp4v')는 실제로 FMP4 코덱이라 브라우저 <video>에서
    # 재생이 안 된다 — 웹앱에서 바로 재생 가능하도록 H.264로 재인코딩.
    reencode(output_video)
    print(f"[full-pipeline] [{clip_name}] 총 {len(frame_paths)}프레임 처리, 알람 {total_alerts}건 발생")
    print(f"[full-pipeline] [{clip_name}] 영상 저장: {output_video}")


def run_source(name: str, frame_dir: Path, max_frames: int, fall_modes: list[str]) -> None:
    clips = discover_clips(frame_dir)
    if not clips:
        print(f"[full-pipeline] {name}: 클립을 찾을 수 없습니다({frame_dir})")
        return
    print(f"[full-pipeline] {name}: {len(clips)}개 클립 발견 "
          f"(길이={[len(c) for c in clips]})")

    for clip_idx, frame_paths in enumerate(clips):
        if max_frames:
            frame_paths = frame_paths[:max_frames]
        clip_name = f"{name}_clip{clip_idx:02d}"
        clip_dir = OUTPUT_DIR / clip_name
        clip_dir.mkdir(parents=True, exist_ok=True)
        (clip_dir / "clip_info.json").write_text(
            json.dumps({
                "source": name,
                "source_dir": str(frame_dir),
                "clip_index": clip_idx,
                "n_frames": len(frame_paths),
                "first_frame": frame_paths[0].name,
                "last_frame": frame_paths[-1].name,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for fall_mode in fall_modes:
            run_one_clip(clip_name, frame_paths, clip_dir, fall_mode)


def main() -> None:
    import argparse
    from src.pipeline import FALL_MODES  # noqa: E402 (지연 import, argparse choices용)
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=0,
                         help="클립당 최대 프레임 수 제한(0=제한 없음, 클립은 이미 짧게 쪼개져 있음)")
    parser.add_argument("--source", type=str, default=None, choices=list(DEMO_SOURCES),
                         help="생략하면 DEMO_SOURCES 전체(여러 샘플)를 순차 실행")
    parser.add_argument("--fall-mode", type=str, default="bbox_heuristic", choices=list(FALL_MODES),
                         help="낙상 감지 방식: bbox_heuristic(기본, 추적 bbox) / "
                              "keypoint_heuristic(실시간 pose 추출 + 같은 휴리스틱) / "
                              "hdgcn(실시간 pose 추출 + 학습된 HD-GCN 5-way 분류)")
    parser.add_argument("--compare-all", action="store_true",
                         help="지정한 source(들)에 대해 3가지 fall_mode를 모두 순차 실행하여 비교 산출물 생성")
    args = parser.parse_args()

    if not PPE_WEIGHTS.exists():
        print(f"[full-pipeline] PPE 가중치가 없습니다: {PPE_WEIGHTS}")
        return

    sources = {args.source: DEMO_SOURCES[args.source]} if args.source else DEMO_SOURCES
    fall_modes = list(FALL_MODES) if args.compare_all else [args.fall_mode]
    for name, frame_dir in sources.items():
        run_source(name, frame_dir, args.max_frames, fall_modes)


if __name__ == "__main__":
    main()
