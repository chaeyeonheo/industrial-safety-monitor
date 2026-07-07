"""전체 통합 파이프라인 진입점.

[사람 탐지+추적(공유 백본)] → ┬→ [낙상 감지 브랜치 — 아래 3종 모드 중 택1]  ┐
                              └→ [PPE 미착용 판정(간접 연결)]              ┴→ [이벤트 통합] → [NLG 알람]

낙상 감지는 `fall_mode`로 3가지 중 하나를 고른다:
  - "bbox_heuristic" (기본): 추적 bbox 종횡비/수직속도/탐지유실 휴리스틱.
    pose 불필요, 가장 가볍고 실시간성 좋음.
  - "keypoint_heuristic": YOLO11n-pose로 실시간 keypoint를 뽑아 그 bounding
    box로 **같은 휴리스틱 로직**(bbox_heuristic과 동일한 FallHeuristicTrigger)을
    돌린다. "keypoint 기반이면 탐지 bbox보다 정확하지 않을까"를 실측 비교하기
    위한 버전.
  - "hdgcn": YOLO11n-pose로 뽑은 keypoint를 30프레임 버퍼로 쌓아 학습된
    HD-GCN(5-way)으로 분류. 단, 학습 데이터는 AIHub 정답 keypoint였고 여기선
    YOLO11n-pose(COCO-17) -> AIHub16 근사 리매핑을 거친 keypoint라 오프라인
    평가(81.8%)와 정확도가 다를 수 있음(실측 비교 대상).

`run_offline()`은 미리 정해진 프레임 목록(녹화된 영상)을 **완전히 분리된
순차 패스**로 처리한다: 추적 모델 전체 패스 -> GPU 내림 -> PPE 모델 전체
패스 -> GPU 내림 -> (필요시) pose 추출 전체 패스 -> GPU 내림 -> 나머지는
GPU 없이 순수 계산. 여러 모델을 프레임마다 번갈아 호출하면 지속 부하가
커진다는 문제(반복되는 시스템 크래시)에 대응해 이렇게 설계함.

PPE 판정은 track이 처음 관찰된 시점(첫 몇 프레임)에만 내리고 그 이후로는
다시 판정하지 않는다(인과적 방식, 사용자 요청) — 자세한 이유는
docs/FINAL_SUMMARY.md 참고.
"""

from __future__ import annotations

import gc
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
from ultralytics import YOLO

from src.detection_tracking.tracker import PersonTracker, Track
from src.event_aggregator import EventAggregator, EventType, SafetyEvent
from src.fall_detection.heuristic_trigger import FallHeuristicTrigger
from src.fall_detection.hdgcn_live import HDGCNLiveClassifier
from src.fall_detection.pose_extractor import RTMPoseExtractor, YoloPoseExtractor
from src.nlg.template_generator import generate_alarm_text
from src.ppe_detection.indirect_association import (
    REQUIRED_ITEMS, PPEDetection, PersonPPEStatus, check_ppe_compliance,
)
from src.zone_intrusion.zone_intrusion import ZoneIntrusionDetector

FALL_MODES = ("bbox_heuristic", "keypoint_heuristic", "hdgcn")


@dataclass
class FrameResult:
    frame_idx: int
    tracks: list[Track]
    ppe_statuses: list[PersonPPEStatus]
    # track_id -> 이 track 전체 구간을 통틀어 확정한 미착용 목록(프레임마다 안 바뀜)
    track_missing_items: dict[int, list[str]] = field(default_factory=dict)
    fall_events: list[SafetyEvent] = field(default_factory=list)  # 이번 프레임에 새로 뜬 낙상 이벤트만
    zone_events: list[SafetyEvent] = field(default_factory=list)  # 이번 프레임에 새로 뜬 구역진입 이벤트만
    alerts: list[SafetyEvent] = field(default_factory=list)
    alarm_texts: list[str] = field(default_factory=list)
    # track_id -> AIHub16 keypoints(16,3). keypoint_heuristic/hdgcn 모드일 때만 채워짐
    # (bbox_heuristic은 pose를 아예 안 뽑으므로 항상 비어있음) — 시각화 오버레이용.
    keypoints: dict[int, np.ndarray] = field(default_factory=dict)


def _keypoint_containment(track_bbox: tuple[float, float, float, float], keypoints: np.ndarray) -> float:
    """keypoints 중 track_bbox 안에 들어오는(보이는) 점의 비율. pose 모델 자체의
    bbox regression 품질과 무관하게 keypoint 위치만으로 track과 매칭하기 위한
    지표 — fine-tuning된 pose 모델은 학습 라벨의 bbox가 keypoint 범위에서
    근사한 것이라 tracker의 실제 bbox와 IoU가 잘 안 맞는 경우가 실측으로
    확인되어(매칭률 15%), IoU 대신 이 방식을 쓴다."""
    x1, y1, x2, y2 = track_bbox
    visible = keypoints[keypoints[:, 2] > 0]
    if len(visible) == 0:
        return 0.0
    inside = (visible[:, 0] >= x1) & (visible[:, 0] <= x2) & (visible[:, 1] >= y1) & (visible[:, 1] <= y2)
    return float(inside.mean())


class SafetyMonitorPipeline:
    def __init__(
        self,
        ppe_weights: str,
        fps: float = 15.0,
        frame_size: tuple[int, int] | None = None,
        ppe_conf: float = 0.3,
        cooldown_seconds: float = 30.0,
        ppe_infer_every_n_frames: int = 3,
        fall_trigger_kwargs: dict | None = None,
        ppe_decision_window_frames: int = 3,
        person_conf_threshold: float = 0.35,
        fall_mode: str = "bbox_heuristic",
        pose_backend: str = "rtmpose",
        pose_weights: str = "weights/yolo11n-pose.pt",
        hdgcn_weights: str | None = None,
        hdgcn_confidence_threshold: float = 0.5,
        zone: tuple[float, float, float, float] | None = None,
    ) -> None:
        if fall_mode not in FALL_MODES:
            raise ValueError(f"fall_mode는 {FALL_MODES} 중 하나여야 함: {fall_mode}")
        if pose_backend not in ("rtmpose", "yolo11n_pose"):
            raise ValueError(f"pose_backend는 rtmpose/yolo11n_pose 중 하나여야 함: {pose_backend}")
        self.fall_mode = fall_mode
        self.pose_backend = pose_backend
        self.ppe_weights = ppe_weights
        self.pose_weights = pose_weights
        self.hdgcn_weights = hdgcn_weights
        self.hdgcn_confidence_threshold = hdgcn_confidence_threshold
        self.frame_size = frame_size
        self.device = "0" if torch.cuda.is_available() else "cpu"
        self.torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[pipeline] device={self.device} fall_mode={fall_mode} "
              f"({torch.cuda.get_device_name(0) if self.device != 'cpu' else 'CPU'})")
        self.ppe_conf = ppe_conf
        self.person_conf_threshold = person_conf_threshold
        self.ppe_infer_every_n_frames = ppe_infer_every_n_frames
        # 이 track이 처음 나타난 뒤 이만큼의 프레임만 보고 착용여부를 확정한다.
        self.ppe_decision_window_frames = ppe_decision_window_frames
        self.fall_trigger_kwargs = fall_trigger_kwargs or {}
        self.fps = fps
        self.zone = zone  # (x1,y1,x2,y2) 픽셀 좌표, None이면 구역감지 비활성화
        self.aggregator = EventAggregator(cooldown_frames=max(1, int(cooldown_seconds * fps)))

    def _free_gpu(self) -> None:
        gc.collect()
        if self.device != "cpu":
            torch.cuda.empty_cache()

    def run_offline(self, frame_paths: list[str]) -> list[FrameResult]:
        print(f"[pipeline] 1/5 사람 탐지+추적 전용 패스 ({len(frame_paths)}프레임)")
        tracker = PersonTracker(conf_threshold=self.person_conf_threshold, device=self.device)
        tracks_by_frame: list[list[Track]] = list(tracker.track_stream(frame_paths))
        del tracker
        self._free_gpu()

        print("[pipeline] 2/5 PPE 탐지 전용 패스")
        ppe_model = YOLO(self.ppe_weights)
        ppe_detections_by_frame: list[list[PPEDetection]] = []
        last_detections: list[PPEDetection] = []
        for frame_idx, frame_path in enumerate(frame_paths):
            if frame_idx % self.ppe_infer_every_n_frames == 0:
                result = ppe_model.predict(
                    source=frame_path, conf=self.ppe_conf, device=self.device, verbose=False)[0]
                last_detections = [
                    PPEDetection(
                        class_name=result.names[int(b.cls[0])],
                        bbox=tuple(b.xyxy[0].tolist()),
                        confidence=float(b.conf[0]),
                    )
                    for b in result.boxes
                ]
            ppe_detections_by_frame.append(last_detections)
        del ppe_model
        self._free_gpu()

        # 3/5: (keypoint_heuristic/hdgcn 모드일 때만) pose 추출 전용 패스 + track_id 매칭
        keypoints_by_frame: list[dict[int, np.ndarray]] = [dict() for _ in frame_paths]
        if self.fall_mode in ("keypoint_heuristic", "hdgcn"):
            print(f"[pipeline] 3/5 pose 추출 전용 패스 ({self.pose_backend})")
            if self.pose_backend == "rtmpose":
                # 기본값. 이 환경은 onnxruntime CUDA EP가 cuDNN8 의존 dll
                # (zlibwapi.dll) 누락으로 로드 실패해 CPU로 폴백된다(실측 확인)
                # — 그래서 처음부터 CPU로 명시. balanced 모드 기준 약 289ms/프레임.
                pose_extractor = RTMPoseExtractor(mode="balanced", device="cpu")
            else:
                # 주의: AIHub163 GT keypoint로 fine-tuning한 모델(yolo11n_pose)은
                # 누운 자세 keypoint는 잘 잡지만(실측 확인), 라벨이 이미지당
                # 1명만 있어(다른 사람은 라벨 없음) 학습 중 나머지 사람을
                # 억제하도록 배워버렸다 — 프레임당 탐지 수가 1개로 붕괴됨
                # (실측: tracker는 3~5명 찾는데 이 모델은 1명만 찾음). 여러
                # 사람이 동시에 있는 실제 현장에는 아직 못 쓴다 — freeze
                # 학습 등으로 고치기 전까지는 pose_backend="rtmpose"를 기본값으로 유지.
                pose_extractor = YoloPoseExtractor(
                    weights=self.pose_weights, conf_threshold=0.3, device=self.device)
            for frame_idx, (frame_path, tracks) in enumerate(zip(frame_paths, tracks_by_frame)):
                pose_detections = pose_extractor.extract(frame_path)
                for track in tracks:
                    # pose_bbox(모델이 직접 예측한 box)는 안 쓴다 — pose 모델의
                    # box regression 품질과 무관하게, keypoint 자체가 tracker
                    # bbox 안에 얼마나 들어오는지로 매칭한다(위 docstring 참고).
                    best_score, best_kp = 0.5, None  # 과반수 keypoint가 안에 들어와야 매칭
                    for _pose_bbox, kp in pose_detections:
                        score = _keypoint_containment(track.bbox, kp)
                        if score > best_score:
                            best_score, best_kp = score, kp
                    if best_kp is not None:
                        keypoints_by_frame[frame_idx][track.track_id] = best_kp
            del pose_extractor
            self._free_gpu()
        else:
            print("[pipeline] 3/5 건너뜀 (bbox_heuristic은 pose 불필요)")

        print("[pipeline] 4/5 track별 첫 관찰 구간만으로 PPE 판정(이후 고정)")
        statuses_by_frame: list[list[PersonPPEStatus]] = []
        detection_count: dict[tuple[int, str], int] = defaultdict(int)
        frames_seen: dict[int, int] = defaultdict(int)
        last_seen_frame: dict[int, int] = {}
        track_missing_items: dict[int, list[str]] = {}
        ppe_decision_events_by_frame: dict[int, list[tuple[int, str]]] = defaultdict(list)

        for frame_idx, (tracks, ppe_detections) in enumerate(zip(tracks_by_frame, ppe_detections_by_frame)):
            person_tracks = [(t.track_id, t.bbox) for t in tracks]
            statuses = check_ppe_compliance(person_tracks, ppe_detections)
            statuses_by_frame.append(statuses)
            for status in statuses:
                track_id = status.track_id
                if track_id in track_missing_items:
                    continue
                last_seen_frame[track_id] = frame_idx
                frames_seen[track_id] += 1
                for item in status.detected_items:
                    detection_count[(track_id, item)] += 1
                if frames_seen[track_id] >= self.ppe_decision_window_frames:
                    missing = [
                        item for item in REQUIRED_ITEMS
                        if detection_count.get((track_id, item), 0) == 0
                    ]
                    track_missing_items[track_id] = missing
                    for item in missing:
                        ppe_decision_events_by_frame[frame_idx].append((track_id, item))

        for track_id, count in frames_seen.items():
            if track_id not in track_missing_items:
                missing = [
                    item for item in REQUIRED_ITEMS
                    if detection_count.get((track_id, item), 0) == 0
                ]
                track_missing_items[track_id] = missing
                for item in missing:
                    ppe_decision_events_by_frame[last_seen_frame[track_id]].append((track_id, item))

        print(f"[pipeline] 5/5 낙상 감지({self.fall_mode}) + 구역감지 + 이벤트 통합 + NLG (GPU 미사용, hdgcn 모드 제외)")
        fall_events_by_frame = self._run_fall_detection(tracks_by_frame, keypoints_by_frame)

        zone_entries_by_frame: dict[int, list[int]] = {}
        if self.zone is not None:
            zone_detector = ZoneIntrusionDetector(zone=self.zone)
            for frame_idx, tracks in enumerate(tracks_by_frame):
                entered = zone_detector.update(tracks)
                if entered:
                    zone_entries_by_frame[frame_idx] = entered

        results = []
        for frame_idx, (tracks, statuses) in enumerate(zip(tracks_by_frame, statuses_by_frame)):
            ppe_events_this_frame = ppe_decision_events_by_frame.get(frame_idx, [])
            fall_events_this_frame = fall_events_by_frame.get(frame_idx, [])
            zone_entries_this_frame = zone_entries_by_frame.get(frame_idx, [])
            results.append(self._combine_frame(
                frame_idx, tracks, statuses, track_missing_items,
                ppe_events_this_frame, fall_events_this_frame, keypoints_by_frame[frame_idx],
                zone_entries_this_frame))
        return results

    def _run_fall_detection(
        self, tracks_by_frame: list[list[Track]], keypoints_by_frame: list[dict[int, np.ndarray]],
    ) -> dict[int, list[SafetyEvent]]:
        events_by_frame: dict[int, list[SafetyEvent]] = defaultdict(list)

        if self.fall_mode == "bbox_heuristic":
            trigger = FallHeuristicTrigger(
                fps=self.fps, frame_size=self.frame_size, **self.fall_trigger_kwargs)
            for frame_idx, tracks in enumerate(tracks_by_frame):
                for t in trigger.update(frame_idx, tracks):
                    events_by_frame[frame_idx].append(SafetyEvent(
                        track_id=t.track_id, event_type=EventType.FALL_SUSPECTED,
                        frame_idx=frame_idx, detail=t.reason.value, confidence=t.confidence_hint,
                    ))
            return events_by_frame

        if self.fall_mode == "keypoint_heuristic":
            # 같은 FallHeuristicTrigger를, 추적 bbox 대신 keypoint bounding box로 돌린다.
            trigger = FallHeuristicTrigger(
                fps=self.fps, frame_size=self.frame_size, **self.fall_trigger_kwargs)
            for frame_idx, tracks in enumerate(tracks_by_frame):
                kp_map = keypoints_by_frame[frame_idx]
                pseudo_tracks = []
                for t in tracks:
                    kp = kp_map.get(t.track_id)
                    if kp is None:
                        continue
                    valid = kp[kp[:, 2] > 0]
                    if len(valid) == 0:
                        continue
                    bbox = (float(valid[:, 0].min()), float(valid[:, 1].min()),
                            float(valid[:, 0].max()), float(valid[:, 1].max()))
                    pseudo_tracks.append(Track(frame_idx=frame_idx, track_id=t.track_id,
                                                bbox=bbox, confidence=t.confidence))
                for trig in trigger.update(frame_idx, pseudo_tracks):
                    events_by_frame[frame_idx].append(SafetyEvent(
                        track_id=trig.track_id, event_type=EventType.FALL_SUSPECTED,
                        frame_idx=frame_idx, detail=trig.reason.value, confidence=trig.confidence_hint,
                    ))
            return events_by_frame

        # hdgcn 모드: GPU 필요 -> 여기서 로드하고 다 쓰면 내림
        print("[pipeline] HD-GCN 모델 로드")
        if not self.hdgcn_weights:
            raise ValueError("fall_mode='hdgcn'이면 hdgcn_weights가 필요합니다")
        classifier = HDGCNLiveClassifier(
            self.hdgcn_weights, frame_size=self.frame_size, device=self.torch_device)
        for frame_idx, tracks in enumerate(tracks_by_frame):
            kp_map = keypoints_by_frame[frame_idx]
            active_ids = {t.track_id for t in tracks}
            for track_id in list(classifier.buffers.keys()):
                if track_id not in active_ids:
                    classifier.drop_track(track_id)
            for t in tracks:
                kp = kp_map.get(t.track_id)
                if kp is None:
                    continue
                result = classifier.update(t.track_id, kp)
                if result is None:
                    continue
                class_name, confidence = result
                if class_name != "normal" and confidence >= self.hdgcn_confidence_threshold:
                    events_by_frame[frame_idx].append(SafetyEvent(
                        track_id=t.track_id, event_type=EventType.FALL_SUSPECTED,
                        frame_idx=frame_idx, detail=class_name, confidence=confidence,
                    ))
        del classifier
        self._free_gpu()
        return events_by_frame

    def _combine_frame(
        self, frame_idx: int, tracks: list[Track], statuses: list[PersonPPEStatus],
        track_missing_items: dict[int, list[str]], ppe_events_this_frame: list[tuple[int, str]],
        fall_events_this_frame: list[SafetyEvent], keypoints: dict[int, np.ndarray],
        zone_entries_this_frame: list[int],
    ) -> FrameResult:
        candidate_events: list[SafetyEvent] = list(fall_events_this_frame)

        for track_id, item in ppe_events_this_frame:
            candidate_events.append(SafetyEvent(
                track_id=track_id,
                event_type=EventType.PPE_MISSING,
                frame_idx=frame_idx,
                detail=item,
                confidence=0.8,
            ))

        for track_id in zone_entries_this_frame:
            candidate_events.append(SafetyEvent(
                track_id=track_id,
                event_type=EventType.ZONE_INTRUSION,
                frame_idx=frame_idx,
                detail="restricted_zone",
                confidence=1.0,
            ))

        alerts = self.aggregator.submit(candidate_events)
        alarm_texts = [generate_alarm_text(e) for e in alerts]
        return FrameResult(
            frame_idx=frame_idx, tracks=tracks, ppe_statuses=statuses,
            track_missing_items=track_missing_items,
            fall_events=[e for e in alerts if e.event_type == EventType.FALL_SUSPECTED],
            zone_events=[e for e in alerts if e.event_type == EventType.ZONE_INTRUSION],
            alerts=alerts, alarm_texts=alarm_texts, keypoints=keypoints,
        )
