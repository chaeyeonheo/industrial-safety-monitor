"""전체 통합 파이프라인 진입점.

[사람 탐지+추적(공유 백본)] → ┬→ [낙상 Stage A 휴리스틱]     ┐
                              └→ [PPE 미착용 판정(간접 연결)] ┴→ [이벤트 통합] → [NLG 알람]

`run_offline()`은 미리 정해진 프레임 목록(녹화된 영상)을 **3단계로 완전히
분리**해서 처리한다: (1) 추적 모델만 GPU에 올려 전체 프레임의 track을 뽑고
모델을 내림, (2) PPE 모델만 GPU에 올려 전체 프레임의 보호구 탐지를 뽑고
모델을 내림, (3) 나머지(이벤트 통합/NLG)는 GPU 없이 순수 계산. 두 모델을
동시에 GPU에 띄워 프레임마다 번갈아 호출하는 대신 완전히 순차적으로 실행해
지속 부하를 낮춘다(반복되는 시스템 크래시에 대한 사용자 요청 반영).

**PPE 판정은 이 track이 처음 관찰된 시점(첫 몇 프레임)에만 내리고, 그 이후로는
다시 판정하지 않고 그대로 유지한다** — 미래 프레임을 미리 들여다보는 게 아니라,
"처음 봤을 때 판단한 걸 계속 쓴다"는 뜻(실시간 스트리밍으로 들어와도 그대로
동작하는 인과적 방식, 사용자 요청). 매 프레임 새로 판정하면 각도/블러 때문에
프레임마다 착용/미착용이 깜빡이는데, 같은 사람의 보호구가 영상 중간에
사라졌다 나타났다 할 리는 없으므로 한 번 정해지면 그 track이 사라질 때까지
고정한다.

Stage B(HD-GCN)는 실시간 pose 추출기가 아직 없어 이 라이브 파이프라인에는
연결하지 못했고, 라벨링된 오프라인 데이터로만 평가한다
(scripts/evaluate_fall_ablation.py, docs/ablation_studies.md 참고).
"""

from __future__ import annotations

import gc
from collections import defaultdict
from dataclasses import dataclass, field

import torch
from ultralytics import YOLO

from src.detection_tracking.tracker import PersonTracker, Track
from src.event_aggregator import EventAggregator, EventType, SafetyEvent
from src.fall_detection.heuristic_trigger import FallHeuristicTrigger
from src.nlg.template_generator import generate_alarm_text
from src.ppe_detection.indirect_association import (
    REQUIRED_ITEMS, PPEDetection, PersonPPEStatus, check_ppe_compliance,
)


@dataclass
class FrameResult:
    frame_idx: int
    tracks: list[Track]
    ppe_statuses: list[PersonPPEStatus]
    # track_id -> 이 track 전체 구간을 통틀어 확정한 미착용 목록(프레임마다 안 바뀜)
    track_missing_items: dict[int, list[str]] = field(default_factory=dict)
    fall_events: list[SafetyEvent] = field(default_factory=list)  # 이번 프레임에 새로 뜬 낙상 이벤트만
    alerts: list[SafetyEvent] = field(default_factory=list)
    alarm_texts: list[str] = field(default_factory=list)


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
        person_conf_threshold: float = 0.5,
    ) -> None:
        self.ppe_weights = ppe_weights
        self.device = "0" if torch.cuda.is_available() else "cpu"
        print(f"[pipeline] device={self.device} "
              f"({torch.cuda.get_device_name(0) if self.device != 'cpu' else 'CPU'})")
        self.ppe_conf = ppe_conf
        self.person_conf_threshold = person_conf_threshold
        self.ppe_infer_every_n_frames = ppe_infer_every_n_frames
        # 이 track이 처음 나타난 뒤 이만큼의 프레임만 보고 착용여부를 확정한다
        # (1프레임만 보면 우연한 오탐/미탐에 취약해서 살짝 여유를 둠). 그 이후
        # 프레임은 이 확정값을 그대로 재사용 — 다시 판정하지 않음.
        self.ppe_decision_window_frames = ppe_decision_window_frames
        self.fall_trigger = FallHeuristicTrigger(
            fps=fps, frame_size=frame_size, **(fall_trigger_kwargs or {}))
        self.aggregator = EventAggregator(cooldown_frames=max(1, int(cooldown_seconds * fps)))

    def _free_gpu(self) -> None:
        gc.collect()
        if self.device != "cpu":
            torch.cuda.empty_cache()

    def run_offline(self, frame_paths: list[str]) -> list[FrameResult]:
        """녹화된 프레임 목록을 대상으로 추적 모델 → PPE 모델 순으로 완전히
        분리해 실행한 뒤, GPU 없이 이벤트를 통합한다."""
        print(f"[pipeline] 1/4 사람 탐지+추적 전용 패스 ({len(frame_paths)}프레임)")
        tracker = PersonTracker(conf_threshold=self.person_conf_threshold, device=self.device)
        tracks_by_frame: list[list[Track]] = list(tracker.track_stream(frame_paths))
        del tracker
        self._free_gpu()

        print("[pipeline] 2/4 PPE 탐지 전용 패스")
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

        print("[pipeline] 3/4 track별 첫 관찰 구간만으로 PPE 판정(이후 고정)")
        statuses_by_frame: list[list[PersonPPEStatus]] = []
        detection_count: dict[tuple[int, str], int] = defaultdict(int)
        frames_seen: dict[int, int] = defaultdict(int)
        track_missing_items: dict[int, list[str]] = {}

        for tracks, ppe_detections in zip(tracks_by_frame, ppe_detections_by_frame):
            person_tracks = [(t.track_id, t.bbox) for t in tracks]
            statuses = check_ppe_compliance(person_tracks, ppe_detections)
            statuses_by_frame.append(statuses)
            for status in statuses:
                track_id = status.track_id
                if track_id in track_missing_items:
                    continue  # 이미 확정됨 — 다시 안 봄(과거 프레임 재판정 없음)
                frames_seen[track_id] += 1
                for item in status.detected_items:
                    detection_count[(track_id, item)] += 1
                if frames_seen[track_id] >= self.ppe_decision_window_frames:
                    track_missing_items[track_id] = [
                        item for item in REQUIRED_ITEMS
                        if detection_count.get((track_id, item), 0) == 0
                    ]

        # 관찰 프레임이 확정 창보다 짧게 끝난(중간에 사라진) track은 마지막에
        # 있는 그대로로 확정한다.
        for track_id, count in frames_seen.items():
            if track_id not in track_missing_items:
                track_missing_items[track_id] = [
                    item for item in REQUIRED_ITEMS
                    if detection_count.get((track_id, item), 0) == 0
                ]

        print("[pipeline] 4/4 이벤트 통합 + NLG (GPU 미사용)")
        results = []
        for frame_idx, (tracks, statuses) in enumerate(zip(tracks_by_frame, statuses_by_frame)):
            results.append(self._combine_frame(frame_idx, tracks, statuses, track_missing_items))
        return results

    def _combine_frame(
        self, frame_idx: int, tracks: list[Track], statuses: list[PersonPPEStatus],
        track_missing_items: dict[int, list[str]],
    ) -> FrameResult:
        candidate_events: list[SafetyEvent] = []
        fall_events: list[SafetyEvent] = []

        for trigger in self.fall_trigger.update(frame_idx, tracks):
            event = SafetyEvent(
                track_id=trigger.track_id,
                event_type=EventType.FALL_SUSPECTED,
                frame_idx=frame_idx,
                detail=trigger.reason.value,
                confidence=trigger.confidence_hint,
            )
            candidate_events.append(event)
            fall_events.append(event)

        alerts = self.aggregator.submit(candidate_events)
        alarm_texts = [generate_alarm_text(e) for e in alerts]
        return FrameResult(
            frame_idx=frame_idx, tracks=tracks, ppe_statuses=statuses,
            track_missing_items=track_missing_items,
            fall_events=[e for e in alerts if e.event_type == EventType.FALL_SUSPECTED],
            alerts=alerts, alarm_texts=alarm_texts,
        )
