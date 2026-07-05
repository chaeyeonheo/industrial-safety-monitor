"""전체 통합 파이프라인 진입점.

[사람 탐지+추적(공유 백본)] → ┬→ [낙상 Stage A 휴리스틱]     ┐
                              └→ [PPE 미착용 판정(간접 연결)] ┴→ [이벤트 통합] → [NLG 알람]

한 프레임 처리마다 track_id별로 발생한 이벤트(낙상 의심/보호구 미착용)를
쿨다운 적용해서 반환한다. Stage B(HD-GCN)는 실시간 pose 추출기가 아직 없어
이 라이브 파이프라인에는 연결하지 못했고, 라벨링된 오프라인 데이터로만 평가한다
(scripts/evaluate_fall_heuristic_vs_hdgcn.py, docs/ablation_studies.md 참고).
"""

from __future__ import annotations

from dataclasses import dataclass

from ultralytics import YOLO

from src.detection_tracking.tracker import PersonTracker, Track
from src.event_aggregator import EventAggregator, EventType, SafetyEvent
from src.fall_detection.heuristic_trigger import FallHeuristicTrigger
from src.nlg.template_generator import generate_alarm_text
from src.ppe_detection.indirect_association import PPEDetection, check_ppe_compliance, PersonPPEStatus


@dataclass
class FrameResult:
    frame_idx: int
    tracks: list[Track]
    ppe_statuses: list[PersonPPEStatus]
    alerts: list[SafetyEvent]
    alarm_texts: list[str]


class SafetyMonitorPipeline:
    def __init__(
        self,
        ppe_weights: str,
        fps: float = 15.0,
        frame_size: tuple[int, int] | None = None,
        ppe_conf: float = 0.3,
        cooldown_seconds: float = 30.0,
    ) -> None:
        self.tracker = PersonTracker(conf_threshold=0.4)
        self.ppe_model = YOLO(ppe_weights)
        self.ppe_conf = ppe_conf
        self.fall_trigger = FallHeuristicTrigger(fps=fps, frame_size=frame_size)
        self.aggregator = EventAggregator(cooldown_frames=max(1, int(cooldown_seconds * fps)))

    def process_frame(self, frame_idx: int, tracks: list[Track], frame_bgr) -> FrameResult:
        candidate_events: list[SafetyEvent] = []

        for trigger in self.fall_trigger.update(frame_idx, tracks):
            candidate_events.append(SafetyEvent(
                track_id=trigger.track_id,
                event_type=EventType.FALL_SUSPECTED,
                frame_idx=frame_idx,
                detail=trigger.reason.value,
                confidence=trigger.confidence_hint,
            ))

        ppe_result = self.ppe_model.predict(source=frame_bgr, conf=self.ppe_conf, verbose=False)[0]
        ppe_detections = [
            PPEDetection(
                class_name=ppe_result.names[int(box.cls[0])],
                bbox=tuple(box.xyxy[0].tolist()),
                confidence=float(box.conf[0]),
            )
            for box in ppe_result.boxes
        ]
        person_tracks = [(t.track_id, t.bbox) for t in tracks]
        statuses = check_ppe_compliance(person_tracks, ppe_detections)
        for status in statuses:
            for missing_item in status.missing_items:
                candidate_events.append(SafetyEvent(
                    track_id=status.track_id,
                    event_type=EventType.PPE_MISSING,
                    frame_idx=frame_idx,
                    detail=missing_item,
                    confidence=0.8,
                ))

        alerts = self.aggregator.submit(candidate_events)
        alarm_texts = [generate_alarm_text(e) for e in alerts]
        return FrameResult(
            frame_idx=frame_idx, tracks=tracks, ppe_statuses=statuses,
            alerts=alerts, alarm_texts=alarm_texts,
        )
