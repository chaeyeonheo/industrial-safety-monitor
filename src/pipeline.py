"""전체 통합 파이프라인 진입점.

[사람 탐지+추적(공유 백본)] → ┬→ [낙상 Stage A 휴리스틱]     ┐
                              └→ [PPE 미착용 판정(간접 연결)] ┴→ [이벤트 통합] → [NLG 알람]

`run_offline()`은 미리 정해진 프레임 목록(녹화된 영상)을 **3단계로 완전히
분리**해서 처리한다: (1) 추적 모델만 GPU에 올려 전체 프레임의 track을 뽑고
모델을 내림, (2) PPE 모델만 GPU에 올려 전체 프레임의 보호구 탐지를 뽑고
모델을 내림, (3) 나머지(이벤트 통합/NLG)는 GPU 없이 순수 계산. 두 모델을
동시에 GPU에 띄워 프레임마다 번갈아 호출하는 대신 완전히 순차적으로 실행해
지속 부하를 낮춘다(반복되는 시스템 크래시에 대한 사용자 요청 반영).

Stage B(HD-GCN)는 실시간 pose 추출기가 아직 없어 이 라이브 파이프라인에는
연결하지 못했고, 라벨링된 오프라인 데이터로만 평가한다
(scripts/evaluate_fall_ablation.py, docs/ablation_studies.md 참고).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
from ultralytics import YOLO

from src.detection_tracking.tracker import PersonTracker, Track
from src.event_aggregator import EventAggregator, EventType, SafetyEvent
from src.fall_detection.heuristic_trigger import FallHeuristicTrigger
from src.nlg.template_generator import generate_alarm_text
from src.ppe_detection.indirect_association import (
    PPEDetection, PersonPPEStatus, PPEStabilityFilter, check_ppe_compliance,
)


@dataclass
class FrameResult:
    frame_idx: int
    tracks: list[Track]
    ppe_statuses: list[PersonPPEStatus]
    stable_missing: dict[int, list[str]]  # track_id -> 시간적으로 안정화된 미착용 목록
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
        ppe_infer_every_n_frames: int = 3,
        fall_trigger_kwargs: dict | None = None,
    ) -> None:
        self.ppe_weights = ppe_weights
        self.device = "0" if torch.cuda.is_available() else "cpu"
        print(f"[pipeline] device={self.device} "
              f"({torch.cuda.get_device_name(0) if self.device != 'cpu' else 'CPU'})")
        self.ppe_conf = ppe_conf
        self.ppe_infer_every_n_frames = ppe_infer_every_n_frames
        self.fall_trigger = FallHeuristicTrigger(
            fps=fps, frame_size=frame_size, **(fall_trigger_kwargs or {}))
        self.aggregator = EventAggregator(cooldown_frames=max(1, int(cooldown_seconds * fps)))
        self.ppe_stability = PPEStabilityFilter()

    def _free_gpu(self) -> None:
        gc.collect()
        if self.device != "cpu":
            torch.cuda.empty_cache()

    def run_offline(self, frame_paths: list[str]) -> list[FrameResult]:
        """녹화된 프레임 목록을 대상으로 추적 모델 → PPE 모델 순으로 완전히
        분리해 실행한 뒤, GPU 없이 이벤트를 통합한다."""
        print(f"[pipeline] 1/3 사람 탐지+추적 전용 패스 ({len(frame_paths)}프레임)")
        tracker = PersonTracker(conf_threshold=0.4, device=self.device)
        tracks_by_frame: list[list[Track]] = list(tracker.track_stream(frame_paths))
        del tracker
        self._free_gpu()

        print("[pipeline] 2/3 PPE 탐지 전용 패스")
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

        print("[pipeline] 3/3 이벤트 통합 + NLG (GPU 미사용)")
        results = []
        for frame_idx, (tracks, ppe_detections) in enumerate(zip(tracks_by_frame, ppe_detections_by_frame)):
            results.append(self._combine_frame(frame_idx, tracks, ppe_detections))
        return results

    def _combine_frame(
        self, frame_idx: int, tracks: list[Track], ppe_detections: list[PPEDetection],
    ) -> FrameResult:
        candidate_events: list[SafetyEvent] = []

        for trigger in self.fall_trigger.update(frame_idx, tracks):
            candidate_events.append(SafetyEvent(
                track_id=trigger.track_id,
                event_type=EventType.FALL_SUSPECTED,
                frame_idx=frame_idx,
                detail=trigger.reason.value,
                confidence=trigger.confidence_hint,
            ))

        person_tracks = [(t.track_id, t.bbox) for t in tracks]
        statuses = check_ppe_compliance(person_tracks, ppe_detections)
        stable_missing: dict[int, list[str]] = {}
        for status in statuses:
            missing_now = self.ppe_stability.update(status)
            stable_missing[status.track_id] = missing_now
            for missing_item in missing_now:
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
            stable_missing=stable_missing, alerts=alerts, alarm_texts=alarm_texts,
        )
