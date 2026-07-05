"""PPE 간접 연결: 사람 bbox(Phase 1)와 보호구 bbox(Phase 3 YOLO)를 결합해
"이 사람이 어떤 보호구를 안 갖췄는지" 판정한다.

원안(지시문)은 "IoU 매칭"이라고 되어 있지만, 보호구(헬멧 등)는 사람 전체
bbox에 비해 훨씬 작은 신체 부위 박스라 전체 bbox 간 IoU는 거의 항상 0에
가까워 쓸모가 없다. 대신 (a) 보호구 bbox 중심이 사람 bbox 안에 포함되는지,
(b) 사람 bbox 내에서 신체 부위상 기대되는 상대 위치(머리=위쪽, 발=아래쪽)에
있는지 두 조건으로 매칭한다 — 여러 사람이 겹쳐 있을 때 남의 보호구를 잘못
매칭하는 것을 줄이기 위함.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Bbox = tuple[float, float, float, float]

REQUIRED_ITEMS = ["helmet", "vest", "harness", "safety_shoes"]

# 사람 bbox 높이를 0(머리)~1(발)로 정규화했을 때, 각 보호구가 있을 것으로
# 기대되는 상대 y축 범위. docs/ppe_class_mapping.md의 실측 관찰 기반.
EXPECTED_Y_RANGE: dict[str, tuple[float, float]] = {
    "helmet": (0.0, 0.35),
    "vest": (0.15, 0.75),
    "harness": (0.15, 0.75),
    "safety_shoes": (0.75, 1.0),
}


@dataclass
class PPEDetection:
    class_name: str
    bbox: Bbox
    confidence: float


@dataclass
class PersonPPEStatus:
    track_id: int
    person_bbox: Bbox
    detected_items: dict[str, PPEDetection] = field(default_factory=dict)

    @property
    def missing_items(self) -> list[str]:
        return [item for item in REQUIRED_ITEMS if item not in self.detected_items]

    @property
    def fully_equipped(self) -> bool:
        return not self.missing_items


def _bbox_center(bbox: Bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def associate_ppe_to_person(person_bbox: Bbox, ppe_detections: list[PPEDetection]) -> dict[str, PPEDetection]:
    """이 사람 bbox에 매칭되는 보호구들을 반환(클래스당 최고 confidence 1개)."""
    px1, py1, px2, py2 = person_bbox
    person_height = max(py2 - py1, 1e-6)

    matched: dict[str, PPEDetection] = {}
    for det in ppe_detections:
        cx, cy = _bbox_center(det.bbox)
        if not (px1 <= cx <= px2 and py1 <= cy <= py2):
            continue  # 이 사람 몸 안에 없음

        rel_y = (cy - py1) / person_height
        lo, hi = EXPECTED_Y_RANGE.get(det.class_name, (0.0, 1.0))
        if not (lo <= rel_y <= hi):
            continue  # 기대 신체 부위 위치와 안 맞음(다른 사람 것일 가능성)

        existing = matched.get(det.class_name)
        if existing is None or det.confidence > existing.confidence:
            matched[det.class_name] = det

    return matched


def check_ppe_compliance(
    person_tracks: list[tuple[int, Bbox]],  # (track_id, bbox)
    ppe_detections: list[PPEDetection],
) -> list[PersonPPEStatus]:
    """한 프레임의 사람 track들과 보호구 탐지 결과를 받아 사람별 착용 현황을 반환."""
    statuses = []
    for track_id, person_bbox in person_tracks:
        detected = associate_ppe_to_person(person_bbox, ppe_detections)
        statuses.append(PersonPPEStatus(track_id=track_id, person_bbox=person_bbox, detected_items=detected))
    return statuses
