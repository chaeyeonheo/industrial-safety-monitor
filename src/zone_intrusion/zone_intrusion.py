"""통제 구역 무단진입 감지.

실제 "위험구역" 라벨/annotation은 이 데이터셋에 없다 — 실배포 시에는 카메라별로
관리자가 사각형 좌표를 설정하는 값이어야 한다. 여기서는 프레임 크기 대비 비율로
정의한 데모용 고정 사각형(DEFAULT_ZONE_RATIO)을 쓴다.

트랙의 bbox 하단 중앙점(발 위치 근사, 사람의 실제 "서 있는 위치"를 bbox 중심보다
더 잘 반영함)이 zone 사각형 안에 들어오는지로 판정한다. PPE/낙상과 같은 설계
원칙(전환 순간만 이벤트화)을 따라 "밖→안" 전환이 일어난 프레임에서만 이벤트를
낸다 — 같은 사람이 zone 안에 계속 머물러도 매 프레임 반복 알람이 뜨지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 데모 기본값: 프레임 중앙부 사각형(가로 30~75%, 세로 35~90%). 실제 카메라
# 배치에서는 관리자가 직접 지정해야 하는 값이며, 여기 비율은 임의의 예시다.
DEFAULT_ZONE_RATIO = (0.30, 0.35, 0.75, 0.90)


def zone_from_ratio(ratio: tuple[float, float, float, float], frame_w: int, frame_h: int
                     ) -> tuple[float, float, float, float]:
    rx1, ry1, rx2, ry2 = ratio
    return rx1 * frame_w, ry1 * frame_h, rx2 * frame_w, ry2 * frame_h


@dataclass
class ZoneIntrusionDetector:
    zone: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel)
    _inside_last_frame: set[int] = field(default_factory=set)

    def update(self, tracks: list) -> list[int]:
        """이번 프레임에 새로 zone에 진입한(직전 프레임엔 zone 밖이었던) track_id
        목록을 반환. tracks는 .track_id/.bbox를 가진 객체 리스트(src.detection_tracking.tracker.Track)."""
        currently_inside = set()
        newly_entered = []
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            foot_x, foot_y = (x1 + x2) / 2, y2
            if self._point_in_zone(foot_x, foot_y):
                currently_inside.add(t.track_id)
                if t.track_id not in self._inside_last_frame:
                    newly_entered.append(t.track_id)
        self._inside_last_frame = currently_inside
        return newly_entered

    def _point_in_zone(self, x: float, y: float) -> bool:
        zx1, zy1, zx2, zy2 = self.zone
        return zx1 <= x <= zx2 and zy1 <= y <= zy2
