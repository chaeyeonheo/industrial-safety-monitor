"""AIHub 프레임 폴더 하나를 "같은 씬"으로 억지로 묶지 않고, 프레임 번호 간격이
크게 벌어지는 지점에서 별도 클립으로 쪼갠다.

AIHub163 keypoint 라벨은 원본 영상에서 드문드문(중앙값 간격 7, 최대 60+)
샘플링된 프레임만 제공한다. 지금까지는 폴더 하나 = 영상 하나로 취급했는데,
간격이 크게 벌어지는 구간은 실제로는 다른 촬영 컷/불연속 구간일 가능성이 커서
"같은 사람이 쭉 이어져 있다"는 tracking/낙상 신호의 전제 자체가 깨진다.
그래서 간격이 임계치를 넘으면 그 지점에서 새 클립으로 분리한다(1~2초짜리
짧은 클립이 나와도 그대로 둔다 — 억지로 이어붙이지 않는 것이 원칙).
"""

from __future__ import annotations

import re
from pathlib import Path

_FRAME_NUM_RE = re.compile(r"(\d+)\.jpg$")

# 5개 데모 소스 전체의 프레임 간격 중앙값이 7이었다(median_gap 실측,
# scratchpad 분석). 4배(≈30) 이상 벌어지면 "정상적인 샘플링 간격"이 아니라
# 별도 컷/불연속으로 보는 게 합리적이라고 판단한 경험적 임계치.
DEFAULT_MAX_GAP = 30
DEFAULT_MIN_CLIP_FRAMES = 5


def _frame_number(path: Path) -> int:
    m = _FRAME_NUM_RE.search(path.name)
    if not m:
        raise ValueError(f"프레임 번호를 못 찾음: {path.name}")
    return int(m.group(1))


def discover_clips(
    frame_dir: Path,
    max_gap: int = DEFAULT_MAX_GAP,
    min_clip_frames: int = DEFAULT_MIN_CLIP_FRAMES,
) -> list[list[Path]]:
    """frame_dir 안의 *.jpg를 프레임 번호 순으로 정렬한 뒤, 번호 간격이 max_gap을
    넘는 지점마다 새 클립으로 분리한다. min_clip_frames보다 짧은 클립은 버린다
    (추적/낙상 신호를 낼 최소 프레임 수도 안 되는 조각은 의미가 없어서)."""
    paths = sorted(frame_dir.glob("*.jpg"), key=_frame_number)
    if not paths:
        return []

    clips: list[list[Path]] = []
    current = [paths[0]]
    for prev, curr in zip(paths, paths[1:]):
        if _frame_number(curr) - _frame_number(prev) > max_gap:
            clips.append(current)
            current = []
        current.append(curr)
    clips.append(current)

    return [c for c in clips if len(c) >= min_clip_frames]
