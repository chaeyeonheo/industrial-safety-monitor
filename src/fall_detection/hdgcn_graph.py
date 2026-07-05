"""HD-GCN용 커스텀 그래프: AIHub 16-keypoint 계층 구조.

third_party/HD-GCN의 그래프 유틸(graph/tools.py)은 재사용하되, NTU 25-joint
전용으로 하드코딩된 계층 대신 우리 16-joint 레이아웃(docs/keypoint_mapping.md의
best-effort 추론 매핑)에 맞는 계층을 직접 정의한다.

CoM(중심) = 9(골반 중심)에서 사지 말단으로 뻗어나가는 5단계 계층:
  level0: [9]                골반 중심
  level1: [2, 10, 11]         척추/좌우 엉덩이
  level2: [1, 3, 4, 12, 13]   목/좌우 어깨/좌우 무릎
  level3: [0, 5, 6, 14, 15]   코/좌우 팔꿈치/좌우 발목
  level4: [7, 8]              좌우 손목

주의: docs/keypoint_mapping.md는 2프레임만 시각 검증한 best-effort 매핑이라
확정이 아니다. 학습 성능이 예상보다 낮으면 이 계층 정의부터 재검토할 것.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HDGCN_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "HD-GCN"
if str(_HDGCN_ROOT) not in sys.path:
    sys.path.insert(0, str(_HDGCN_ROOT))

from graph import tools  # noqa: E402  (third_party/HD-GCN/graph/tools.py)

NUM_NODE = 16
COM = 9  # 골반 중심(pelvis center), 0-indexed

GROUPS_0INDEXED: list[list[int]] = [
    [9],
    [2, 10, 11],
    [1, 3, 4, 12, 13],
    [0, 5, 6, 14, 15],
    [7, 8],
]


def _get_edgeset_0indexed(groups: list[list[int]]):
    """third_party/HD-GCN/graph/tools.py의 get_edgeset()과 동일한 로직이지만,
    NTU 1-indexed 관절 번호 대신 우리 0-indexed 관절 번호를 그대로 쓴다."""
    identity = []
    forward_hierarchy = []
    reverse_hierarchy = []

    for i in range(len(groups) - 1):
        self_link = groups[i] + groups[i + 1]
        identity.append([(j, j) for j in self_link])

        forward_g = [(j, k) for j in groups[i] for k in groups[i + 1]]
        forward_hierarchy.append(forward_g)

        reverse_g = [(j, k) for j in groups[-1 - i] for k in groups[-2 - i]]
        reverse_hierarchy.append(reverse_g)

    edges = []
    for i in range(len(groups) - 1):
        edges.append([identity[i], forward_hierarchy[i], reverse_hierarchy[-1 - i]])
    return edges


class Graph:
    """HD-GCN Model이 `import_class(graph)(**graph_args)`로 동적 생성하는 클래스.
    `self.A`가 (인접행렬, CoM) 튜플이어야 한다(third_party/HD-GCN/model/HDGCN.py 참고)."""

    def __init__(self, CoM: int = COM, labeling_mode: str = "spatial"):
        self.num_node = NUM_NODE
        self.CoM = CoM
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode: str | None = None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == "spatial":
            edges = _get_edgeset_0indexed(GROUPS_0INDEXED)
            A = tools.get_hierarchical_graph(NUM_NODE, edges)  # (L, 3, 16, 16)
        else:
            raise ValueError(labeling_mode)
        return A, self.CoM


# --- AHA(Attention-guided Hierarchy Aggregation) 모듈 호환용 monkeypatch ---
#
# HD_Gconv 내부의 AHA 모듈은 우리 커스텀 Graph 클래스를 거치지 않고
# `graph.tools.get_groups(dataset='NTU', CoM=CoM)`를 직접 하드코딩 호출한다
# (third_party/HD-GCN/model/HDGCN.py의 AHA.__init__). NTU용 get_groups는
# CoM=1/2/21만 정의되어 있어 우리 CoM=9를 넘기면 ValueError가 난다.
#
# 이 함수를 감싸서, CoM이 우리 값(COM=9)일 때만 우리 계층을 반환하고 그 외에는
# 원래 동작을 그대로 유지한다. AHA는 반환된 groups 리스트를 제자리에서
#변형(mutate)하므로, 호출마다 새 복사본을 반환해야 한다(안 그러면 레이어가
# 여러 개 생성될 때 두 번째 호출부터 이미 변형된 값을 또 변형해 깨진다).
_ORIGINAL_GET_GROUPS = tools.get_groups
GROUPS_1INDEXED_FOR_AHA = [[j + 1 for j in group] for group in GROUPS_0INDEXED]


def _patched_get_groups(dataset: str = "NTU", CoM: int = 21):
    if CoM == COM:
        return [list(group) for group in GROUPS_1INDEXED_FOR_AHA]
    return _ORIGINAL_GET_GROUPS(dataset=dataset, CoM=CoM)


tools.get_groups = _patched_get_groups

# HDGCN.py가 `from graph.tools import get_groups`로 이미 이름을 로컬에
# 바인딩했을 수 있으므로, 그 모듈이 import돼 있다면 거기도 같이 패치한다.
if "model.HDGCN" in sys.modules:
    sys.modules["model.HDGCN"].get_groups = _patched_get_groups
