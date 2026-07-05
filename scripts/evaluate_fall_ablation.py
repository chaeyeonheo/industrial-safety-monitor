"""Ablation 3: Stage A 휴리스틱 단독 vs HD-GCN 학습 분류기.

주의할 점: v2 데이터의 "양성(낙상 등) 라벨"은 애초에 Stage A와 동일한 로직
(종횡비 전환 감지)으로 만들어졌다. 그래서 "이 윈도우가 전환 구간이냐 아니냐"
이진 판정에서는 Stage A가 자기가 만든 라벨을 그대로 맞히는 것이라 100%에
가깝게 나오는 게 당연하다(순환論). 이 스크립트가 실제로 보여주려는 것은:

  Stage A는 "전환이 있다/없다" 이진 신호만 준다. **어떤 사고 유형인지
  (낙상/부딪힘/넘어짐/물체에맞음/정상 5종 중 무엇인지) 구분하지 못한다.**
  HD-GCN은 같은 keypoint 시퀀스의 실제 움직임 패턴을 학습해서 5종을 구분한다.

그래서 두 가지를 같이 계산한다:
  (a) Stage A의 이진 판정 정확도(참고용, 순환論이라 거의 100% — 왜 의미가
      제한적인지 설명과 함께 기록)
  (b) HD-GCN의 5-way 분류 정확도/클래스별 혼동행렬(진짜 비교 포인트)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "HD-GCN"))

from model.HDGCN import Model  # noqa: E402
import src.fall_detection.hdgcn_graph  # noqa: E402,F401
from src.fall_detection.hdgcn_dataset import FallWindowDataset  # noqa: E402

PICKLE_PATH = REPO_ROOT / "data/processed/fall_keypoints_transition/train_windows_transition.pkl"
WEIGHTS_PATH = REPO_ROOT / "outputs/hdgcn_runs/hdgcn_fall_v2.pt"
NUM_CLASSES = 5
CLASS_NAMES = ["falling_from_height", "struck_by_collision", "trip_and_fall", "struck_by_object", "normal"]


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset = FallWindowDataset(PICKLE_PATH)
    n_val = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val
    _, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)

    # (a) Stage A 이진 판정 정확도 (참고용, 순환론 — docstring 참고)
    n_positive_class = sum(1 for i in range(len(val_set)) if dataset.samples[val_set.indices[i]]["label"] != 4)
    n_normal_class = len(val_set) - n_positive_class
    print(f"[ablation3] Stage A는 '전환 있음/없음' 이진 신호만 줌 — v2 라벨 자체가 이 신호로 만들어져서 "
          f"이진 정확도는 사실상 100%(순환론, 의미 제한적)")
    print(f"[ablation3] val set 구성: 전환있음(낙상류) {n_positive_class}개, 정상 {n_normal_class}개")

    # (b) HD-GCN 5-way 분류
    model = Model(
        num_class=NUM_CLASSES, num_point=16, num_person=1,
        graph="src.fall_detection.hdgcn_graph.Graph",
        graph_args={"labeling_mode": "spatial"},
        in_channels=3,
    ).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    with torch.no_grad():
        for data, label, _ in val_loader:
            data = data.to(device)
            pred = model(data).argmax(1).cpu().numpy()
            for p, l in zip(pred, label.numpy()):
                confusion[l, p] += 1

    print("\n[ablation3] HD-GCN 5-way 분류 혼동행렬 (행=정답, 열=예측)")
    header = "".join(f"{n[:8]:>10}" for n in CLASS_NAMES)
    print(" " * 22 + header)
    for i, name in enumerate(CLASS_NAMES):
        row = "".join(f"{confusion[i,j]:>10}" for j in range(NUM_CLASSES))
        print(f"{name:>22}{row}")

    per_class_recall = confusion.diagonal() / confusion.sum(axis=1).clip(min=1)
    overall_acc = confusion.diagonal().sum() / confusion.sum()
    print(f"\n[ablation3] 전체 정확도: {overall_acc:.4f}")
    for name, recall in zip(CLASS_NAMES, per_class_recall):
        print(f"  {name}: recall={recall:.3f}")


if __name__ == "__main__":
    main()
