"""YOLO11n-pose를 AIHub163 GT keypoint(측면/누운 자세 포함)로 fine-tuning한다.

실측으로 확인된 문제 1: YOLO11n-pose/RTMPose 모두 COCO/body7 사전학습 데이터가
서 있는/걷는 정면 위주라 측면·누운 자세에서 keypoint 품질이 크게 떨어진다
(docs/ablation_studies.md). AIHub163은 낙상/넘어짐 사고 카테고리라 누운 자세가
실제로 포함돼 있어, 이 데이터로 fine-tuning하면 그 blind spot을 메울 수 있다.

실측으로 확인된 문제 2(1차 시도 후 발견): AIHub163 라벨은 이미지 한 장에
사고 당사자 1명만 라벨링돼 있고, 같은 화면의 다른 사람들은 라벨이 없다.
box/cls 손실을 기본값 그대로 학습하면 "라벨 안 된 사람 = 배경"이라고
학습돼버려서, 실제로 3~5명이 있는 화면에서도 1명만 찾도록 회귀한다(실측
확인: conf=0.01/max_det=300으로 풀어도 후보 자체가 1~2개뿐). 그래서 이번엔
box/cls/dfl 손실 가중치를 0으로 줘서 "다른 사람은 배경"이라고 가르치는
경로를 차단하고, pose(keypoint 위치) 손실만으로 keypoint 정확도만 개선한다
— 사전학습된 다중 사람 검출 능력은 그대로 보존하려는 의도.

하드웨어 안정성 이력(반복 크래시) 때문에 처음엔 적은 epoch/작은 데이터로
GPU가 안정적으로 버티는지부터 확인하고, 문제 없으면 점진적으로 늘린다
(사용자 확인된 진행 방식).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = REPO_ROOT / "data/processed/pose_finetune/dataset.yaml"
BASE_WEIGHTS = REPO_ROOT / "weights/yolo11n-pose.pt"  # 원본(버그 있던 1차 fine-tune 결과 아님)에서 다시 시작


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="작게 시작(사용자 요청)")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", type=str, default="train_kpt_only",
                         help="1차 시도(train/)와 구분되는 새 run 이름")
    parser.add_argument("--box", type=float, default=0.0, help="박스 위치 손실 가중치(0=학습 안 함)")
    parser.add_argument("--cls", type=float, default=0.0, help="사람/배경 분류 손실 가중치(0=학습 안 함)")
    parser.add_argument("--dfl", type=float, default=0.0, help="박스 분포 손실 가중치(0=학습 안 함)")
    args = parser.parse_args()

    model = YOLO(str(BASE_WEIGHTS))
    model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        device="0",
        project=str(REPO_ROOT / "outputs/pose_finetune_runs"),
        name=args.name,
        exist_ok=True,
        patience=0,
        verbose=True,
    )


if __name__ == "__main__":
    main()
