"""PPE(안전보호구) 탐지 YOLO 파인튜닝.

`scripts/convert_aihub163_to_yolo.py`로 만든 데이터셋(helmet/vest/harness/safety_shoes
4개 클래스)으로 YOLO11n을 파인튜닝한다. GPU 메모리 안정성 문제(반복 크래시 이력)
때문에 배치 크기를 보수적으로 잡는다 — 기본값을 낮게 두고 필요시에만 늘릴 것.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = REPO_ROOT / "data/processed/ppe_yolo/dataset.yaml"
WEIGHTS_DIR = REPO_ROOT / "weights"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4, help="보수적 기본값. GPU 메모리 여유 보고 조정")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=2, help="CPU 스레드 경합도 줄이기 위해 보수적으로")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if not DATASET_YAML.exists():
        print(f"[train-ppe] 데이터셋이 없습니다: {DATASET_YAML}. "
              f"먼저 scripts/convert_aihub163_to_yolo.py를 실행하세요.")
        return

    device = args.device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[train-ppe] device={device} "
          f"({torch.cuda.get_device_name(0) if device != 'cpu' and torch.cuda.is_available() else 'CPU'})")
    if device != "cpu" and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[train-ppe] GPU 메모리: 여유 {free/1e9:.2f}GB / 전체 {total/1e9:.2f}GB (학습 시작 전)")

    model_path = WEIGHTS_DIR / args.model
    model = YOLO(str(model_path) if model_path.exists() else args.model)

    results = model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=device,
        project=str(REPO_ROOT / "outputs" / "ppe_yolo_runs"),
        name="train",
        exist_ok=True,
    )
    print(f"[train-ppe] 학습 완료. 결과: {results.save_dir}")


if __name__ == "__main__":
    main()
