"""AIHub163 keypoint 라벨(16점, docs/keypoint_mapping.md)을 Ultralytics YOLO
pose 학습 포맷으로 변환한다.

목적: 실측으로 확인된 문제(YOLO11n-pose/RTMPose 모두 측면·누운 자세에서
keypoint 품질이 크게 떨어짐, docs/ablation_studies.md)를 해결하기 위해, 이미
확보한 AIHub163 GT keypoint(낙상/넘어짐 등 사고 카테고리라 누운 자세가 실제로
포함됨)로 pose 모델을 fine-tuning한다.

라벨 zip(11MB, 압축된 JSON)과 원본 이미지 zip(15~16GB)은 따로 있다 — 이미지
zip 전체를 풀면 디스크/시간이 크게 들어서, 필요한 이미지만 zipfile로 선택
추출한다(전체 압축 해제 없이 특정 멤버만 꺼냄).

바운딩박스가 라벨에 별도로 없어서 보이는(visibility>0) keypoint의 min/max에
여유(padding)를 줘서 근사한다 — keypoint 자체가 GT라 이 근사가 pose 학습에
큰 문제가 되지 않는다(박스 살짝 크거나 작아도 keypoint 회귀 학습엔 이 라벨의
keypoint 좌표가 핵심이므로).
"""

from __future__ import annotations

import argparse
import random
import zipfile
from pathlib import Path

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    "falling": "1.떨어짐",  # 떨어짐(고소작업 낙상)
    "trip": "3.넘어짐",     # 넘어짐
}
OUTPUT_DIR = REPO_ROOT / "data/processed/pose_finetune"
N_KEYPOINTS = 16
BBOX_PADDING_RATIO = 0.15


def _bbox_from_keypoints(points: list[list[float]], img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    visible = [(x, y) for x, y, v in points if v > 0]
    if len(visible) < 4:
        return None
    xs = [p[0] for p in visible]
    ys = [p[1] for p in visible]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pad_x = (x2 - x1) * BBOX_PADDING_RATIO
    pad_y = (y2 - y1) * BBOX_PADDING_RATIO
    x1, x2 = max(0, x1 - pad_x), min(img_w, x2 + pad_x)
    y1, y2 = max(0, y1 - pad_y), min(img_h, y2 + pad_y)
    return x1, y1, x2, y2


def _to_yolo_pose_line(points: list[list[float]], bbox: tuple[float, float, float, float],
                        img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    for x, y, v in points:
        parts.extend([f"{x/img_w:.6f}", f"{y/img_h:.6f}", str(int(v))])
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=list(CATEGORIES), required=True)
    parser.add_argument("--n-samples", type=int, default=200,
                         help="작게 시작(사용자 요청) — 기본 200장(160 train + 40 val)")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    category_name = CATEGORIES[args.category]
    label_zip_path = REPO_ROOT / f"data/raw/ppe_construction_aihub163/keypoints/train/labels/{category_name}.zip"
    image_zip_path = (REPO_ROOT / "data/raw/ppe_construction_aihub163/keypoints/train/source"
                       / f"105.공사현장_안전장비_인식_데이터/01.데이터/0.키포인트/1.Tranining/원천데이터(zip)/{category_name}.zip")

    import json
    with zipfile.ZipFile(label_zip_path) as label_zip:
        all_label_names = [n for n in label_zip.namelist() if n.endswith(".json")]
        random.shuffle(all_label_names)

        for split_dir in ("images/train", "images/val", "labels/train", "labels/val"):
            (OUTPUT_DIR / split_dir).mkdir(parents=True, exist_ok=True)

        n_val = int(args.n_samples * args.val_ratio)
        written = {"train": 0, "val": 0}

        with zipfile.ZipFile(image_zip_path) as image_zip:
            image_names = set(image_zip.namelist())
            for label_name in all_label_names:
                if written["train"] + written["val"] >= args.n_samples:
                    break
                with label_zip.open(label_name) as f:
                    data = json.load(f)
                img_filename = data["image"]["filename"]
                if img_filename not in image_names:
                    continue
                img_w, img_h = data["image"]["resolution"]
                annotations = data["annotations"]
                if not annotations:
                    continue
                points = annotations[0]["point"]
                if len(points) != N_KEYPOINTS:
                    continue
                bbox = _bbox_from_keypoints(points, img_w, img_h)
                if bbox is None:
                    continue

                split = "val" if written["val"] < n_val else "train"
                stem = Path(img_filename).stem

                with image_zip.open(img_filename) as img_f:
                    img_bytes = img_f.read()
                (OUTPUT_DIR / f"images/{split}/{img_filename}").write_bytes(img_bytes)

                line = _to_yolo_pose_line(points, bbox, img_w, img_h)
                (OUTPUT_DIR / f"labels/{split}/{stem}.txt").write_text(line + "\n", encoding="utf-8")

                written[split] += 1
                if sum(written.values()) % 20 == 0:
                    print(f"진행: train={written['train']} val={written['val']}")

    dataset_yaml = {
        "path": str(OUTPUT_DIR),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [N_KEYPOINTS, 3],
        "names": {0: "person"},
    }
    (OUTPUT_DIR / "dataset.yaml").write_text(
        yaml.dump(dataset_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print(f"완료: train={written['train']}장, val={written['val']}장")
    print(f"dataset.yaml: {OUTPUT_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
