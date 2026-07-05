"""AIHub 163 물류센터 PPE bbox 라벨 → YOLO txt 포맷 변환.

`docs/ppe_class_mapping.md`에서 실제 이미지로 확인한 클래스 매핑을 사용한다
(공식 문서 없이 추론한 best-effort, 확정 아님). "안전보호구 착용 여부 전반"을
탐지하는 게 목적이라 헬멧 하나만이 아니라 신뢰도 있게 확인된 4개 클래스
(안전모/안전조끼/안전벨트/안전화)를 모두 추출한다. 03/04는 우리가 다운로드한
촬영지 이미지에 아예 없고(다른 촬영지 라벨로 추정), 06/08은 너무 드물고
시각적으로도 확신이 낮아 제외했다.

"사람" 클래스는 Phase 1의 YOLO11n(COCO person)이 이미 담당하므로 여기서는
만들지 않는다 — 나중에 간접 연결(IoU 매칭, src/ppe_detection/)에서 결합한다.

라벨 zip은 3개 촬영지(반도체클러스터/화물터미널E/E2)를 포함하지만, 원천이미지는
반도체클러스터(파일명 접두사 S2-N1401/1402/1403)만 다운로드해뒀으므로 그 프레임만
사용한다(나머지는 이미지가 없어 학습에 쓸 수 없음).
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_ZIP = REPO_ROOT / ("data/raw/ppe_construction_aihub163/labels/train/"
                         "105.공사현장_안전장비_인식_데이터/01.데이터/1.Training/"
                         "라벨링데이터_241008_add/5.물류센터.zip")
IMAGE_ZIP = REPO_ROOT / ("data/raw/ppe_construction_aihub163/images/train/"
                         "105.공사현장_안전장비_인식_데이터/01.데이터/1.Training/"
                         "원천데이터_210818_add/5.물류센터/"
                         "5.물류센터_반도체_장비_클러스터_신축공사.zip")
OUTPUT_DIR = REPO_ROOT / "data/processed/ppe_yolo"

AVAILABLE_IMAGE_PREFIXES = ("S2-N1401", "S2-N1402", "S2-N1403")
PPE_MIDDLE_CLASS = "01"  # 안전보호구
# (class 코드) -> (YOLO class_id, 이름). docs/ppe_class_mapping.md 참고, best-effort.
PPE_CLASS_MAP: dict[str, tuple[int, str]] = {
    "07": (0, "helmet"),
    "02": (1, "vest"),
    "01": (2, "harness"),
    "05": (3, "safety_shoes"),
}
CLASS_NAMES = [name for _, name in sorted(PPE_CLASS_MAP.values())]


def fix_encoding(name: str) -> str:
    try:
        return name.encode("cp437").decode("cp949")
    except Exception:
        return name


def box_to_yolo_line(box: list[float], img_w: int, img_h: int, class_id: int = 0) -> str:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=2000,
                         help="사용할 최대 프레임 수(빠른 1차 학습을 위해 21,037개 전체 대신 샘플링)")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not LABEL_ZIP.exists() or not IMAGE_ZIP.exists():
        print(f"[convert-ppe] 필요한 파일이 없습니다:\n  {LABEL_ZIP}\n  {IMAGE_ZIP}")
        return

    with zipfile.ZipFile(LABEL_ZIP) as zl:
        names = zl.namelist()
        fixed = [fix_encoding(n) for n in names]
        candidates = [
            raw for raw, n in zip(names, fixed)
            if "5.전체" in n and n.endswith(".json")
            and any(p in n for p in AVAILABLE_IMAGE_PREFIXES)
        ]
        print(f"[convert-ppe] 이미지가 있는 라벨 후보: {len(candidates)}개")

        random.seed(args.seed)
        if len(candidates) > args.max_frames:
            candidates = random.sample(candidates, args.max_frames)
        print(f"[convert-ppe] 실제 사용할 프레임: {len(candidates)}개 (--max-frames={args.max_frames})")

        random.shuffle(candidates)
        n_val = int(len(candidates) * args.val_ratio)
        split_map = {name: ("val" if i < n_val else "train") for i, name in enumerate(candidates)}

        for split in ("train", "val"):
            (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

        n_with_any_ppe = 0
        n_boxes_by_class: dict[str, int] = {name: 0 for name in CLASS_NAMES}
        with zipfile.ZipFile(IMAGE_ZIP) as zi:
            image_names = set(zi.namelist())
            for label_name in candidates:
                with zl.open(label_name) as f:
                    d = json.load(f)
                filename = d["image"]["filename"]
                if filename not in image_names:
                    continue
                img_w, img_h = d["image"]["resolution"]

                lines = []
                for ann in d["annotations"]:
                    if ann.get("box") is None:
                        continue
                    if ann.get("middle classification") != PPE_MIDDLE_CLASS:
                        continue
                    mapped = PPE_CLASS_MAP.get(ann.get("class"))
                    if mapped is None:
                        continue
                    class_id, class_name = mapped
                    lines.append(box_to_yolo_line(ann["box"], img_w, img_h, class_id=class_id))
                    n_boxes_by_class[class_name] += 1

                split = split_map[label_name]
                stem = Path(filename).stem
                if lines:
                    n_with_any_ppe += 1

                with zi.open(filename) as img_f:
                    (OUTPUT_DIR / "images" / split / filename).write_bytes(img_f.read())
                (OUTPUT_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))

    dataset_yaml = {
        "path": str(OUTPUT_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    with open(OUTPUT_DIR / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, allow_unicode=True)

    print(f"[convert-ppe] 총 {len(candidates)}프레임 중 PPE bbox가 하나라도 있는 프레임: {n_with_any_ppe}개")
    print(f"[convert-ppe] 클래스별 bbox 수: {n_boxes_by_class}")
    print(f"[convert-ppe] dataset.yaml 저장: {OUTPUT_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
