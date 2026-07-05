"""AIHub 163 키포인트 라벨(떨어짐/부딪힘/넘어짐/물체에맞음) → pyskl PoseDataset pickle 변환.

pyskl(https://github.com/kennymckormick/pyskl)의 PoseDataset은 각 샘플이
{frame_dir, label, img_shape, original_shape, total_frames, keypoint(M,T,V,C),
keypoint_score(M,T,V)} 형태인 pickle(list[dict])을 요구한다.

전제(scripts/analyze_keypoint_labels.py 전수 분석 + 이 스크립트로 추가 확인한 실측 결과,
results/RESULTS.md 참고):
- train 라벨은 `path`(원본 영상 ID)별로 그룹핑된다. 프레임 간격의 **중앙값**은 1이지만,
  실제로 `max_gap_for_continuity=1`(끊김 허용 없음)로 연속 구간을 나눠보면 그룹당
  최대 4000여 프레임이 **median 4프레임짜리 조각 4795개**로 잘게 파편화된다(사람이
  프레임에서 잠깐씩 사라지는 순간이 매우 잦다는 뜻 — Phase 1에서 확인한 "쓰러진 자세를
  탐지기가 놓친다"는 문제와 같은 원인일 가능성이 높다). 이대로면 학습에 쓸 수 있는 긴
  시퀀스가 거의 안 나온다.
- 그래서 **`max_gap_for_continuity=3`**(2~3프레임짜리 짧은 끊김은 보간해서 이어붙임)을
  기본값으로 쓴다. 이 설정에서는 median 런 길이가 4→61프레임으로 크게 늘어남을 실측
  확인했다(같은 스크립트로 max_gap 1/3/5/10을 비교). 3을 넘는 값(5, 10)은 실제 장면
  전환까지 이어붙일 위험이 커 보수적으로 3을 택함. 끊긴 구간의 keypoint는 선형 보간하고
  `keypoint_score`를 0으로 마킹해 "보간된 프레임"임을 학습 코드가 구분할 수 있게 한다.
- 슬라이딩 윈도우(`window_length`, 기본 30프레임, stride `window_stride` 기본 15)로
  고정 길이 클립을 만든다. 마지막에 남는 window_length 미만 잔여 프레임은 버린다
  (패딩 대신 버리는 쪽을 택함 — 짧은 클립을 패딩하면 정지 프레임처럼 보여 낙상 동작의
  시간적 패턴을 왜곡할 수 있어서 초기 버전에서는 보수적으로 버림. 필요시 패딩 전략으로
  전환 가능).

**미해결 사항**: AIHub 163 keypoint는 COCO-17이 아니라 16개 점이며, 각 점이 해부학적으로
무엇을 의미하는지(어깨/팔꿈치/무릎 등) 알려주는 공식 문서를 아직 확보하지 못했다.
임의로 COCO-17에 대응시키면 잘못된 그래프 연결을 만들 위험이 있어(지시문 5장 규칙:
근거 없는 값을 지어내지 않는다), **원본 16-포인트 순서를 그대로 보존**하고
`joint_0`..`joint_15`라는 이름 없는 인덱스로만 다룬다. pyskl 쪽에서 이 레이아웃에 맞는
커스텀 Graph 설정(`layout='custom_aihub16'`, 인접 행렬 별도 정의)이 필요하며, 이는
AIHub 데이터 가이드 문서를 확보한 뒤 사람이 직접 확인하고 채워야 한다.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS_DIR = REPO_ROOT / "data/raw/ppe_construction_aihub163/keypoints/train/labels"
OUTPUT_DIR = REPO_ROOT / "data/processed/fall_keypoints"

CATEGORIES = {
    "1.떨어짐.zip": "falling_from_height",
    "2.부딪힘.zip": "struck_by_collision",
    "3.넘어짐.zip": "trip_and_fall",
    "4.물체에_맞음.zip": "struck_by_object",
}
LABEL_TO_INT = {name: i for i, name in enumerate(CATEGORIES.values())}

FRAME_NUM_RE = re.compile(r"(\d+)\.jpg$")
N_JOINTS = 16


def load_category_frames(zip_path: Path) -> dict[str, list[tuple[int, np.ndarray, tuple[int, int]]]]:
    """path별로 (frame_num, keypoints(V,3), resolution) 리스트를 모아 반환."""
    groups: dict[str, list[tuple[int, np.ndarray, tuple[int, int]]]] = defaultdict(list)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".json")]
        for name in names:
            with z.open(name) as f:
                d = json.load(f)
            path = d["image"]["path"]
            m = FRAME_NUM_RE.search(d["image"]["filename"])
            if not m:
                continue
            frame_num = int(m.group(1))
            anns = d.get("annotations", [])
            if not anns:
                continue
            points = anns[0].get("point", [])
            if len(points) != N_JOINTS:
                continue  # 포맷이 다른 라벨(예: 다른 keypoint 수)은 스킵
            kp = np.array(points, dtype=np.float32)  # (16, 3) - x, y, visibility
            resolution = (d["image"]["resolution"][0], d["image"]["resolution"][1])
            groups[path].append((frame_num, kp, resolution))
    return groups


def split_into_continuous_runs(frames: list[tuple[int, np.ndarray, tuple[int, int]]],
                                max_gap_for_continuity: int = 3) -> list[list[tuple[int, np.ndarray, tuple[int, int]]]]:
    """gap <= max_gap_for_continuity인 구간은 같은 run으로 묶는다(끊긴 프레임은 아직 채우지 않음,
    densify_run에서 보간). gap이 그보다 크면 실제 장면 전환으로 보고 run을 분리한다."""
    frames = sorted(frames, key=lambda x: x[0])
    runs: list[list[tuple[int, np.ndarray, tuple[int, int]]]] = []
    current: list[tuple[int, np.ndarray, tuple[int, int]]] = []
    prev_frame_num = None
    for frame_num, kp, res in frames:
        if prev_frame_num is not None and frame_num - prev_frame_num > max_gap_for_continuity:
            if current:
                runs.append(current)
            current = []
        current.append((frame_num, kp, res))
        prev_frame_num = frame_num
    if current:
        runs.append(current)
    return runs


def densify_run(run: list[tuple[int, np.ndarray, tuple[int, int]]]) -> list[tuple[int, np.ndarray, tuple[int, int]]]:
    """run 내부의 빠진 프레임 번호를 양 옆 실측 프레임의 선형 보간으로 채운다.
    보간된 프레임은 keypoint의 visibility 채널(index 2)을 0으로 덮어써 실측과 구분한다."""
    dense: list[tuple[int, np.ndarray, tuple[int, int]]] = []
    for (f0, kp0, res0), (f1, kp1, _res1) in zip(run, run[1:]):
        dense.append((f0, kp0, res0))
        gap = f1 - f0
        for step in range(1, gap):
            t = step / gap
            interp = kp0 * (1 - t) + kp1 * t
            interp[:, 2] = 0.0  # 보간된 프레임은 score=0으로 마킹
            dense.append((f0 + step, interp, res0))
    dense.append(run[-1])
    return dense


def make_windows(run: list[tuple[int, np.ndarray, tuple[int, int]]],
                  window_length: int, window_stride: int):
    for start in range(0, len(run) - window_length + 1, window_stride):
        yield run[start:start + window_length]


def window_to_pyskl_sample(window, label_name: str, frame_dir: str) -> dict:
    kps = np.stack([kp[:, :2] for _, kp, _ in window])       # (T, V, 2)
    scores = np.stack([kp[:, 2] for _, kp, _ in window])      # (T, V) - AIHub visibility(0이면 보간된 프레임)
    width, height = window[0][2]
    return {
        "frame_dir": frame_dir,
        "label": LABEL_TO_INT[label_name],
        "img_shape": (height, width),
        "original_shape": (height, width),
        "total_frames": kps.shape[0],
        "keypoint": kps[None, ...].astype(np.float16),        # (M=1, T, V, C=2)
        "keypoint_score": scores[None, ...].astype(np.float16),  # (M=1, T, V)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-length", type=int, default=30)
    parser.add_argument("--window-stride", type=int, default=15)
    parser.add_argument("--max-gap-for-continuity", type=int, default=3)
    args = parser.parse_args()

    if not LABELS_DIR.exists():
        print(f"[convert] 라벨 디렉토리가 없습니다: {LABELS_DIR}")
        print("[convert] AIHub 163 키포인트 train 라벨을 받아 이 경로에 배치한 뒤 다시 실행하세요.")
        return

    samples = []
    for zip_name, label_name in CATEGORIES.items():
        zip_path = LABELS_DIR / zip_name
        if not zip_path.exists():
            print(f"[convert] SKIP {zip_name} (파일 없음 — {LABELS_DIR}에 배치 필요)")
            continue

        groups = load_category_frames(zip_path)
        n_windows_this_category = 0
        for path_id, frames in groups.items():
            runs = split_into_continuous_runs(frames, args.max_gap_for_continuity)
            for run_idx, run in enumerate(runs):
                dense_run = densify_run(run) if len(run) > 1 else run
                for win_idx, window in enumerate(make_windows(dense_run, args.window_length, args.window_stride)):
                    frame_dir = f"{label_name}/{path_id}/run{run_idx}_win{win_idx}"
                    samples.append(window_to_pyskl_sample(window, label_name, frame_dir))
                    n_windows_this_category += 1
        print(f"[convert] {label_name}: {len(groups)}개 영상 그룹 → {n_windows_this_category}개 윈도우")

    if not samples:
        print("[convert] 생성된 샘플이 없습니다. 라벨 데이터를 확인하세요.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "train_windows.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(samples, f)
    print(f"[convert] 총 {len(samples)}개 윈도우를 {out_path}에 저장 (window_length={args.window_length}, stride={args.window_stride})")
    print("[convert] 주의: 16-포인트가 COCO-17 등 표준 레이아웃과 어떻게 대응되는지 공식 문서를 "
          "아직 확보하지 못해 원본 인덱스를 그대로 보존했습니다. pyskl 학습 전 커스텀 Graph 설정이 필요합니다.")


if __name__ == "__main__":
    main()
