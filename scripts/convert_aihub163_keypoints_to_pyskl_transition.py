"""AIHub 163 키포인트 라벨 → pyskl pickle 변환 (전환 감지 기반, v2).

`scripts/convert_aihub163_keypoints_to_pyskl.py`(v1, 기계적 슬라이딩 윈도우)의 한계를
보완한 버전. 두 버전을 **둘 다 남겨서 나중에 ablation 비교**한다(사용자 요청).

## v1의 문제 (실측으로 확인, docs/data_preprocessing.md 참고)

AIHub 라벨 zip은 "이 영상 폴더 전체가 낙상 카테고리"라는 폴더 단위 라벨만 있고,
프레임 단위로 "지금 넘어지는 중"인지 "그냥 서 있는 중"인지 알려주는 필드가 없다.
실측 결과 하나의 영상(~4000프레임)에서 실제로 "누운 자세"(keypoint bbox 가로/세로
비율 > 1.3)로 볼 수 있는 프레임은 18.3%뿐이었다. v1처럼 영상 전체를 기계적으로
30프레임씩 잘라 전부 "낙상" 라벨을 붙이면, 그중 상당수가 실제로는 "서 있는 모습"인데
"낙상"으로 잘못 라벨링된다.

## v2의 해결 방식

Stage A 휴리스틱(`src/fall_detection/heuristic_trigger.py`)에서 쓴 것과 같은
"bbox(여기서는 keypoint bbox) 종횡비 급변" 신호를 **정답 keypoint 시퀀스에 그대로
적용**해서, 영상 안에서 실제로 서 있다가 넘어지는 전환 순간을 자동으로 찾는다.

1. 각 연속 구간(run, v1과 동일한 gap-bridging 재사용)에서 프레임별
   `ratio = (keypoint bbox 가로) / (keypoint bbox 세로)`를 계산.
2. `window_frames`(Stage A와 동일하게 기본 15) 간격을 두고 `ratio`가
   `aspect_ratio_delta_threshold`(Stage A와 동일하게 기본 0.5) 이상 증가하는 지점을
   "전환(transition)"으로 표시.
3. 전환 지점 주변 `window_length`프레임을 **양성(해당 카테고리) 샘플**로 추출.
4. 전환 지점에서 `negative_exclusion_margin`프레임 이상 떨어진 구간에서
   **음성("normal", 낙상 아님) 샘플**을 추출 — 지금까지 4개 카테고리가 전부
   "이상행동"이라 정상 데이터가 아예 없었는데, 이 방식으로 부산물로 확보된다.

이 방식도 완벽하지 않다: (a) ratio 임계값 하나로 모든 낙상 유형(떨어짐/부딪힘/
넘어짐/물체에맞음)의 전환을 다 잡을 수 있는지는 검증 안 됨(카테고리별로 실제
포착률을 비교해볼 것), (b) 여전히 사람이 직접 라벨링한 "진짜 전환 시점"이 아니라
휴리스틱 추정치. 정확한 비교는 v1/v2 각각으로 학습한 분류기의 다운스트림 성능으로
판단해야 한다(ablation).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_aihub163_keypoints_to_pyskl import (  # noqa: E402
    CATEGORIES,
    LABELS_DIR,
    OUTPUT_DIR as V1_OUTPUT_DIR,
    densify_run,
    load_category_frames,
    split_into_continuous_runs,
)

OUTPUT_DIR = V1_OUTPUT_DIR.parent / "fall_keypoints_transition"
LABEL_TO_INT = {name: i for i, name in enumerate(CATEGORIES.values())}
NORMAL_LABEL_ID = len(CATEGORIES)  # falling(0)/collision(1)/trip(2)/struck_by_object(3) 다음 = 4


def compute_ratio_series(run: list[tuple[int, np.ndarray, tuple[int, int]]]) -> np.ndarray:
    ratios = np.empty(len(run), dtype=np.float32)
    for i, (_frame_num, kp, _res) in enumerate(run):
        xs, ys = kp[:, 0], kp[:, 1]
        w = xs.max() - xs.min()
        h = ys.max() - ys.min()
        ratios[i] = w / max(h, 1e-6)
    return ratios


def find_transition_indices(ratios: np.ndarray, window_frames: int,
                             aspect_ratio_delta_threshold: float) -> list[int]:
    """Stage A와 동일한 신호(구간 시작·끝 ratio 증가폭)로 전환 후보 프레임 인덱스를 찾는다.
    연속된 후보는 하나의 이벤트로 묶어 중앙 인덱스만 남긴다."""
    raw_hits = []
    for t in range(window_frames, len(ratios)):
        delta = ratios[t] - ratios[t - window_frames]
        if delta >= aspect_ratio_delta_threshold:
            raw_hits.append(t)

    if not raw_hits:
        return []

    events: list[list[int]] = [[raw_hits[0]]]
    for t in raw_hits[1:]:
        if t - events[-1][-1] <= window_frames:
            events[-1].append(t)
        else:
            events.append([t])
    return [group[len(group) // 2] for group in events]


def extract_positive_windows(run, transition_indices: list[int], window_length: int):
    half = window_length // 2
    windows = []
    for center in transition_indices:
        start = max(0, center - half)
        end = start + window_length
        if end > len(run):
            end = len(run)
            start = max(0, end - window_length)
        if end - start == window_length:
            windows.append(run[start:end])
    return windows


def extract_negative_windows(run, transition_indices: list[int], window_length: int,
                              negative_exclusion_margin: int, max_negatives: int):
    excluded = np.zeros(len(run), dtype=bool)
    for center in transition_indices:
        lo = max(0, center - negative_exclusion_margin)
        hi = min(len(run), center + negative_exclusion_margin)
        excluded[lo:hi] = True

    windows = []
    start = 0
    while start + window_length <= len(run) and len(windows) < max_negatives:
        if not excluded[start:start + window_length].any():
            windows.append(run[start:start + window_length])
            start += window_length
        else:
            start += 1
    return windows


def window_to_pyskl_sample(window, label_id: int, frame_dir: str) -> dict:
    kps = np.stack([kp[:, :2] for _, kp, _ in window])
    scores = np.stack([kp[:, 2] for _, kp, _ in window])
    width, height = window[0][2]
    return {
        "frame_dir": frame_dir,
        "label": label_id,
        "img_shape": (height, width),
        "original_shape": (height, width),
        "total_frames": kps.shape[0],
        "keypoint": kps[None, ...].astype(np.float16),
        "keypoint_score": scores[None, ...].astype(np.float16),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-length", type=int, default=30)
    parser.add_argument("--max-gap-for-continuity", type=int, default=3)
    parser.add_argument("--stage-a-window-frames", type=int, default=15)
    parser.add_argument("--aspect-ratio-delta-threshold", type=float, default=0.5)
    parser.add_argument("--negative-exclusion-margin", type=int, default=45)
    parser.add_argument("--max-negatives-per-run", type=int, default=20)
    args = parser.parse_args()

    if not LABELS_DIR.exists():
        print(f"[convert-v2] 라벨 디렉토리가 없습니다: {LABELS_DIR}")
        return

    samples = []
    transition_counts = {}
    for zip_name, label_name in CATEGORIES.items():
        zip_path = LABELS_DIR / zip_name
        if not zip_path.exists():
            print(f"[convert-v2] SKIP {zip_name} (파일 없음)")
            continue

        groups = load_category_frames(zip_path)
        n_pos, n_neg, n_transitions = 0, 0, 0
        for path_id, frames in groups.items():
            runs = split_into_continuous_runs(frames, args.max_gap_for_continuity)
            for run_idx, run in enumerate(runs):
                dense_run = densify_run(run) if len(run) > 1 else run
                if len(dense_run) < args.window_length:
                    continue
                ratios = compute_ratio_series(dense_run)
                transitions = find_transition_indices(
                    ratios, args.stage_a_window_frames, args.aspect_ratio_delta_threshold)
                n_transitions += len(transitions)

                for i, w in enumerate(extract_positive_windows(dense_run, transitions, args.window_length)):
                    frame_dir = f"{label_name}/{path_id}/run{run_idx}_transition{i}"
                    samples.append(window_to_pyskl_sample(w, LABEL_TO_INT[label_name], frame_dir))
                    n_pos += 1

                for i, w in enumerate(extract_negative_windows(
                        dense_run, transitions, args.window_length,
                        args.negative_exclusion_margin, args.max_negatives_per_run)):
                    frame_dir = f"normal/{path_id}/run{run_idx}_neg{i}"
                    samples.append(window_to_pyskl_sample(w, NORMAL_LABEL_ID, frame_dir))
                    n_neg += 1

        transition_counts[label_name] = n_transitions
        print(f"[convert-v2] {label_name}: 전환 감지 {n_transitions}건 → 양성 윈도우 {n_pos}개, 음성(정상) 윈도우 {n_neg}개")

    if not samples:
        print("[convert-v2] 생성된 샘플이 없습니다.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "train_windows_transition.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(samples, f)

    from collections import Counter
    label_dist = Counter(s["label"] for s in samples)
    print(f"\n[convert-v2] 총 {len(samples)}개 윈도우를 {out_path}에 저장")
    print(f"[convert-v2] 클래스 분포(0~3=낙상 카테고리, {NORMAL_LABEL_ID}=normal): {dict(label_dist)}")


if __name__ == "__main__":
    main()
