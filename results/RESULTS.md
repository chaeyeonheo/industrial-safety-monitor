# 실행 결과 (실측치만 기록, 가상 수치 없음)

## Phase 1: 탐지 + 추적 공유 백본

**실행 명령**: `python scripts/demo_tracking.py`
**실행 환경**: RTX 4070 Laptop (8GB VRAM), CUDA 12.4, torch 2.5.1, ultralytics 8.4.87
**모델**: `weights/yolo11n.pt` (COCO 사전학습, person class만 필터링)

### 정지 이미지 sanity check (ultralytics 내장 `bus.jpg`, conf_threshold=0.25)

실제 실행 결과 (2026-07-05):

| track_id | bbox (x1,y1,x2,y2) | confidence |
|---|---|---|
| 0 | (671.0, 394.8, 809.8, 878.7) | 0.888 |
| 1 | (47.4, 399.6, 239.3, 904.2) | 0.878 |
| 2 | (223.1, 408.7, 344.5, 860.4) | 0.856 |
| 3 | (0.0, 556.1, 68.9, 872.4) | 0.622 |

이미지 내 실제 사람 4명을 모두 탐지함(오버레이: `results/figures/tracking_demo_bus.png`).
이 이미지는 정지 이미지라 프레임 간 트랙 ID 일관성 검증에는 사용할 수 없음.

### 비디오(프레임 시퀀스) 기반 트랙 ID 일관성 검증

**실행 완료** (2026-07-05). AIHub 163 키포인트 Validation 원천데이터 `3.넘어짐.zip`에서
동일 영상(`path=S2-N6001M00001`)에 속한 프레임 443장을 파일명(프레임 번호) 순으로 정렬해
`PersonTracker.track_stream(list[str])`에 순서대로 입력.

```
python scripts/demo_tracking.py --source data/raw/ppe_construction_aihub163/keypoints/val/source/_frames_S2N6001
```

**실측 결과**: 443프레임 처리, 서로 다른 track_id **38개** 발생(오버레이 영상:
`results/figures/tracking_demo_video.mp4`). track_id 대부분이 1~20프레임의 짧은 구간만
유지되고 끊김.

**원인 분석(실측, 최초 분석)**: 이 zip의 라벨 JSON을 프레임 번호로 분석한 결과, 같은 영상
내에서도 라벨링된 프레임 간 간격이 최소 1, 중앙값 7, 최대 60프레임으로 불균일하게
듬성듬성 샘플링되어 있었음. 프레임 사이 사람의 이동량이 IoU 기반 ByteTrack이 가정하는
프레임 간 근접성을 자주 벗어나 트랙이 끊어진 것으로 추정.

**⚠️ 정정 (2026-07-05, 전수 분석 이후)**: 위 분석은 validation 셋의 작은 서브셋 하나
(`3.넘어짐.zip`의 `path=S2-N6001M00001`, 443프레임)만 본 것이었다. `scripts/analyze_keypoint_labels.py`로
**train 셋 4개 카테고리 전체(95,798개 파일)를 전수 분석**한 결과는 다르다:

| 카테고리 | 파일 수 | 영상 그룹 수 | 그룹 크기(최소/중앙값/최대) | 프레임 간격(최소/중앙값/최대) |
|---|---|---|---|---|
| 떨어짐 | 23,840 | 7 | 14 / 3986 / 4039 | 1 / **1** / 12 |
| 부딪힘 | 23,990 | 8 | 1479 / 3002 / 4011 | 1 / **1** / 6 |
| 넘어짐 | 23,899 | 7 | 3145 / 3410 / 3666 | 1 / **1** / 7 |
| 물체에 맞음 | 24,069 | 10 | 831 / 2041 / 6929 | 1 / **1** / 7 |

**train 셋은 프레임 간격 중앙값이 1(거의 연속 프레임)이고 그룹당 수천 프레임에 달한다.**
즉 처음 관찰한 "불균일한 희소 샘플링"은 train 셋 전반의 특성이 아니라, 내가 트래킹
검증에 썼던 validation 서브셋 하나에 국한된 예외였다. Phase 1 트래킹 검증 결과(38개
track으로 파편화)는 여전히 사실이지만, 그 원인이 "AIHub 데이터 전체의 특성"이라고
일반화한 부분은 부정확했음 — **train 셋 연속 영상으로는 재검증 필요.**
프레임 gap>1인 지점은 실제 장면 전환/촬영 끊김일 가능성이 높으므로,
`convert_aihub163_keypoints_to_pyskl.py`에서는 이 지점을 시퀀스 경계로 분리해 슬라이딩
윈도우를 구성한다(전수 분석 결과 반영).

**추가로 확인된 실측 사실(중요)**: 프레임 150(`results/figures/tracking_demo_frame150.png`)은
작업자가 안전매트 위에 엎드려 쓰러진 자세인데, YOLO11n이 사람으로 전혀 탐지하지 못해
바운딩박스가 아예 없다. 반면 프레임 50(`results/figures/tracking_demo_frame50.png`)처럼
서 있는 자세는 안정적으로 탐지된다(안전벨트·헬멧 착용 작업자, track_id=9). 즉 **일반 사람
탐지기는 넘어진/누운 자세에서 recall이 급격히 떨어질 수 있다**는 것을 실측으로 확인함.
이는 지시문의 2단계(Stage A 휴리스틱 → Stage B 포즈 분류) 설계가 필요한 이유를 실증하는
근거이며, 동시에 Stage A 트리거 자체도 "탐지가 끊기는 순간"을 낙상 신호로 활용할 수 있다는
설계 아이디어로 이어진다(Phase 2에서 반영 검토).

## Phase 2: 낙상 감지 — Stage A 휴리스틱 트리거

**실행 명령**: `python scripts/demo_fall_trigger.py`
**실행 환경**: RTX 4070 Laptop GPU (device=0, 코드에서 명시적으로 확인 출력), CUDA 12.4
**입력**: 위와 동일한 AIHub 낙상 시퀀스 443프레임(`_frames_S2N6001`)

`src/fall_detection/heuristic_trigger.py`의 `FallHeuristicTrigger`는 세 가지 독립 신호로
Stage B(포즈 분류기) 실행 여부를 결정한다: (1) aspect_ratio_delta_threshold=0.5 초과,
(2) vertical_velocity_threshold=200px/s 초과, (3) track_loss(10프레임 이상 추적되던 track이
2프레임 연속 사라짐). (3)은 위에서 실측한 "쓰러진 자세 미탐지" 문제 때문에 추가한 신호이며,
화면 가장자리 근처(`frame_edge_margin_ratio=0.08`)에서 사라진 경우 confidence_hint를
0.5→0.2로 낮춰 "화면 이탈" 오탐 가능성을 표시한다.

**실측 결과** (fps=5.0 가정, 라벨용 희소 프레임이라 실제 fps 불명 — 보수적 가정치):

| frame | track_id | reason | confidence_hint | near_frame_edge |
|---|---|---|---|---|
| 1 | 1 | vertical_velocity_spike | 0.35 | False |
| 19 | 1 | track_lost | 0.20 | **True** |
| 404 | 114 | vertical_velocity_spike | 0.31 | False |

443프레임 중 트리거 3건만 발생. frame=19의 track_lost는 `near_frame_edge=True`로 정확히
가장자리 근처였음을 필터가 검출해 confidence_hint를 낮춤(의도대로 동작 확인).

**한계(실측 기반)**: Phase 1에서 확인했듯 이 프레임 시퀀스는 38개의 짧은 track으로
파편화되어 있어(대부분 20프레임 미만 지속), track_loss_min_history_frames=10 조건을
만족하는 track 자체가 드물다 — 즉 이 트리거 세트는 **연속 영상에서 재검증이 필요**하며,
현재 수치는 "코드가 정상 동작한다"는 근거이지 최종 성능 지표가 아니다. Precision/Recall/
F1/Time-to-Detection 등은 Stage B(포즈 분류기) 구현 및 라벨 매칭 이후 `scripts/evaluate_fall.py`에서
측정 예정.
