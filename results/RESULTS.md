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

## Phase 2: pyskl 변환 스크립트 — 실측 시행착오와 최종 결과

**실행 명령**: `python scripts/convert_aihub163_keypoints_to_pyskl.py`

**1차 시도** (`max_gap_for_continuity=1`, 끊김 허용 없음): 실제로 돌려보니 그룹당
최대 4000여 프레임이 **median 4프레임짜리 조각 4795개**로 파편화됨을 확인(전체
4개 카테고리 중 falling_from_height 기준 실측). window_length=30 조건을 만족하는
런이 카테고리별로 7~10개(영상 그룹 수와 거의 같음)뿐이라 **총 30개 윈도우**만
생성됨 — 학습에 쓰기엔 턱없이 부족한 양이었다.

**원인 분석**: 중앙값 gap=1이라는 통계만으로는 "연속"이라 오판하기 쉬우나, 실제로는
2~3프레임짜리 짧은 끊김(사람이 잠깐 프레임에서 사라지는 순간 — Phase 1에서 확인한
탐지 실패와 같은 원인으로 추정)이 매우 잦아 `max_gap_for_continuity=1` 기준으로는
런이 잘게 쪼개짐.

**개선**: `max_gap_for_continuity`를 1/3/5/10으로 바꿔가며 런 길이 분포를 실측 비교:

| max_gap | n_runs | median | p90 | max |
|---|---|---|---|---|
| 1 | 4795 | 4 | 11 | 38 |
| **3** | 224 | **61** | 269 | 681 |
| 5 | 30 | 28 | 3723 | 4039 |
| 10 | 9 | 3950 | 4039 | 4039 |

gap=5 이상은 실제 장면 전환까지 이어붙일 위험이 커 보수적으로 **gap=3**을 채택하고,
끊긴 구간은 선형 보간(`densify_run`)으로 채우며 보간된 프레임은 `keypoint_score=0`으로
마킹해 실측과 구분되게 했다.

**최종 실행 결과** (`max_gap_for_continuity=3`, `window_length=30`, `window_stride=15`):

| 카테고리 | 영상 그룹 수 | 생성된 윈도우 수 |
|---|---|---|
| falling_from_height (떨어짐) | 7 | 1663 |
| struck_by_collision (부딪힘) | 8 | 1698 |
| trip_and_fall (넘어짐) | 7 | 1647 |
| struck_by_object (물체에 맞음) | 10 | 1679 |
| **합계** | 32 | **6687** |

생성된 `data/processed/fall_keypoints/train_windows.pkl`을 다시 로드해 검증:
`keypoint` shape `(1, 30, 16, 2)`, `keypoint_score` shape `(1, 30, 16)`, 클래스
분포 거의 균등(1647~1698개/클래스). pyskl `PoseDataset` 포맷 요구사항(frame_dir,
label, img_shape, original_shape, total_frames, keypoint, keypoint_score) 충족 확인.

**미해결**: AIHub 16-keypoint가 COCO-17 등 표준 스켈레톤 레이아웃과 어떻게
대응되는지 공식 문서를 아직 확보하지 못해, pyskl 학습에 필요한 Graph(인접행렬)
설정은 아직 만들지 못함 — 원본 16개 인덱스를 그대로 보존만 해둔 상태.

**⚠️ v1의 근본 한계 (사용자 지적, 2026-07-06 실측으로 확인)**: v1은 영상 하나
전체(최대 4000+프레임)를 기계적으로 30프레임씩 잘라 그 영상의 폴더 카테고리
라벨을 그대로 붙인다. 그런데 `S2-N4601M00001`(3995프레임)의 keypoint bbox
가로/세로 비율을 프레임 순서대로 10구간으로 나눠 보면:

```
구간0: 0.96  구간1: 0.83  구간2: 0.94  구간3: 1.00  구간4: 0.83
구간5: 0.86  구간6: 0.87  구간7: 0.90  구간8: 1.00  구간9: 0.90
```

영상 전체에 걸쳐 고르게 섞여 있고 "누움"(ratio>1.3)으로 볼 수 있는 프레임은
**18.3%뿐**이다. 즉 영상 하나에 낙상 동작이 여러 번 반복 시연되고 나머지는 서
있거나 회복하는 구간인데, 라벨 JSON에는 프레임 단위 상태 필드가 없다
(`data ID`, `middle classification`, `class`, `point` 네 개뿐이고 `class`는
프레임마다 바뀌는 게 아니라 영상 하나 전체에서 고정값). 그래서 v1의 윈도우
상당수는 "그냥 서 있는 모습"인데 "낙상"으로 잘못 라벨링됐을 가능성이 높다.
상세 원인 분석과 대안 설계는 `docs/data_preprocessing.md` 참고.

## Phase 2: pyskl 변환 스크립트 v2 — 전환 감지 기반 (v1과 별도 보존, ablation용)

**실행 명령**: `python scripts/convert_aihub163_keypoints_to_pyskl_transition.py`

v1의 한계를 보완하기 위해 Stage A 휴리스틱(`aspect_ratio_delta_threshold=0.5`,
`window_frames=15`, `src/fall_detection/heuristic_trigger.py`와 동일 임계값)을
**정답 keypoint 시퀀스에 직접 적용**해 영상 안에서 실제 "서 있다가 눕는" 전환
순간을 자동 검출하고, 그 주변만 양성(낙상) 윈도우로, 전환에서 먼 구간을
음성("normal", 낙상 아님) 윈도우로 추출했다.

**실측 결과**:

| 카테고리 | 검출된 전환 수 | 양성 윈도우 | 음성(normal) 윈도우 |
|---|---|---|---|
| falling_from_height | 581 | 581 | 34 |
| struck_by_collision | 164 | 164 | 559 |
| trip_and_fall | 568 | 568 | 49 |
| struck_by_object | 193 | 193 | 567 |
| **합계** | 1506 | **1506** | **1209** |

총 2715개 윈도우 (`data/processed/fall_keypoints_transition/train_windows_transition.pkl`,
검증 완료: `keypoint` shape `(1, 30, 16, 2)`).

**흥미로운 실측 패턴**: falling_from_height/trip_and_fall은 전환이 카테고리당
500건 이상 검출된 반면, struck_by_collision/struck_by_object는 164~193건뿐이다.
"bbox 종횡비가 급격히 커진다(서 있다가 눕는다)"는 신호가 낙상·넘어짐에는 잘
맞지만, 부딪힘·물체에맞음은 반드시 눕는 자세로 이어지지 않을 수 있어(맞고
휘청이거나 웅크리는 정도로 끝날 수 있음) 이 신호로는 잘 안 잡힐 수 있다는
뜻이다. **즉 이 v2 방식도 카테고리에 따라 검출 편향이 있을 수 있어, v1과의
다운스트림 분류 성능 비교(ablation)가 필요하다** — 사용자가 시간 여유가 되면
진행하기로 함.

## Phase 2: "탐지+추적만" vs "keypoint 전환감지 병행" 비교 (2026-07-06)

**실행 명령**: `python scripts/demo_tracking_vs_transition.py`
같은 낙상 시퀀스(S2-N6001, 443프레임)에 대해 위쪽엔 YOLO11n+ByteTrack bbox
추적 결과를, 아래쪽엔 정답 keypoint + Stage A 방식 전환감지(종횡비 급변,
`aspect_ratio_delta_threshold=0.5`, `window_frames=15`, Stage A와 동일 임계값)
결과를 나란히 오버레이했다.

**실측 결과**:
- 전체 443프레임 중 **224프레임(약 50.6%)에서 YOLO11n 탐지 자체가 실패**함
  (Phase 1에서 확인한 문제가 이 영상 전체로 보면 훨씬 심각하다는 게 드러남).
- 그중 **29프레임은 keypoint 기반 전환감지가 대신 "낙상 전환"으로 정확히
  잡아냄** (`results/figures/compare_tracking_miss_transition_catches.png`
  참고 — 사람이 매트에 완전히 누워 YOLO bbox는 "DETECTION LOST"인데, 같은
  프레임의 정답 keypoint로 계산한 ratio=3.09로 "FALL TRANSITION DETECTED"가
  뜬다).
- 나머지 195프레임의 탐지 실패는 전환 시점과 무관하다 — 즉 낙상 때문이
  아니라 다른 이유(각도, 부분 가려짐 등)로 탐지가 실패하는 경우가 더 많다는
  뜻이며, 이는 별도로 조사가 필요한 사실로 남겨둔다.

**해석에 유의**: 이 비교의 아래쪽은 "우리 파이프라인이 실시간으로 pose를
추출해 판단한 것"이 아니라 **AIHub 정답 keypoint에 Stage A 로직을 적용한
것**이다. 즉 "keypoint 신호(정답이든, 향후 실제 pose 추출기의 출력이든)가
있으면 bbox가 사라지는 구간에서도 낙상을 감지할 수 있다"는 근거를 보여주는
것이며, 아직 실시간 pose 추출기를 통합한 최종 비교는 아니다(pose 추출기는
Stage B 구현 시 추가 예정). 전체 비교 영상은 `outputs/tracking_vs_transition.mp4`
(로컬에만 존재, git 미포함).

## Phase 3: PPE(안전보호구) YOLO 파인튜닝 (2026-07-06)

**데이터**: `scripts/convert_aihub163_to_yolo.py`로 물류센터(반도체클러스터) 라벨 중
이미지가 있는 21,037프레임에서 2,000개 샘플 추출. 662프레임에 PPE bbox 존재,
클래스당 380~451개(helmet 395, vest 380, harness 451, safety_shoes 421).
train/val = 1800/200 (9:1).

**학습 명령**: `python scripts/train_ppe_yolo.py --epochs 20 --batch 16`
**환경**: RTX 4070 Laptop, YOLO11n, imgsz=640. GPU 메모리 사용량 **2.25GB/8GB**로 확인.
총 학습 시간 **약 7.1분**(20 epoch, epoch당 ~21초).

**실측 결과 (val, 200장)**:

| 클래스 | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| helmet | 0.698 | 0.839 | 0.804 | 0.456 |
| vest | 0.874 | 0.633 | 0.782 | 0.446 |
| harness | 0.710 | 0.694 | 0.729 | 0.319 |
| safety_shoes | 0.684 | 0.804 | 0.811 | 0.503 |
| **전체** | 0.741 | 0.742 | 0.781 | 0.431 |

시각적으로도 실제 작업자 사진에서 헬멧/조끼/안전화/하네스가 합리적인 confidence로
잡힘을 확인(`results/figures/ppe_yolo_val_predictions.png`, 학습 곡선은
`results/figures/ppe_yolo_training_curves.png`). 2,000프레임(그중 662개만 실제
라벨 존재)·20 epoch짜리 1차 실험치고는 양호한 수치.

**한계**:
- 클래스 매핑이 공식 문서 없이 실측 추론한 것이라(`docs/ppe_class_mapping.md`)
  harness/safety_shoes 이름이 틀렸을 가능성이 있음(helmet/vest는 시각적으로 고신뢰).
- 훈련 프레임이 반도체클러스터 촬영지 하나뿐이라 다른 현장(화물터미널E/E2 등)에
  대한 일반화는 검증 안 됨.

**추가(2026-07-06)**: "미착용" 판정(간접 연결)을 `src/ppe_detection/indirect_association.py`로
구현 완료. 전체 bbox IoU 대신 "보호구 bbox 중심이 사람 bbox 안에 있고, 신체
부위별 기대 y범위에 맞는지"로 매칭(헬멧처럼 작은 부분 박스는 전체 IoU가 항상
0에 가까워 무의미하기 때문). 실제 val 이미지로 확인: 헬멧/조끼/안전화를 착용한
작업자에서 하네스만 정확히 "미착용"으로 판정됨(`results/figures/ppe_compliance_*.png`).

## Phase 2: HD-GCN 학습 및 ablation (2026-07-06)

**데이터**: v2(전환감지, `data/processed/fall_keypoints_transition/train_windows_transition.pkl`,
2,715개 윈도우, 5클래스). **학습 명령**: `python scripts/train_hdgcn_fall.py --epochs 15 --batch 32`
**환경**: RTX 4070 Laptop, GPU 메모리 사용량 1.2GB/8GB(여유 충분). 학습 세부사항과
겪은 이슈(einops 누락, 업스트림 저장소의 누락된 그래프 모듈, AHA 모듈의 하드코딩된
NTU 그래프 호출 monkeypatch)는 `docs/PROGRESS.md` "문제 4" 참고.

**실측 결과 (val 407개)**: 전체 정확도 **81.8%** (최고 epoch10 기준 83.8%).
클래스별 recall: falling_from_height 0.827, struck_by_collision 0.545,
trip_and_fall 0.798, struck_by_object 0.633, normal 0.886.

**Ablation 3 핵심 발견**: struck_by_collision/struck_by_object의 recall이
확실히 낮은데(0.55~0.63), 이는 Ablation 2-보충에서 발견한 "이 두 카테고리는
Stage A 전환 검출 자체가 적었다(164~193건 vs 낙상/넘어짐 500+건)"는 패턴이
다운스트림 분류 성능까지 그대로 이어진 것 — 데이터 구성 단계의 약점이
학습 결과에 그대로 반영됨을 실측으로 확인. 전체 ablation 정리는
`docs/ablation_studies.md` 참고.
