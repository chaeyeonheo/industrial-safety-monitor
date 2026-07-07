# Ablation 3종 정리 (발표용)

이 문서는 프로젝트 전체에서 실제로 측정한 3가지 "단순한 방법 vs 개선한 방법"
비교를 한곳에 모은 것이다. 모든 수치는 실제 실행 결과이며(가상 수치 없음),
출처가 되는 원본 로그는 `results/RESULTS.md`에 있다.

---

## Ablation 1: 사람 탐지·추적 단독 vs keypoint 신호 병행

**질문**: bbox 기반 탐지+추적(YOLO11n+ByteTrack)만으로 낙상을 감지할 수 있는가,
아니면 keypoint 신호가 추가로 필요한가?

**실험**: 동일한 낙상 시퀀스(443프레임)에 대해 (a) YOLO11n+ByteTrack bbox 추적만
돌린 결과와 (b) 정답 keypoint에 Stage A 종횡비 로직을 적용한 결과를 비교.

| 지표 | 결과 |
|---|---|
| 전체 프레임 중 YOLO 탐지 실패 | **224 / 443 (50.6%)** |
| 탐지 실패 프레임 중 keypoint 신호가 대신 낙상으로 잡아낸 것 | **29건** |

**결론**: 사람이 완전히 눕는 순간 일반 사람 탐지기의 recall이 급격히 떨어진다
(전체 프레임의 절반!). bbox 추적 단독으로는 이 구간을 놓치므로, keypoint 기반
신호(지금은 정답 keypoint, 실제 배포 시엔 pose 추출기 출력)를 반드시 병행해야
한다. → Stage A에 TRACK_LOST를 별도 트리거로 추가한 근거.

*출처*: `scripts/demo_tracking_vs_transition.py`, `results/RESULTS.md` "탐지+추적
단독 vs keypoint 전환감지 병행 비교" 절, 스크린샷
`results/figures/compare_tracking_miss_transition_catches.png`.

---

## Ablation 2: 단순 그룹핑(끊김 허용 0) vs 연속성 고려 그룹핑

**질문**: 라벨링된 keypoint 프레임을 시퀀스로 묶을 때, 끊김을 전혀 허용하지
않고 그룹핑하는 것과 짧은 끊김을 보간해서 이어붙이는 것 중 어느 쪽이 학습
데이터를 더 많이/정확하게 만드는가?

**실험**: 같은 낙상 카테고리 라벨(falling_from_height, 7개 영상 그룹)에서
`max_gap_for_continuity`(허용 가능한 최대 프레임 끊김)를 1/3/5/10으로 바꿔가며
연속 구간(run) 길이 분포를 비교.

| max_gap | 생성된 run 수 | run 길이 중앙값 | 30프레임 윈도우 확보량 |
|---|---|---|---|
| **1**(끊김 미허용) | 4,795 | **4프레임** | 사실상 0(윈도우 30개/전체) |
| **3**(채택) | 224 | **61프레임** | 6,687개 |
| 5 | 30 | 28 | 장면전환까지 이어붙임(위험) |
| 10 | 9 | 3,950 | 장면전환까지 이어붙임(위험) |

**결론**: "프레임 간격 중앙값이 1이니 거의 연속"이라는 통계만 믿고 끊김을
전혀 허용하지 않으면(gap=1), 2~3프레임씩 잠깐 사라지는 순간이 매우 잦아
데이터가 조각조각 부서진다(런 4개 중앙값). 반대로 너무 관대하게 이어붙이면
(gap≥5) 서로 다른 촬영 컷까지 하나의 시퀀스로 취급하는 위험이 생긴다.
**gap=3**이 "짧은 탐지 유실은 이어붙이되 장면전환은 분리"하는 절충점이었다.
→ 학습 가능한 윈도우가 30개에서 6,687개로 늘어남.

*출처*: `scripts/convert_aihub163_keypoints_to_pyskl.py`, `docs/data_preprocessing.md`
2.2절, `results/RESULTS.md` "pyskl 변환 스크립트" 절.

---

## Ablation 2-보충: 영상 전체 라벨 vs 전환 시점 기반 라벨(v1 vs v2)

Ablation 2와 이어지는 문제: gap=3으로 그룹핑해도, "영상 폴더 전체 = 낙상"이라는
**영상 단위 라벨**을 모든 30프레임 윈도우에 그대로 붙이면(v1) 실제로는 "그냥
서 있는 모습"까지 낙상으로 잘못 라벨링된다(같은 영상의 keypoint bbox 종횡비를
10구간으로 나눠보니 "누움" 프레임은 전체의 18.3%뿐).

| | v1(기계적 슬라이딩) | v2(Stage A 전환감지 적용) |
|---|---|---|
| 라벨 정확도 | 낮음(대부분 서있는 모습에 낙상 라벨) | 높음(실제 전환 주변만 낙상 라벨) |
| 데이터 양 | 6,687개 | 2,715개(양성 1,506 + 정상 1,209) |
| 정상(normal) 클래스 | 없음 | 있음 |

*출처*: `docs/data_preprocessing.md`, `results/RESULTS.md` "v1의 근본 한계" /
"pyskl 변환 스크립트 v2" 절.

---

## Ablation 3: 단순 휴리스틱(Stage A 단독) vs 학습된 분류기(HD-GCN)

**질문**: Stage A 휴리스틱(종횡비 급변/수직속도/탐지유실)만으로 낙상을 판단하는
것과, 그 뒤에 학습된 분류기(HD-GCN)를 붙이는 것 중 어느 쪽이 더 정확한가 —
특히 오탐(false positive)이 얼마나 줄어드는가?

**중요한 함정**: v2 데이터의 "양성(낙상 등) 라벨"은 애초에 Stage A와 똑같은
로직(종횡비 전환 감지)으로 만들어졌다. 그래서 "전환이 있다/없다"는 이진 판정만
놓고 보면 Stage A는 자기가 만든 라벨을 그대로 맞히는 셈이라 사실상 100%에
가깝다 — **이건 의미 있는 비교가 아니라 순환論이다.** 진짜 비교 포인트는:
Stage A는 "전환 있음/없음" 이진 신호만 주지 **어떤 사고 유형인지(낙상/부딪힘/
넘어짐/물체에맞음/정상 5종 중 무엇인지)는 구분하지 못한다.** HD-GCN은 같은
keypoint 시퀀스의 실제 움직임 패턴으로 5종을 구분해야 하는, Stage A가 원천적으로
할 수 없는 task를 수행한다.

**실측 결과 (HD-GCN, batch=32, epoch=15, val 407개)**:

전체 정확도 **81.8%**. 클래스별 recall:

| 클래스 | recall |
|---|---|
| falling_from_height | 0.827 |
| struck_by_collision | 0.545 |
| trip_and_fall | 0.798 |
| struck_by_object | 0.633 |
| normal | 0.886 |

혼동행렬(행=정답, 열=예측):

```
                       falling_  struck_b  trip_and  struck_b    normal
  falling_from_height        67         2        10         1         1
  struck_by_collision         0        12         0         1         9
        trip_and_fall         0         0        71         1        17
     struck_by_object         0         2         3        19         6
               normal         7         4        10         0       164
```

**해석**: falling_from_height/trip_and_fall(recall 0.80~0.83)은 잘 구분되는데,
struck_by_collision/struck_by_object(recall 0.55~0.63)는 확실히 약하다 —
**Ablation 2-보충에서 이미 발견한 패턴(이 두 카테고리는 Stage A 전환 검출
자체가 164~193건뿐으로 적었음)이 다운스트림 분류 성능에도 그대로 이어진다.**
"부딪힘/물체에맞음"은 반드시 눕는 자세로 이어지지 않아 학습 신호 자체가
약했고, 그 결과 분류기도 이 두 클래스를 정상(normal)이나 다른 사고 유형과
자주 헷갈린다(예: struck_by_collision 21건 중 9건이 normal로 오분류).

또한 normal 클래스도 185개 중 21개(11.4%)가 사고 유형으로 오분류된다 —
Stage A는 이 negative 샘플들에 대해 구조적으로 오탐이 0%(애초에 Stage A가
트리거 안 한 구간만 normal로 뽑았으므로)인데, HD-GCN은 실제 움직임을 학습하며
약간의 오탐이 생긴다. **즉 HD-GCN이 다중 클래스 구분 능력을 얻는 대신, Stage A가
구조적으로 못 만들던 종류의 오탐(false positive)을 일부 도입한다는 트레이드오프가
있다** — 이 부분은 정직하게 한계로 기록한다.

*출처*: `scripts/train_hdgcn_fall.py`(학습), `scripts/evaluate_fall_ablation.py`
(평가), `results/RESULTS.md` "Phase 2: HD-GCN 학습 및 ablation" 절.

---

## Ablation 3-보충: 오프라인 평가(81.8%) vs 실시간 파이프라인 재현

**질문**: Ablation 3의 81.8%는 AIHub **정답(GT) keypoint**로 평가한 것이다.
실시간 파이프라인에서는 정답 keypoint가 없으므로 YOLO11n-pose(COCO-17)로 직접
keypoint를 뽑고 AIHub 16점 레이아웃으로 근사 리매핑(`pose_extractor.py`)해서
써야 한다. 이 **이중 근사** 위에서도 HD-GCN이 오프라인 성능만큼 낙상을 잡아낼
수 있는가?

**실험**: 데모 5개 영상(각 200프레임)에 대해 **동일한 추적+PPE 결과** 위에서
낙상 감지 로직만 3가지로 교체해 비교했다(`src/pipeline.py`의 `fall_mode`
파라미터, `python scripts/demo_full_pipeline.py --compare-all`).

- **bbox_heuristic**: 추적 bbox 종횡비/수직속도 휴리스틱(기존 Stage A, 최경량)
- **keypoint_heuristic**: 같은 휴리스틱 로직을 그대로 두고, 입력만 추적 bbox
  대신 YOLO11n-pose가 실시간으로 뽑은 keypoint의 bounding box로 교체
- **hdgcn**: 같은 keypoint를 30프레임 버퍼로 쌓아 학습된 HD-GCN(5-way)으로 분류

| 영상 | bbox_heuristic | keypoint_heuristic | hdgcn(실시간) |
|---|---|---|---|
| S2-N6001_trip | 3건 | 5건 | **0건** |
| S2-N6301_trip | 4건 | 9건 | **0건** |
| S2-N6401_trip | 5건 | 13건 | **0건** |
| S2-N4601_fall | 9건 | 9건 | **0건** |
| S2-N4701_fall | 10건 | 12건 | **0건** |
| **합계** | **31건** | **48건** | **0건** |

**HD-GCN이 0건인 게 버그가 아님을 확인**: threshold를 걷어내고 raw softmax를
직접 찍어봤다(`scratchpad/debug_hdgcn_check.py`, S2-N4601_fall). 실제 낙상이
발생한 트랙(43번, bbox 휴리스틱이 frame=156에서 낙상 검출)에 대해서도 HD-GCN은
frame 166~185 구간 내내 "normal"을 **70~100% 확신**으로 예측했다 — 애매하게
낮은 confidence로 새는 게 아니라, 아예 낙상 쪽 클래스로 갈 확률 자체가 거의
없었다. 정규화 코드(`hdgcn_dataset.py`의 학습 전처리와 `hdgcn_live.py`의 실시간
전처리)도 대조해 일치함을 확인했으므로 전처리 버그는 아니다.

**결론**: 학습 데이터(AIHub 정답 keypoint)와 실시간 입력(YOLO11n-pose + COCO17
→ AIHub16 근사 리매핑) 사이의 분포 차이가 너무 커서, 오프라인 81.8%라는 수치가
**실시간 배포에는 전혀 이전되지 않는다.** 반면 keypoint_heuristic은 같은
실시간 keypoint를 쓰면서도 bbox_heuristic보다 항상 같거나 더 많은 낙상을
잡아냈다(31건 → 48건, +55%) — keypoint가 tracker bbox보다 자세 변화에 더
민감한 신호이기 때문으로 보인다. **단, 이 수치는 재현율(recall) 관점의 비교이며
프레임 단위 정답 라벨이 없어 오탐률(precision)은 별도로 검증하지 못했다** —
keypoint_heuristic이 더 많이 잡는 이유가 실제 낙상을 더 잘 잡아서인지, 민감도가
높아져 오탐도 같이 늘어난 것인지는 구분할 수 없다.

**실무적 시사점**: 지금 상태로는 HD-GCN을 실시간 경로에 배포하는 것보다
keypoint_heuristic이 더 나은 선택이다. HD-GCN을 실시간에서 쓰려면 (1) 학습
데이터 자체를 YOLO11n-pose 출력으로 다시 만들거나, (2) 리매핑 없이 COCO-17
레이아웃으로 HD-GCN을 재학습하는 두 방향 중 하나가 필요하다 — 둘 다 이번
세션 범위 밖의 향후 과제로 남긴다.

*출처*: `src/pipeline.py`(`fall_mode` 3분기), `src/fall_detection/pose_extractor.py`,
`src/fall_detection/hdgcn_live.py`, `scripts/demo_full_pipeline.py --compare-all`
실행 로그.

---

## Ablation 4: pose 추출기 비교 (YOLO11n-pose vs RTMPose) + fine-tuning 2회 시도

**질문**: Ablation 3-보충에서 실시간 keypoint 품질 자체가 문제라는 게 드러났다
(HD-GCN 0건 검출의 근본 원인). YOLO11n-pose를 다른 pose 추출기로 바꾸거나,
우리 데이터로 fine-tuning하면 이 문제를 고칠 수 있는가?

**실측 1 — YOLO11n-pose의 실패 지점을 직접 시각화**: 프레임 위에 keypoint를
그려서 확인한 결과, 두 가지 실패 패턴이 나왔다.
- 완전히 쓰러져 누운 사람은 **아예 탐지 자체가 안 됨**(box도 keypoint도 없음).
  같은 프레임에 있던 나머지 2명은 정상 검출.
- 낙하 도중(모션 블러 있는 프레임)에는 탐지는 되지만 keypoint가 실제 신체와
  전혀 다른 위치로 어긋남(다리가 화면 밖 다른 사람 쪽으로 이어지는 등).

이는 Ablation 1에서 발견한 "YOLO 탐지기가 누운 자세에서 50.6% 실패"와 같은
근본 원인(COCO 사전학습 데이터가 서 있는/걷는 자세 위주)이 pose 추출기에도
그대로 나타난 것이다.

**실측 2 — RTMPose로 교체**: 같은 프레임에 RTMPose(rtmlib, balanced 모드)를
돌리자 YOLO11n-pose가 완전히 놓쳤던 쓰러진 사람을 정상적으로 검출했다(시각
비교 스크린샷 확보). 단, 이 환경은 onnxruntime의 CUDA 실행공급자가 cuDNN8
의존 dll(zlibwapi.dll) 누락으로 GPU 가속이 실패해 CPU로 폴백된다(실측 확인) —
balanced 모드 기준 약 289ms/프레임으로, GPU 기반 YOLO11n-pose보다는 느리지만
쓰러진 사람을 실제로 잡는 게 더 중요하다고 판단해 **RTMPose를 기본 pose
backend로 채택**했다.

**실측 3 — AIHub163 GT keypoint로 YOLO11n-pose fine-tuning 시도(2회, 모두 실패)**:

| 시도 | 방법 | 결과 |
|---|---|---|
| 1차 | 1,500장/30epoch, box/cls/dfl 손실 기본 가중치 | Pose mAP50 0→0.395로 개선, 놓쳤던 쓰러진 사람도 검출(conf 0.74). **그러나** conf=0.01·max_det=300까지 풀어도 한 프레임에서 후보가 1~2개뿐 — 3~5명이 실제로 있는 장면에서 다중인원 recall이 붕괴됨 |
| 2차 | 같은 데이터, box=0/cls=0/dfl=0로 재학습 | 후보 수는 19개로 회복(억제 문제 해결)됐지만, confidence 보정 능력 자체가 사라져 **사람 없는 자재 더미(conf 0.51)가 실제 사람(conf 0.12)보다 높은 점수**를 받음 — 실사용 불가 |

**원인**: AIHub163 라벨은 이미지 한 장에 **사고 당사자 1명만** 라벨링돼 있고
같은 화면의 다른 사람은 라벨이 없다(실측 확인: 200개 샘플 중 다인원 라벨
0건). box/cls 손실을 그대로 학습하면 "라벨 안 된 사람 = 배경"이라고
학습되어 다중인원 탐지가 무너지고, 손실을 꺼버리면 이번엔 사람/배경 구분
능력 자체가 사라진다 — 단일 인물 라벨 데이터로는 detection head까지
fine-tuning하는 게 근본적으로 어렵다.

**결론**: pipeline.py의 pose backend 기본값을 **RTMPose로 유지**하고, YOLO
fine-tuning은 (다인원 라벨 데이터 확보 또는 keypoint 헤드만 분리 학습하는
방법 없이는) 보류한다. "실패한 시도를 실측으로 확인하고 정확한 원인을
진단해 다음 결정을 내렸다"는 과정 자체를 기록으로 남긴다.

*출처*: `src/fall_detection/pose_extractor.py`(`RTMPoseExtractor`/`YoloPoseExtractor`),
`scripts/convert_aihub163_keypoints_to_yolo_pose.py`, `scripts/finetune_pose_yolo.py`,
`src/pipeline.py`(`pose_backend` 파라미터).

---

## Version 2: VLM(Gemini) 단일 호출 vs CV 파이프라인

**질문**: 지금까지의 CV 파이프라인(탐지+추적 → PPE 모델 → pose 모델 →
휴리스틱/HD-GCN → 이벤트 통합)은 데이터 라벨링·모델 학습·여러 단계가 필요하다.
VLM(Gemini)에 프레임 몇 장을 한 번에 넣고 "낙상/PPE 미착용"을 바로 물어보면
어느 정도까지 대체 가능한가?

**실험**: 같은 데모 클립 7개(CV 파이프라인이 이미 낙상 유/무를 판정해둔 것,
실제 낙상 있는 클립과 없는 클립을 섞음)에 대해 원본 프레임 중 최대 12장을
균등 샘플링해서 Gemini(`gemini-flash-latest`)에 한 번에 전달, 구조화된 JSON으로
(1) 낙상 여부·횟수(허위 카운팅 금지 지시 포함), (2) 보호구 4종(헬멧/조끼/벨트/
안전화) 개별 미착용 여부, (3) 장면 요약을 받았다.

| 클립 | VLM 낙상 판정(횟수) | CV 파이프라인 낙상 판정(이벤트 수) | 일치 |
|---|---|---|---|
| S2-N6001_trip_clip00 | 있음(1회) | 있음(2건) | ✓ |
| S2-N4601_fall_clip12 | 없음(0회) | 있음(5건) | ✗ |
| S2-N4601_fall_clip01 | 있음(2회) | 없음(0건) | ✗ |
| S2-N6301_trip_clip04 | 있음(6회) | 있음(5건) | ✓ |
| S2-N6301_trip_clip00 | 있음(1회) | 없음(0건) | ✗ |
| S2-N6401_trip_clip02 | 있음(2회) | 있음(5건) | ✓ |
| S2-N6401_trip_clip09 | 있음(2회) | 없음(0건) | ✗ |
| **일치율** | | | **3/7 (43%)** |

**속도**: VLM은 클립 길이(10~132프레임)와 무관하게 매번 **15~25초**(네트워크
왕복 포함), CV 파이프라인은 프레임 수에 비례하지만 짧은 클립은 훨씬 빠르다.

**정직하게 기록할 한계**:
- 절대 정답(ground truth) 라벨이 없어 "VLM과 CV 중 어느 쪽이 더 정확한가"는
  판단할 수 없다 — 이 표는 **일치/불일치 패턴**만 보여준다.
- VLM은 쭈그렸다 일어서는 반복 동작(안전 테스트/훈련 장면)을 낙상으로 판단하는
  경향이 있다(clip01, clip00, clip09) — 실제 낙상과 의도적 낙하 테스트를 구분
  못 하는 것으로 보인다.
- **프롬프트 민감도**: 초기 프롬프트(낙상 유무만 질문)와 개선 프롬프트(횟수
  카운팅 + "추측 금지" 명시 지시 추가) 사이에 같은 이미지·같은 모델인데도
  7개 중 2개 클립의 판정이 뒤집혔다(clip12: 있음→없음, clip00: 없음→있음).
  VLM 결과는 프롬프트 문구에 민감하다는 뜻이며, 이는 프롬프트 하나 잘못
  쓰면 결과가 흔들릴 수 있다는 별도의 리스크로 기록해둔다.
- VLM은 track_id별 시계열 타임라인을 만들지 못한다 — 클립 전체에 대한
  한 번의 정성적 판단만 가능하다.

**결론**: VLM은 데이터 준비·모델 학습이 전혀 필요 없이 즉시 배포 가능하다는
확실한 장점이 있지만, 지금 프롬프트/샘플링 방식으로는 CV 파이프라인과의
판정 일치율이 43%로 낮고, 프롬프트 문구 변화에도 판정이 흔들린다 — 상용
배포 전 더 엄밀한 검증(정답 라벨 확보, 프롬프트 고정·버저닝, 반복 안정성
테스트)이 필요하다.

*출처*: `scripts/vlm_safety_check.py`, `outputs/<클립>/compare/vlm_result.json`.
