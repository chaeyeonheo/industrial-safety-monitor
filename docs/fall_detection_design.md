# 낙상 감지 vs PPE/구역침입 — 왜 분리했고 각 브랜치는 어떻게 동작하는가

이 문서는 보고서 작성을 위해 설계 근거를 상세히 남긴다. 요약이 아니라 "왜 이렇게
결정했는지"와 "실제로 무엇을 실측해서 그 결정을 뒷받침했는지"를 함께 기록한다.

## 1. 전체 구조: 공유되는 부분은 "탐지 + 추적"뿐이다

```
Video Frame
   │
   ▼
[YOLO11 person detect + ByteTrack]  ← 세 브랜치가 공유하는 유일한 부분
   │
   ├──▶ 낙상 브랜치 (fall_detection)
   │      Stage A: 경량 휴리스틱(모델 아님, 규칙 기반) → 후보 track만 선별
   │      Stage B: 후보 track에 한해 pose 추출 → 시계열 분류기(ST-GCN 등)
   │
   ├──▶ PPE 브랜치 (ppe_detection)
   │      별도로 파인튜닝한 헬멧/조끼 탐지 YOLO를 같은 프레임에 통째로 실행
   │      → person bbox와 IoU 매칭(간접) 또는 직접 클래스 모델(직접)
   │
   └──▶ 구역침입 브랜치 (zone_intrusion)
          person track의 발위치를 ROI 폴리곤/호모그래피와 기하학적으로 비교
          (학습 모델 불필요)
   │
   ▼
[Event Aggregator] → [NLG] → 알람
```

**중요한 오해 하나를 먼저 정리한다**: 이 구조는 "입력이 어떤 상황인지 보고 낙상
브랜치로 갈지 PPE 브랜치로 갈지 정하는" 라우팅/분기 구조가 **아니다**. 세 브랜치는
**매 프레임, 매 person track마다 항상 전부 병렬로 실행**된다. 한 사람이 헬멧도
안 쓰고 동시에 넘어질 수 있는 것처럼, 세 상황은 서로 배타적이지 않기 때문이다.
그래서 "이 프레임을 낙상 문제로 볼지 PPE 문제로 볼지 결정하는 게이트"는 애초에
필요하지 않다. 각 브랜치가 독립적으로 이벤트를 내거나 안 내거나 할 뿐이고,
Event Aggregator가 그 결과들을 모아서 알람으로 합친다.

## 2. 왜 낙상 감지와 PPE 감지를 완전히 분리된 모델로 두는가

두 문제는 **입력 표현(input representation)** 자체가 다르다.

| | PPE 감지 | 낙상 감지 |
|---|---|---|
| 필요한 정보 | 한 프레임의 RGB 픽셀 (헬멧이 보이는가) | 여러 프레임에 걸친 자세 변화 |
| 모델 유형 | 단일 프레임 bbox 탐지기 (YOLO) | 시계열 그래프 신경망 (ST-GCN/CTR-GCN) 또는 시계열 CNN-LSTM |
| 시간 축 | 불필요 — 사진 한 장으로 판단 가능 | 필수 — 한 프레임만으로는 "웅크림"과 "쓰러짐"을 구분 못 함 |

YOLO 계열 탐지기는 프레임마다 독립적으로 추론하며 메모리(시간 축 상태)가 없다.
따라서 "헬멧 탐지 모델에 낙상 클래스를 하나 더 추가"하는 식으로는 애초에 시계열
정보를 표현할 방법이 없어 구조적으로 성립하지 않는다. 반대로 ST-GCN류 모델은
스켈레톤 시퀀스를 입력으로 받으므로 "헬멧 색깔이 파란지 노란지" 같은 순수 외형
정보를 판단하는 데는 적합하지 않다. 두 태스크를 하나의 멀티태스크 모델로 합치려면
입력 표현부터 다시 설계해야 하는데(예: 프레임별 RGB + 시계열 포즈를 모두 받는
멀티모달 모델), 이는 복잡도 대비 이득이 불분명하고 각 태스크의 검증된 오픈소스
구현(Ultralytics, pyskl)을 그대로 재사용하기도 어려워진다. 그래서 지시문 원안대로
완전히 분리된 두 모델을 유지한다.

## 3. 낙상 브랜치: Stage A(휴리스틱) → Stage B(포즈 분류) 2단계 구조

### 3.1 왜 2단계인가

Stage B(포즈 추출 + ST-GCN/CTR-GCN 분류)는 무거운 연산이다. 매 프레임, 매
person마다 항상 돌리면 실시간성을 해친다. 그래서 가벼운 규칙 기반 Stage A로
먼저 "낙상일 수도 있는" 후보만 걸러내고, Stage B는 그 후보에 대해서만 실행한다
(`src/fall_detection/heuristic_trigger.py`).

### 3.2 Stage A의 세 가지 트리거 신호

구현: `FallHeuristicTrigger.update(frame_idx, tracks)` — 프레임마다 호출하면
`window_frames`(기본 15프레임) 안의 bbox 이력을 track_id별로 들고 있다가 아래
세 조건 중 하나라도 만족하면 `TriggerEvent`를 낸다.

**(1) ASPECT_RATIO_SPIKE — bbox 종횡비 급변**

```
ratio = bbox_width / bbox_height
```

서 있는 사람은 세로로 길쭉해 `ratio`가 작다(대략 0.3~0.5). 쓰러지면 가로로
납작해져 `ratio`가 커진다(대략 1.5 이상). `window_frames` 구간의 시작과 끝에서
`ratio`의 증가폭(`ratio_delta`)이 `aspect_ratio_delta_threshold`(기본 0.5)를
넘으면 트리거한다.

**(2) VERTICAL_VELOCITY_SPIKE — 중심점 수직 하강 속도**

```
centroid_y = (bbox.y1 + bbox.y2) / 2
vertical_velocity = (centroid_y_끝 - centroid_y_시작) / dt초
```

이미지 좌표계는 아래로 갈수록 y가 커지므로, 짧은 시간에 `centroid_y`가 빠르게
증가하면(px/s 기준 `vertical_velocity_threshold_px_per_s`, 기본 200) 급격한
하강(넘어짐/떨어짐) 후보로 본다.

**(3) TRACK_LOST — 추적되던 track이 갑자기 사라짐 (2026-07-05 실측으로 추가)**

`results/RESULTS.md`의 Phase 1 실측 결과: AIHub 낙상 시퀀스에서 사람이 안전매트
위에 엎드려 쓰러진 프레임을 YOLO11n이 **전혀 탐지하지 못하는** 사례를 확인했다
(서 있을 때는 정상 탐지됨, `tracking_demo_frame50.png` vs `tracking_demo_frame150.png`
비교 참고). 이 경우 bbox 자체가 존재하지 않으므로 (1), (2) 신호는 계산할 대상이
없어 애초에 발동이 불가능하다. 즉 **탐지기가 놓치는 바로 그 순간이 낙상의 증거일
수 있는데, 앞의 두 신호는 정확히 그 순간을 놓친다.**

그래서 세 번째 독립 신호로 "일정 프레임 이상(`track_loss_min_history_frames`,
기본 10) 안정적으로 추적되던 track이 `track_loss_grace_frames`(기본 2)만큼
연속으로 사라짐"을 추가했다. ByteTrack 자체도 내부적으로 lost-track 버퍼(기본
약 30프레임)를 두고 짧은 가려짐을 재매칭으로 흡수하므로, 우리 `Track` 출력
단계까지 도달하는 TRACK_LOST는 ByteTrack의 버퍼링을 넘어선 경우다.

**TRACK_LOST의 오탐 원인과 완화**: 이 신호는 진짜 낙상 외에도 "사람이 화면 밖으로
걸어나감" 같은 정상 상황에서도 발생한다. 이를 구분하기 위해 마지막으로 확인된
bbox가 화면 가장자리 근처(`frame_edge_margin_ratio`, 기본 0.08 = 프레임 크기의
8%)에 있었는지 확인하는 `near_frame_edge` 필터를 추가했다. 가장자리 근처에서
사라졌으면 화면 이탈 가능성이 높다고 보고 `confidence_hint`를 0.5 → 0.2로
낮춘다(완전히 버리지는 않음 — 최종 판단은 Stage B 또는 사람 확인 몫).

**가려짐(occlusion)은 어떻게 되는가**: 다른 사람/사물에 짧게 가려지는 경우는
위에서 설명했듯 ByteTrack의 lost-track 버퍼가 대부분 흡수해 우리 쪽 TRACK_LOST로
넘어오지 않는다. 버퍼를 넘어서는 긴 가려짐은 원리상 "화면 중앙 부근에서 갑자기
사라짐"으로 나타나므로 `near_frame_edge=False`인 TRACK_LOST와 구분이 안 된다 —
이는 현재 구조의 알려진 한계이며, Stage B가 마지막 위치에서 포즈 추출을
시도해 확인하거나(아래 3.3), 그마저 실패하면 사람이 최종 확인하는 것으로
처리한다. 향후 개선 방향으로는 재등장 위치 추적(사라진 위치 근처에서 새 track이
곧 나타나면 같은 사람으로 간주해 오탐 제거) 등이 있으나 아직 구현하지 않았다.

### 3.3 TRACK_LOST 이벤트를 Stage B로 어떻게 넘기는가 (설계, 아직 미구현)

(1), (2) 신호는 살아있는 bbox가 있으므로 그 위치에서 바로 pose crop을 뜨면 된다.
TRACK_LOST는 살아있는 bbox가 없다. 계획은:

1. 마지막으로 확인된 `last_known_bbox` 위치에서 현재 프레임을 크롭해 pose 추출을
   **best-effort로 시도**한다(사람 탐지기는 놓쳤어도 pose 추정 모델은 다른 학습
   편향 덕에 찾아낼 수도 있음 — 아직 검증 안 됨).
2. pose 추출도 실패하면 `confidence_hint`(0.5 또는 0.2)를 그대로 실은 낮은 확신도의
   "낙상 의심(확인 불가)" 이벤트를 Event Aggregator로 바로 보내, 사람이 최종
   확인하도록 한다.

이 부분(`pipeline.py`의 오케스트레이션, pose 추출기 자체)은 아직 코드로 작성하지
않았다 — pose 추출기(Phase 2 남은 작업)가 먼저 필요하다.

## 4. PPE 브랜치: IoU 매칭 방식의 의미

`ppe_detection.mode: indirect`(간접 연결) 방식은 person bbox와 헬멧/조끼 bbox를
각각 독립적으로 탐지한 뒤 IoU로 매칭한다. 매칭에 실패한 person(=자신의 bbox
영역 안에 헬멧 bbox가 없는 사람)을 "미착용"으로 판정한다.

이 방식의 핵심 장점(사용자 확인, 2026-07-05): 헬멧이 바닥에 놓여 있거나 사람이
손에 들고 있는 경우, 그 헬멧 bbox는 **그 사람의 person bbox와 겹치지 않거나
매우 낮은 IoU**를 갖는다. 즉 "화면 어딘가에 헬멧이 존재하는가"가 아니라 "이
사람이 헬멧을 착용한 상태로 겹쳐 있는가"를 보므로, 헬멧을 들고 있거나 근처에
놓아둔 상황이 오탐(착용으로 오판)되지 않는다. 우리가 실제로 찾아야 하는 건
"헬멧을 쓰고 있지 않은 사람"이지 "화면에 헬멧이 있는지 여부"가 아니므로, 이
매칭 방식이 문제 정의와 정확히 들어맞는다.

대안인 `direct`(직접 클래스) 방식은 "헬멧 미착용 사람"을 아예 별도 클래스로
라벨링해 단일 모델이 한 번에 판정하게 한다. 두 방식 모두 구현해 Phase 3에서
비교할 예정(지시문 원안).

## 5. 지금까지 실측으로 확인된 사실 요약 (results/RESULTS.md 원문 참고)

- 서 있는 사람은 YOLO11n이 안정적으로 탐지(conf 0.85~0.89 수준).
- 쓰러져 엎드린 사람은 YOLO11n이 **전혀 탐지하지 못함** — Stage A에 TRACK_LOST를
  추가한 직접적 근거.
- Phase 1 트래킹 검증에 쓴 프레임 시퀀스(validation 서브셋 하나, 443프레임)는 간격이
  불균일(중앙값 7, 최대 60)해 ByteTrack이 38개의 짧은 track으로 파편화됐다. **단,
  이후 train 셋 4개 카테고리 전체(95,798개 파일)를 전수 분석한 결과 이는 예외적
  케이스였고, train 셋은 실제로 프레임 간격 중앙값이 1(거의 연속)임을 확인했다**
  (`scripts/analyze_keypoint_labels.py`, `results/RESULTS.md` 참고). Stage A의
  track_loss 조건은 연속 프레임 기준으로 재검증이 필요하다.
- Stage A를 실제로 돌려본 결과 443프레임 중 3건 트리거, 그중 TRACK_LOST 1건은
  `near_frame_edge` 필터가 의도대로 낮은 신뢰도를 부여함을 확인.

## 6. 아직 남은 것

- Stage B(pose 추출기 + ST-GCN/CTR-GCN/1D-CNN-LSTM 분류기) 구현
- TRACK_LOST → best-effort pose 추출 오케스트레이션(`pipeline.py`)
- Precision/Recall/F1/Time-to-Detection 등 정량 평가(`scripts/evaluate_fall.py`) —
  라벨과의 정합이 필요하므로 Stage B 이후 진행
- PPE `direct` 클래스 모델과 `indirect` IoU 매칭 방식의 실측 비교
