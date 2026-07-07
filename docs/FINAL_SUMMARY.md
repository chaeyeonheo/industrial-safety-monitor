# 최종 요약 — 전체 프레임워크 작동 방식 및 발표 흐름 제안

이 문서는 프로젝트 전체를 하나의 이야기로 엮은 최종 정리 문서다. 각 절의
"상세 문서" 링크를 따라가면 실측 수치와 시행착오 전체를 볼 수 있다. **발표
슬라이드는 이 문서의 절 순서를 그대로 따라가면 된다.**

## 0. 문제 정의

산업 현장(공사현장/물류센터) CCTV 영상에서 (1) 낙상/사고성 이상행동과
(2) PPE(안전보호구) 미착용을 실시간으로 감지해 "누가/무엇을/언제" 형태의
자연어 알람을 만드는 시스템. 오픈소스(Ultralytics YOLO11, HD-GCN)를 최대한
재사용하고, 두 시나리오가 하나의 공유 백본 위에서 병렬로 동작하도록 설계했다.

## 1. 전체 아키텍처

```
                    [YOLO11n person detect + ByteTrack]   ← 공유 백본(Phase 1)
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                                            ▼
   [낙상 감지 브랜치 — fall_mode 3종 중 택1]         [PPE 미착용 판정 브랜치]
   ① bbox_heuristic: 추적 bbox 종횡비/수직속도/       YOLO11n(4클래스: 헬멧/조끼/
     탐지유실 휴리스틱(최경량, 기본값)                 벨트/안전화) + 간접연결(IoU
   ② keypoint_heuristic: RTMPose 실시간 추출          대신 중심점+신체부위 매칭)
     + 같은 휴리스틱(①보다 민감, 실측 +55%)           track 첫 관찰 시점에만 판정,
   ③ hdgcn: 같은 keypoint를 HD-GCN 5-way 분류         이후 고정(인과적, 안 깜빡임)
     (오프라인 81.8%였지만 실시간 0건 — 분포 차이)

   ※ pose backend는 YOLO11n-pose→RTMPose로 교체(실측: 누운 사람 탐지 실패
     확인 후). AIHub163 GT로 YOLO11n-pose fine-tuning도 2회 시도했으나 모두
     실패(Ablation 4) — RTMPose를 기본값으로 유지.
              │                                            │
              └─────────────────────┬─────────────────────┘
                                    ▼
                          [이벤트 통합 + 쿨다운]
                                    ▼
                    [템플릿 NLG] → 화면 오버레이 알람
                                    ▼
                [이벤트 타임라인 JSON] → [Gemini VQA 웹앱] → 자연어 질의응답
```

**핵심 설계 원칙**: 공유되는 건 "사람 탐지+추적"까지뿐이다. 그 이후로는
분기/라우팅이 없고, 추적된 모든 사람에 대해 낙상 브랜치와 PPE 브랜치가
**항상 동시에 병렬로** 실행된다(한 사람이 헬멧도 안 쓰고 동시에 넘어질 수
있으므로 서로 배타적이지 않음).

## 2. 컴포넌트별 상세

| 단계 | 무엇을 하는가 | 왜 이렇게 했는가 | 상세 문서 |
|---|---|---|---|
| Phase 1 | YOLO11n(COCO person) + ByteTrack | 사람 탐지는 이미 COCO에 있어 파인튜닝 불필요. 두 시나리오가 공유 | `results/RESULTS.md` Phase 1 |
| Phase 2 데이터 | AIHub keypoint 라벨 → pyskl 스타일 윈도우 | v1(기계적 슬라이딩)의 라벨 오염 문제 발견 → v2(Stage A 전환감지 재사용) | `docs/data_preprocessing.md` |
| Phase 2 keypoint 매핑 | 16개 점 관절 이름 실측 추론 | 공식 문서 없어 이미지에 점 찍어 시각 확인(best-effort) | `docs/keypoint_mapping.md` |
| Phase 2 Stage A | 종횡비 급변/수직속도/탐지유실 휴리스틱 | 학습 없이 실시간 동작하는 1차 필터 | `docs/fall_detection_design.md` |
| Phase 2 Stage B | HD-GCN(ICCV 2023) 5-way 분류 | Stage A는 이진 신호뿐이라 사고 유형 구분 불가 → 학습 분류기로 보완 | `docs/ablation_studies.md` Ablation 3 |
| Phase 2 실시간 pose | RTMPose(rtmlib) + AIHub16 근사 리매핑 | YOLO11n-pose는 누운 자세 탐지 실패(실측 확인)→RTMPose로 교체. AIHub163 GT로 fine-tuning 2회 시도했으나 모두 실패(다중인원 억제/confidence 붕괴) | `docs/ablation_studies.md` Ablation 4 |
| Phase 2 실시간 3-way 비교 | bbox_heuristic / keypoint_heuristic / hdgcn 중 택1(`fall_mode`) | 셋 다 실측 비교해야 어떤 게 실배포에 맞는지 판단 가능 — HD-GCN은 분포 차이로 실시간 0건 검출 확인 | `docs/ablation_studies.md` Ablation 3-보충 |
| Version 2 (비교용) | VLM(Gemini) 프레임 샘플링 1회 호출로 낙상/PPE 판단 | CV 파이프라인 대비 "학습 없이 즉시 배포" 트레이드오프를 실측으로 보여주기 위한 대안 버전 | `docs/ablation_studies.md` Version 2, `scripts/vlm_safety_check.py` |
| Phase 3 PPE 클래스 매핑 | bbox class 코드 실측 추론(헬멧/조끼/벨트/안전화) | 공식 문서 없어 이미지에 bbox 그려 시각 확인 | `docs/ppe_class_mapping.md` |
| Phase 3 PPE 탐지 | YOLO11n 4클래스 파인튜닝(mAP50 0.781→**0.920**, 2732라벨/40ep로 재학습) | 사람 클래스는 Phase 1이 이미 담당, 보호구만 추가 학습. 데이터를 2000→8000프레임 샘플로 늘려 재학습 | `results/RESULTS.md` Phase 3 |
| Phase 3 미착용 판정 | 간접 연결(중심점+신체부위 매칭) | 전체 bbox IoU는 부분 박스(헬멧 등)엔 항상 0에 가까워 무의미 | `src/ppe_detection/indirect_association.py` |
| 통합 파이프라인 | track이 처음 관찰된 첫 몇 프레임(기본 6)만 보고 PPE 확정, 이후 재판정 없이 고정 | 매 프레임 새로 판정하면 각도/블러로 미착용 여부가 계속 바뀌는 문제 발견(사용자 실측) → 인과적(미래를 안 봄) 방식으로 1회만 확정, 실시간 스트리밍에도 그대로 적용 가능 | `src/pipeline.py` |
| 파이프라인 실행 방식 | 추적 모델 전체 패스 → GPU 내림 → PPE 모델 전체 패스 → GPU 내림 → 이벤트 통합(GPU 없음) | 두 모델을 프레임마다 번갈아 호출하면 지속 부하가 커짐(반복되는 시스템 크래시 대응) — 완전 순차 실행으로 전환 | `src/pipeline.py` `run_offline()` |
| NLG/VQA | 템플릿(실시간 알람) + Gemini(사후 질의응답, 이벤트 타임라인 JSON을 근거로 제공) | 실시간 알람은 LLM 호출 없이 즉시, 사후 질의는 자연어로 유연하게. Gemini는 타임라인에 없는 내용은 지어내지 않도록 시스템 프롬프트로 제약 | `src/nlg/vqa_gemini.py`, `webapp/` |

## 3. Ablation 4종 + Version 2 비교 (핵심 실증) — 상세: `docs/ablation_studies.md`

1. **탐지+추적 단독 vs keypoint 신호 병행**: 낙상 시퀀스 443프레임 중 YOLO
   탐지 실패 50.6%, 그중 29건은 keypoint 전환감지가 대신 잡아냄.
2. **단순 그룹핑(gap=1) vs 연속성 고려 그룹핑(gap=3)**: 윈도우 수 30개 →
   6,687개로 증가. 이어서 v1(영상 단위 라벨) vs v2(전환 시점 라벨)의 라벨
   품질 문제도 발견.
3. **Stage A 휴리스틱 단독 vs HD-GCN 학습 분류기**: Stage A는 이진 신호뿐이라
   5-way 분류 불가. HD-GCN 실측 정확도 81.8%(falling/trip 0.80~0.83,
   collision/struck-by 0.55~0.63 — 데이터 구성 단계의 약점이 그대로 이어짐).
4. **pose 추출기 비교(YOLO11n-pose vs RTMPose) + fine-tuning 2회 시도**:
   YOLO11n-pose는 누운 사람을 아예 못 찾음(실측 확인) → RTMPose로 교체해
   해결. AIHub163 GT keypoint로 YOLO11n-pose fine-tuning은 2회 모두 실패
   (1차: 다중인원 recall 붕괴, 2차: confidence 보정 능력 상실) — 원인은
   라벨이 이미지당 1명뿐이라는 데이터 한계로 진단.
5. **Version 2 — VLM(Gemini) 단일호출 vs CV 파이프라인**: 동일 클립 7개 비교,
   낙상 판정 일치율 3/7(43%), VLM은 학습 불필요하지만 클립당 15~25초, 프롬프트
   문구에 따라 판정이 뒤집히는 민감도도 확인.

## 4. 실측 수치 요약

| 지표 | 값 | 출처 |
|---|---|---|
| Phase1 탐지 실패율(낙상 영상) | 50.6% (224/443프레임) | RESULTS.md |
| PPE YOLO mAP50 (1차, 662라벨/20ep) | 0.781 | RESULTS.md |
| PPE YOLO mAP50 (재학습, 2732라벨/40ep) | **0.920** | RESULTS.md |
| HD-GCN 5-way 정확도 (오프라인, 정답 keypoint) | 81.8% | ablation_studies.md |
| HD-GCN 실시간 낙상 검출 (5개 영상×200프레임, YOLO11n-pose 입력) | **0건**(오프라인 성능 미이전, 분포 차이 확인됨) | ablation_studies.md Ablation 3-보충 |
| 낙상 검출 건수: bbox 휴리스틱 vs keypoint 휴리스틱 (5개 영상 합계) | 31건 → 48건 (+55%) | ablation_studies.md Ablation 3-보충 |
| 낙상 라벨 윈도우 수 (v1 → gap 튜닝) | 30개 → 6,687개 | data_preprocessing.md |
| pose fine-tuning 1차 시도 (Box mAP50 / Pose mAP50) | 0.952 / 0.395 (그러나 다중인원 recall 붕괴) | ablation_studies.md Ablation 4 |
| VLM(Gemini) vs CV 파이프라인 낙상 판정 일치율 (7개 클립) | 3/7 (43%) | ablation_studies.md Version 2 |
| VLM 처리 시간 (클립 길이 10~132프레임 무관) | 15~25초/클립 | ablation_studies.md Version 2 |

## 5. 알려진 한계

- keypoint 관절 매핑, PPE 클래스 매핑 모두 공식 문서 없이 실측 추론(best-effort,
  확정 아님).
- HD-GCN은 실시간 pose 추출기(YOLO11n-pose)와 연결은 됐지만, 학습 데이터(AIHub
  정답 keypoint)와 실시간 입력(YOLO11n-pose + COCO17→AIHub16 근사 리매핑) 사이
  분포 차이가 커서 오프라인 81.8%가 실시간에는 전혀 재현되지 않음(5개 영상
  1,000프레임에서 0건 검출, ablation_studies.md Ablation 3-보충 참고). 현재
  실시간 배포에는 keypoint_heuristic이 더 나은 선택.
- keypoint_heuristic이 bbox_heuristic보다 낙상을 더 많이 잡아내지만(31→48건),
  프레임 단위 정답 라벨이 없어 이게 recall 향상인지 오탐 증가인지는 구분 못함.
- PPE 모델이 학습 도메인(물류센터)과 다른 장면(낙상 시연 영상)에서는 정확도가
  떨어짐(도메인 일반화 한계, 실측으로 확인).
- 사람 탐지기가 특이한 물체(자재 더미 등)를 사람으로 오탐하는 사례 발견.
- ByteTrack이 사람이 일시적으로 가려지거나 프레임 샘플링 간격이 벌어지면(AIHub
  원본 프레임 번호 간격이 불규칙, 중앙값 7·최대 53) track_id를 새로 부여하는
  경우가 있음 — 짧은 시간 같은 사람이 다른 ID로 갈리는 현상. ByteTrack 자체의
  재식별(Re-ID) 한계이며, 프레임 보간 없이는 완전히 해결하기 어려움.
- 컴퓨터 하드웨어 안정성 이슈(반복 크래시)로 전체 443프레임이 아닌 200프레임
  단위로 데모를 제한함.
- AIHub163 GT keypoint로 YOLO11n-pose를 fine-tuning하려는 시도가 2회 모두
  실패함(Ablation 4) — 라벨이 이미지당 1명뿐이라 detection head까지
  fine-tuning하는 게 근본적으로 어려움. 다인원 라벨 데이터 확보 또는
  keypoint 헤드만 분리 학습하는 방법이 향후 과제로 남음.
- 무단 구역 진입(zone intrusion)은 구현 완료(`src/zone_intrusion/zone_intrusion.py`)
  — 다만 지금은 고정 사각형 하나만 지원하고, 카메라 좌표를 실제 도면(BEV)에
  매핑하는 부분은 안 함. 실제 현장에 적용하려면 구역 좌표를 카메라별로
  수동 설정하는 과정이 필요함.
- VLM(Gemini) 버전은 CV 파이프라인과 낙상 판정 일치율이 43%로 낮고, 프롬프트
  문구를 바꾸자 같은 이미지에서도 판정이 뒤집히는 프롬프트 민감성을 확인함
  — 상용 배포 전 정답 라벨 기반 정밀 검증과 프롬프트 버저닝이 필요.

## 6. 발표 흐름 제안 (슬라이드 순서)

1. 문제 정의 (0절)
2. 전체 아키텍처 다이어그램 (1절) — "공유 백본 + 병렬 브랜치" 강조
3. Phase 1 실증: 탐지 실패율 50.6% → 왜 keypoint 신호가 필요한가 (Ablation 1)
4. Phase 2 데이터 전처리 시행착오 (Ablation 2, 2-보충) — "라벨 품질 문제를
   어떻게 발견하고 고쳤는가"의 스토리
5. Stage A vs Stage B (Ablation 3) — 혼동행렬로 어떤 사고 유형이 어려운지
6. Phase 3 PPE — 클래스 매핑 방법론(실측 시각화) + 학습 결과 + 간접연결 설계
7. pose 추출기 비교 + fine-tuning 2회 실패 스토리 (Ablation 4) — 실패를
   실측으로 진단하고 다음 결정을 내린 과정
8. 통합 파이프라인 데모 영상(3개 샘플) + VQA 웹앱 시연
9. Version 2 — VLM(Gemini) vs CV 파이프라인 비교(일치율 43%, 속도, 프롬프트
   민감도)
10. 한계와 향후 과제
