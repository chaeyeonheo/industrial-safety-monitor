# 진행 상황 체크포인트

GPU/CPU 자원이 넉넉하지 않아 세션이 언제 끊길지 모르므로, 다시 이어서 작업할 수 있도록
현재 상태를 여기에 계속 갱신한다. 새 세션에서 작업을 재개할 때 이 문서부터 읽을 것.

## 목표

산업 현장 CCTV 영상에서 (1) 낙상/사고성 이상행동, (2) PPE 미착용·위험구역 침입을
실시간으로 감지해 자연어 알람을 생성하는 파이프라인. 전체 계획은 최초 지시문(Phase 0~7)
기준이며, 오픈소스(Ultralytics, pyskl, mmpose 등)를 최대한 재사용하고 실제로 실행한
결과만 `results/RESULTS.md`에 기록한다(가상 수치 금지).

## 환경

- OS: Windows 11, 작업 디렉토리: `C:\Users\cyheo\Desktop\채연\Project\industrial-safety-monitor`
- GPU: NVIDIA RTX 4070 Laptop (8GB VRAM), CUDA 12.4, torch 2.5.1 — `torch.cuda.is_available()==True`
- 디스크 여유공간: 약 1.5TB (2026-07-05 기준)
- conda env 이름: `safety` (ultralytics가 `C:\Users\cyheo\miniconda3\envs\safety\Lib\site-packages`에 설치됨)
- git: 로컬 저장소 설정 완료, origin = `https://github.com/chaeyeonheo/industrial-safety-monitor.git`, 브랜치 `main`
- git 커밋 계정: user.name=chaeyeonheo, user.email=chaeyeonheo0@gmail.com (이 저장소에만 로컬 설정, --global 아님)

## AIHub 다운로드 도구

- `.aihub_tool/aihubshell` (공식 CLI, `curl -o aihubshell https://api.aihub.or.kr/api/aihubshell.do`로 받음, `.gitignore`에는 없지만 스크립트 자체라 커밋해도 무방 — 단, 폴더명 앞에 `.`이 있어 현재 안 보임)
- **API 키는 이 문서/깃에 절대 기록하지 않음.** 사용자가 발급받은 키를 세션에서 직접 사용.
  키가 필요하면 사용자에게 다시 요청할 것.
- datasetkey = `163` (공사현장 안전장비 인식 데이터, https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=163)
- 사용법: `./aihubshell -mode d -datasetkey 163 -filekey <key1>,<key2> -aihubapikey '<key>'`
  (다운로드는 실행한 디렉토리 기준으로 저장되며, 여러 파트로 나뉜 zip은 자동 병합됨. 압축 해제는 수동으로 `unzip` 필요)

### 파일키 참조 테이블 (재다운로드/이어받기용)

| 용도 | 경로(datasetkey=163 트리 내) | 크기 | filekey | 로컬 목적지 |
|---|---|---|---|---|
| 키포인트 라벨 train (떨어짐/부딪힘/넘어짐/물체에맞음) | 0.키포인트/1.Tranining/라벨링데이터(zip) | 11MB×4 | 559788,559789,559790,559791 | `data/raw/ppe_construction_aihub163/keypoints/train/labels` ✅ 완료 |
| 키포인트 라벨 val | 0.키포인트/2.Vaildation/라벨링데이터(zip) | 1MB×4 | 559796,559797,559798,559799 | `data/raw/ppe_construction_aihub163/keypoints/val/labels` (다운로드 중) |
| 키포인트 원천 train: 넘어짐, 떨어짐 | 0.키포인트/1.Tranining/원천데이터(zip) | 15GB+15GB | 559794,559792 | `data/raw/ppe_construction_aihub163/keypoints/train/source` (다운로드 중, 백그라운드) |
| 키포인트 원천 train: 부딪힘, 물체에맞음 (미다운로드) | 동일 | 18GB+18GB | 559793,559795 | 필요시 추가 다운로드 |
| 키포인트 원천 val: 넘어짐, 떨어짐 | 0.키포인트/2.Vaildation/원천데이터(zip) | 2GB+2GB | 559802,559800 | `data/raw/ppe_construction_aihub163/keypoints/val/source` (다운로드 중) |
| 물류센터 PPE bbox 라벨 train | 1.Training/라벨링데이터_241008_add | 165MB | 559929 | `data/raw/ppe_construction_aihub163/labels/train` (다운로드 중) |
| 물류센터 PPE bbox 라벨 val | 2.Validation/라벨링데이터(zip)_241008_add | 4MB | 559947 | `data/raw/ppe_construction_aihub163/labels/val` (다운로드 중) |
| 물류센터 원천 train (반도체 장비 클러스터, 최소 용량 우선 선택) | 1.Training/원천데이터_210818_add/5.물류센터 | 16GB | 559918 | `data/raw/ppe_construction_aihub163/images/train` (다운로드 중) |
| 물류센터 원천 train (화물터미널E/E2, 미다운로드) | 동일 | 34GB+38GB | 559919,559920 | 필요시 추가(용량 큼, 일단 보류) |
| 물류센터 원천 val | 2.Validation/원천데이터_210818_add/5.물류센터 | 2GB | 559938 | `data/raw/ppe_construction_aihub163/images/val` (다운로드 중) |

**설계 결정 근거**: 물류센터 원천 3종(16/34/38GB) 중 가장 작은 것 하나만 우선 다운로드해
Phase 3 PPE 학습 파이프라인을 먼저 검증하고, 필요하면 나머지 2종을 추가한다(사용자가
전체 1TiB 다운로드를 부담스러워해서 스코프를 줄임). 키포인트는 4종(떨어짐/부딪힘/넘어짐/
물체에맞음) 모두 원천데이터가 있으나, "낙상 감지(Phase 2)" 시나리오와 가장 직접적으로
관련된 넘어짐/떨어짐 2종만 우선 다운로드. 부딪힘/물체에맞음은 여유가 되면 추가.

### 키포인트 라벨 포맷 확인 결과 (2-1 항목)

`1.떨어짐.zip` (train)을 열어 확인함 (`keypoints/train/labels/_inspect_falling/`, 임시 검사용
폴더, 커밋 대상 아님):
- **단일 정지 이미지 단위 라벨**이다 (`{"image": {"filename": "S2-N4601M00103.jpg", "path": "S2-N4601M00001", ...}, "annotations": [{"point": [[x,y,v], ...16개 keypoint]}]}` 형태).
- 같은 `path` 값("S2-N4601M00001" 등)을 공유하는 파일이 다수 존재 → 하나의 원본 영상/시퀀스에서
  추출된 프레임들로 추정됨(파일명 뒤 숫자가 프레임 인덱스로 보이나 연속적이지 않고 듬성듬성
  건너뜀 — 103,104,105,106,107,108,109,112,113,116... 식으로 간격이 일정하지 않음).
- **전수 분석 완료** (`scripts/analyze_keypoint_labels.py`, 2026-07-05): 위에서 본 간격
  불균일은 **하나의 특이 케이스(validation 서브셋)였고, train 4개 카테고리 전체(95,798개
  파일, 7~10개 영상 그룹, 그룹당 최대 6929프레임)를 전수 분석하니 프레임 간격 중앙값이
  **1**(거의 연속)임을 확인함. 즉 train 셋은 실제로 연속에 가까운 영상 프레임이다.
  결론: ST-GCN/CTR-GCN 학습 시 같은 `path`로 그룹핑 후 프레임 번호로 정렬, gap>1인
  지점을 시퀀스 경계(장면전환)로 보고 분리한 뒤 슬라이딩 윈도우 구성. 상세 수치는
  `results/RESULTS.md` Phase 1 절의 "정정" 항목 참고.

## ⚠️ 컴퓨터가 반복적으로 꺼짐 (2026-07-06 새벽 기준, 최소 5~6회) — 읽고 시작할 것

**증상**: `clock_watchdog_timeout` BSOD가 반복됨. 처음엔 "AIHub 대용량 다운로드+zip
무결성 검사+YOLO 추론이 겹쳐서"라고 생각해 (1) GPU를 명시적으로 강제(`device="0"`),
(2) `torch.backends.cudnn.benchmark = False`(첫 추론 시 cuDNN 알고리즘 탐색이 만드는
메모리 스파이크 방지), (3) 50프레임마다 `torch.cuda.empty_cache()` 를 추가했다.

**그런데 사용자가 타당한 반론 제기**: `empty_cache()`는 유휴 메모리를 드라이버에
반환할 뿐이라, 반환 후 다음 프레임에서 다시 할당받는 저수준 드라이버 호출이 오히려
잦아져서 드라이버가 불안정하다면 도움이 안 되거나 더 나쁠 수 있다. 또한 "데이터를
잘게 쪼개서 넣자"는 제안에 대해서는 — `track_stream`이 이미 프레임 1개씩만 처리하는
스트리밍 구조라 더 쪼갤 여지가 없다는 것도 확인함.

**중요한 재평가**: 크래시가 (a) AIHub 다운로드(네트워크/디스크, GPU 무관), (b) zip
무결성 검사(순수 CPU, GPU 무관), (c) GPU 추론까지 **성격이 전혀 다른 작업들에서 공통
발생**하고 있다. 즉 이건 내 코드의 GPU 메모리 관리 문제라기보다 **지속적인 고부하
자체에 대한 하드웨어/발열/전원 쪽 불안정성일 가능성이 높다** — 소프트웨어 튜닝으로
근본 해결이 안 될 수 있음을 사용자에게 솔직히 전달함. 노트북 냉각/전원 어댑터 연결
여부, Windows 이벤트 뷰어의 정확한 BSOD 코드, GPU 드라이버 최신 여부 등 하드웨어/시스템
레벨 점검을 사용자에게 제안하는 것이 다음 단계로 필요할 수 있음.

**적용된 코드 변경(`src/detection_tracking/tracker.py`, 아직 커밋 안 함)**:
- `device` 자동감지 대신 명시적으로 `"0"`(CUDA) 강제, 실제 사용 디바이스를 로그 출력
- `torch.backends.cudnn.benchmark = False`
- 50프레임마다 `torch.cuda.empty_cache()` — **사용자 반론 이후 이 부분을 유지할지
  제거할지는 아직 결론 안 남. 다음 세션에서 판단 필요.**

## 현재 상태 (2026-07-06 새벽 기준, 컴퓨터 과부하로 세션이 여러 번 끊겨 자주 갱신 중)

- [x] Phase 0: 환경설정, 저장소 스캐폴딩, requirements.txt, README, pipeline.yaml (커밋+push 완료)
- [x] AIHub API 키 검증 완료 (datasetkey=163 파일트리 조회 성공, 사용자 제공 파일키와 일치 확인)
- [x] Phase 1: 탐지+추적 공유 백본 — **완전히 완료, 커밋 완료.**
  - `src/detection_tracking/tracker.py` (YOLO11n + ByteTrack, `weights/yolo11n.pt`)
  - 정지이미지(bus.jpg) sanity check: 사람 4명 정탐
  - **실제 AIHub 낙상 시퀀스(443프레임)로 비디오 트래킹 검증도 완료.** 핵심 발견:
    (1) 쓰러진/엎드린 자세는 YOLO11n이 아예 탐지 못함(bbox 없음) — 서 있을 때는 잘 잡힘
    (2) 라벨용 프레임은 간격이 불균일(중앙값 7, 최대 60프레임)해서 ByteTrack ID가 자주 끊김
        (트래커 버그 아니라 데이터 특성)
    상세: `results/RESULTS.md`, 스크린샷: `results/figures/tracking_demo_frame50.png`(정탐),
    `tracking_demo_frame150.png`(탐지 실패 사례)
- [~] **Phase 2 착수함(진행 중, 미완료, 미커밋)**: Stage A 휴리스틱 트리거
  - `src/fall_detection/heuristic_trigger.py` 작성 완료: aspect_ratio 급증 / 수직 하강 속도 /
    **TRACK_LOST(탐지 유실)** 3가지 트리거. TRACK_LOST는 위 (1)번 발견 때문에 추가함 — 서
    있으면 감지되던 track이 쓰러지는 순간 bbox 자체가 사라지는 걸 확인했기 때문.
  - 화면 가장자리 근처에서 사라진 경우(화면 이탈 가능성)는 confidence_hint를 낮추는 필터
    (`near_frame_edge`)도 추가함. 가려짐(occlusion)은 대부분 ByteTrack 자체 lost-buffer가
    흡수하므로 우리 쪽까지 오는 TRACK_LOST는 "화면 이탈" 아니면 "진짜 낙상/긴 가려짐" 둘 중
    하나라고 가정, 최종 확인은 Stage B(포즈 분류기, 아직 미구현) 또는 사람 확인 몫으로 둠.
  - `scripts/demo_fall_trigger.py` 실행 완료: 443프레임 중 트리거 3건(vertical_velocity_spike
    2건, track_lost 1건). track_lost 1건은 near_frame_edge=True로 필터가 의도대로 낮은
    신뢰도(0.2)를 매김. 실측치는 `results/RESULTS.md` Phase 2 절 참고.
  - **docs/fall_detection_design.md 작성 완료** (보고서용 상세 설계 문서, 커밋 완료).
  - GPU 명시 강제 완료: `PersonTracker`가 이제 device를 자동감지에 맡기지 않고 명시적으로
    CUDA(device="0")를 선택하며 실제 사용 디바이스를 로그로 출력함(사용자가 "CPU로 몰래
    떨어지지 않게 해달라"고 요청해서 반영).
- [x] 키포인트 라벨 → pyskl 포맷 변환 스크립트 v1 (`scripts/convert_aihub163_keypoints_to_pyskl.py`)
      **완료, 실행 검증 완료.** 결과물: `data/processed/fall_keypoints/train_windows.pkl`
      (6687개 윈도우, gitignore 대상 — 로컬에만 존재, 재실행하면 다시 생성됨).
      **v1의 근본 한계를 사용자가 지적**: 영상 하나 전체(최대 4000+프레임)에 폴더
      단위 라벨(예: "낙상")을 그대로 붙이는데, 실측해보니 실제 "누운 자세" 프레임은
      전체의 18.3%뿐이라 대부분의 윈도우가 "서 있는 모습"인데 "낙상"으로 잘못
      라벨링됐을 가능성이 높음.
- [x] **v2 구현 완료**: Stage A 휴리스틱(종횡비 급변)을 정답 keypoint에 직접 적용해
      실제 전환 순간을 검출하는 `scripts/convert_aihub163_keypoints_to_pyskl_transition.py`.
      v1은 그대로 보존(사용자가 나중에 ablation 비교 예정). 결과: 양성 1506개 +
      정상(normal, 지금까지 없던 클래스) 1209개 = 총 2715개. 단 낙상/넘어짐은 전환이
      500건+ 검출되는데 부딪힘/물체에맞음은 164~193건뿐이라(둘 다 "눕는 자세"까지
      안 갈 수 있어서) 카테고리별 검출 편향이 있음 — 이것도 한계로 기록해둠.
  - **`docs/data_preprocessing.md` 작성 완료** — v1/v2 시행착오 전체를 보고서용으로
    상세 기록(사용자가 나중에 이용할 것이라 요청).
- [x] **HD-GCN(ICCV 2023) `third_party/HD-GCN`에 clone 완료**. 원래 지시문은
      ST-GCN/CTR-GCN(pyskl)을 지정했지만, 사용자가 먼저 "HPI-GCN"(github.com/lizaowo/HPI-GCN,
      star 6개 미검증 소규모 저장소)을 제안 → 구조/코드 확인 후 별문제는 없었으나
      검증 수준이 낮음을 안내했더니 사용자가 **HD-GCN(Jho-Yonsei/HD-GCN, ICCV 2023,
      167 star, MIT license, 2s-AGCN/CTR-GCN 계보)으로 대신 제안** → 확인 후 채택.
      **아직 실제 학습 코드 통합/그래프 설정은 안 함** — 다음 세션 작업.
- [x] **HD-GCN을 git submodule로 전환 완료** (`git submodule add`, `.gitmodules` 커밋됨).
      third_party/는 이제 submodule 방식으로 관리(사용자 요청). `.gitignore`의 `third_party/`
      항목은 주석 처리.
- [x] **keypoint 16개 점 관절 매핑을 실제 이미지로 시각화해 추론 완료**
      (`docs/keypoint_mapping.md`, `results/figures/keypoint_index_mapping_frame{7,15}.png`).
      "중심선 4개(코/목/척추/골반) + 좌우대칭 6종×2=12" 패턴으로 16개가 맞아떨어짐을
      발견. **단 2프레임만 검증, 확정 아님** — HD-GCN 그래프 설정 전 추가 검증 권장.
- [x] **"추적만" vs "keypoint 전환감지 병행" 비교 데모 완료**
      (`scripts/demo_tracking_vs_transition.py`). 실측: 443프레임 중 YOLO 탐지 실패
      **224건(50.6%)**, 그중 **29건**은 keypoint 기반 전환감지가 대신 잡아냄. 나머지
      195건은 낙상과 무관한 이유로 탐지 실패(원인 미조사). 결과: `results/RESULTS.md`,
      예시 이미지 `results/figures/compare_tracking_miss_transition_catches.png`.
      **2026-07-06 수정 진행 중(미완료, 미커밋)**: 사용자 피드백 반영 —
      (a) 아래쪽(keypoint) 오버레이에도 bbox를 그려서 위/아래를 "박스 vs 박스"로
      직접 비교 가능하게 함, (b) 합본 영상뿐 아니라 tracking_only.mp4/transition_only.mp4
      개별 저장도 추가. **코드는 고쳤지만 컴퓨터가 다시 꺼져서 재실행 결과 확인 전.**
      다음 세션에서 `python scripts/demo_tracking_vs_transition.py` 재실행부터 할 것.
- [~] **PPE(안전모) YOLO 학습 착수 중** — 물류센터 라벨/이미지는 이미 받아둠
      (`data/raw/ppe_construction_aihub163/labels/train`, `images/train`). **라벨 포맷을
      아직 열어보지 못함**(다음 세션 최우선 작업 — bbox 좌표 형식, 클래스 목록 확인 후
      `scripts/convert_aihub163_to_yolo.py` 작성).
- [ ] NLG 방식 3-way 비교 계획 확정: 템플릿(Jinja2) / Gemini API / Ollama 로컬 — 사용자 요청,
      Phase 2~3 완료 후 착수 예정. API 키는 사용자가 직접 발급. **아직 코드 착수 전.**
- [ ] Phase 2 나머지(HD-GCN 실제 학습 — v1/v2 데이터 각각, ablation 비교, 1D-CNN-LSTM,
      RGB baseline), Phase 3(PPE 학습 진행 중, 구역침입 미착수), Phase 4~7: 미착수

## 문제-해결 기록 (사용자 요청 — "내가 문제를 해결한 방법"으로 보고서에 쓸 것)

### 문제 1: pyskl 변환 1차 시도에서 윈도우가 30개밖에 안 나옴

- **증상**: `window_length=30, max_gap_for_continuity=1`로 첫 실행했더니 그룹당 4000여
  프레임이 있는데도 전체 4개 카테고리 합쳐 윈도우가 **30개**밖에 안 나옴(사실상 영상
  그룹 수당 1개꼴).
- **원인 진단**: "프레임 간격 중앙값이 1"이라는 이전 분석 결과만 믿고 "거의 연속"이라고
  판단한 게 안일했다. 실제로 `max_gap_for_continuity=1`(끊김 허용 0)로 연속 구간을
  잘라보니, 사람이 프레임에서 2~3프레임씩 잠깐씩 사라지는 일이 매우 잦아서 4000프레임짜리
  그룹이 **median 4프레임짜리 조각 4795개**로 산산조각났다. (이 "잠깐 사라짐"은 Phase 1에서
  실측한 "쓰러진 자세를 탐지기가 놓친다"는 문제와 같은 뿌리일 가능성이 높음.)
- **해결**: 끊김 허용치(`max_gap_for_continuity`)를 1/3/5/10으로 바꿔가며 런 길이 분포를
  실측 비교(`results/RESULTS.md` Phase 2 pyskl 절의 표 참고). gap=3까지는 median 런
  길이가 4→61프레임으로 확 늘고, gap=5부터는 런 하나가 4000프레임에 육박해 "실제 장면
  전환까지 이어붙이는" 위험한 영역으로 넘어감. 그래서 **gap=3**을 채택하고, 끊긴 2~3프레임은
  버리지 않고 양 옆 실측 프레임을 선형 보간해서 채우되(`densify_run` 함수),
  보간된 프레임은 `keypoint_score=0`으로 표시해 나중에 학습 코드가 "이건 진짜 관측이
  아니라 보간값"이라고 구분할 수 있게 했다.
- **결과**: 윈도우 수가 30개 → **6687개**로 늘었고, 4개 클래스 분포도 1647~1698개로
  거의 균등했다.

### 문제 2: 컴퓨터가 반복적으로 꺼짐 (clock_watchdog_timeout)

- **증상**: 세션 중 최소 3번, "AIHub 대용량(15GB+) zip 다운로드 + 무결성 검증(체크섬
  계산)"을 진행하는 도중에 컴퓨터가 blue screen으로 꺼짐.
- **원인**: 명확히 진단은 못 함(하드웨어/드라이버 수준 오류라 코드로 알 수 없음). 다만
  다운로드(디스크 I/O)+zip 무결성 검사(CPU로 대용량 체크섬)+YOLO 추론이 겹쳐서 부하가
  쌓인 시점과 자꾸 겹쳤다.
- **해결(완화)**: (1) 무거운 작업을 동시에 겹치지 않고 하나씩 순차 진행, (2) 반복 실패한
  대용량 다운로드 2건(넘어짐 원천 15GB, 물류센터 val 원천 2GB)은 세션이 아니라 **사용자가
  직접 본인 터미널(Git Bash)에서 받는 것으로 전환**, (3) 매번 껐다 켜져도 이어갈 수 있도록
  이 문서(PROGRESS.md)를 각 단계마다 커밋.

### 문제 3: 콘다 환경 미지정으로 torch가 안 잡힘

- **증상**: 컴퓨터 재시작 후 새 세션에서 `python -c "import torch"`가 `ModuleNotFoundError`.
- **원인**: 새 셸이 `miniconda3\python.exe`(base, 3.13)를 기본으로 잡는데, ultralytics/torch는
  `miniconda3\envs\safety\python.exe`(3.11)에 설치되어 있었음.
- **해결**: 이후 모든 스크립트 실행은 `/c/Users/cyheo/miniconda3/envs/safety/python`
  전체 경로로 명시 호출.

## AIHub 다운로드 현재 상태 (2026-07-05 23:00, 반복 확인 필요)

컴퓨터가 여러 번 꺼지면서 백그라운드 다운로드가 예고 없이 중단되는 일이 반복됨.
**"완료" 메시지/exit code 0을 믿지 말고 매번 아래처럼 zip 무결성을 직접 검증할 것**:
```python
import zipfile; z = zipfile.ZipFile(path); z.testzip()  # None이면 정상, 예외면 손상
```

| 데이터 | 상태 |
|---|---|
| 키포인트 라벨 train 4종 | ✅ 완료 (검증됨) |
| 키포인트 라벨 val 4종 | ✅ 완료 (검증됨) |
| 키포인트 원천 val: 넘어짐+떨어짐 (~4GB) | ✅ 완료 (검증됨, `keypoints/val/source/`) |
| 키포인트 원천 train: 떨어짐 (~16GB) | ✅ 완료 (검증됨) |
| 키포인트 원천 train: 넘어짐 (~15GB) | ❌ 3번 연속 실패(다운로드 도중 컴퓨터가 매번 꺼짐). **사용자가 본인 터미널(Git Bash)에서 직접 재시도 중** (filekey 559794, datasetkey 163) |
| 물류센터 라벨 train/val | ✅ 완료 (검증됨) |
| 물류센터 원천 train (반도체클러스터, ~17GB) | ✅ 완료 (검증됨, `images/train/`) |
| 물류센터 원천 val (~2GB) | ❌ 실패. **사용자가 직접 재시도 중** (filekey 559938) |

**주의**: 위 두 건은 세션(백그라운드 bash)에서 시도할 때마다 컴퓨터가 clock_watchdog_timeout으로
꺼지는 패턴이 반복되어(2026-07-05 밤) 사용자가 본인 터미널에서 직접 받는 것으로 전환함.
재개 시 이 두 파일이 받아졌는지 먼저 확인(zip 무결성 검사)하고, 안 받아졌으면 다시 세션에서
시도하지 말고 사용자에게 직접 받아달라고 요청할 것.

## 다음 세션에서 할 일 (재개 체크리스트) — 2026-07-06 새벽 기준 최신

1. **git status 먼저 확인** — `src/detection_tracking/tracker.py`,
   `scripts/demo_tracking_vs_transition.py` 수정이 아직 미커밋 상태일 수 있음.
2. 넘어짐 원천(filekey 559794)/물류센터 val 원천(filekey 559938) 다운로드가 사용자 쪽에서
   끝났는지 확인(zip 무결성 검사, 위 표 참고). 안 끝났으면 세션에서 다시 시도하지 말 것.
3. `empty_cache()` 관련 사용자 반론(위 "컴퓨터가 반복적으로 꺼짐" 절 참고) — 유지할지
   제거할지 판단하고, 필요하면 되돌릴 것.
4. `python scripts/demo_tracking_vs_transition.py` 재실행해서 수정된 버전(박스 vs 박스
   비교, 개별 영상 3개 저장) 결과 확인 후 커밋.
5. **PPE 라벨 포맷 확인부터 시작** — `data/raw/ppe_construction_aihub163/labels/train`
   압축 해제해서 bbox 좌표 형식/클래스 목록 확인 → `scripts/convert_aihub163_to_yolo.py`
   작성 → `scripts/train_ppe_yolo.py`로 학습.
6. 키포인트 라벨 전수(23,840개 × 4카테고리) path별 그룹/프레임간격 분석은 이미 완료됨
   (`results/RESULTS.md`, `docs/data_preprocessing.md` 참고) — 재분석 불필요.
7. HD-GCN을 실제 학습 코드에 연결(그래프 설정에 `docs/keypoint_mapping.md`의 추론
   매핑 사용, v1/v2 데이터 각각으로 학습해 ablation 비교)
8. TRACK_LOST → best-effort pose 추출 오케스트레이션 (`docs/fall_detection_design.md` 3.3절
   설계는 되어 있으나 코드 미작성)
