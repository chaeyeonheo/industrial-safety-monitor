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

## 현재 상태 (2026-07-05 23:00 기준, 컴퓨터 과부하로 세션이 여러 번 끊겨 자주 갱신 중)

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
- [ ] NLG 방식 3-way 비교 계획 확정: 템플릿(Jinja2) / Gemini API / Ollama 로컬 — 사용자 요청,
      Phase 2~3 완료 후 착수 예정. API 키는 사용자가 직접 발급. **아직 코드 착수 전.**
- [ ] Phase 2 나머지(1D-CNN-LSTM/ST-GCN/RGB baseline), Phase 3~7: 미착수

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

## 다음 세션에서 할 일 (재개 체크리스트)

1. 넘어짐 원천(filekey 559794)/물류센터 val 원천(filekey 559938) 다운로드가 사용자 쪽에서
   끝났는지 확인(zip 무결성 검사). 안 끝났으면 세션에서 다시 시도하지 말고 사용자에게 요청.
2. 키포인트 라벨 전수(23,840개 × 4카테고리) path별 그룹/프레임간격 분석
   (`keypoints/train/labels`의 4개 zip 압축 해제 후 분석 — 아직 안 함)
3. 위 분석 결과로 `scripts/convert_aihub163_keypoints_to_pyskl.py` 작성
4. Phase 2 나머지: pose 추출기, 1D-CNN-LSTM, ST-GCN(pyskl), RGB baseline 구현 및 비교
5. TRACK_LOST → best-effort pose 추출 오케스트레이션 (`docs/fall_detection_design.md` 3.3절
   설계는 되어 있으나 코드 미작성)
6. Phase 3(PPE/구역침입) 착수 — 물류센터 데이터는 이미 받아둠(images/train, labels/train)
