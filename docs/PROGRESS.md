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
- **결론**: ST-GCN/CTR-GCN처럼 연속 시퀀스가 필요한 시계열 분류기를 학습하려면, 같은 `path`를
  공유하는 프레임들을 그룹핑하고 프레임 번호로 정렬한 뒤 슬라이딩 윈도우를 구성하는 전략이
  필요하다(프레임 간격이 일정하지 않으므로 보간 또는 시간축 리샘플링 고려).
  전체 23,840개 파일에 대한 path별 그룹 크기/프레임 범위 전수 분석은 아직 미완료
  (다음 세션에서 이어서 할 것 — `scripts/convert_aihub163_keypoints_to_pyskl.py` 작성 전에 필요).

## 현재 상태 (2026-07-05 기준)

- [x] Phase 0: 환경설정, 저장소 스캐폴딩, requirements.txt, README, pipeline.yaml (커밋 완료, push 완료)
- [x] AIHub API 키 검증 완료 (datasetkey=163 파일트리 조회 성공, 사용자 제공 파일키와 일치 확인)
- [~] AIHub 데이터 다운로드 진행 중 (백그라운드 2개 프로세스, 위 테이블 참고)
  - 하나는 아직 명령/프로세스 상태를 세션 재시작 후 알 수 없음 → 재개 시 `ls -la` 로 실제
    다운로드된 파일 존재 여부/크기를 먼저 확인해서 어디까지 받았는지 판단할 것
- [ ] Phase 1: 탐지+추적 공유 백본 (진행 예정, AIHub 데이터 불필요 — YOLO11 pretrained + 샘플
      영상만 있으면 됨. 다운로드와 병렬로 진행 중)
- [ ] Phase 2~7: 미착수

## 다음 세션에서 할 일 (재개 체크리스트)

1. `data/raw/ppe_construction_aihub163/` 하위 각 폴더에 실제로 몇 GB가 받아졌는지 확인
   (`du -sh` 또는 `ls -la`)
2. 다운로드가 끊겼다면 위 파일키 테이블로 이어받기
3. 키포인트 라벨 전수(23,840개 × 4카테고리) path별 그룹 분석 완료 후
   `scripts/convert_aihub163_keypoints_to_pyskl.py` 작성
4. Phase 1 코드(`src/detection_tracking/`) 진행 상황 확인 후 이어서 구현
