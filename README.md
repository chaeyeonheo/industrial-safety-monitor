# industrial-safety-monitor

산업 현장 CCTV 영상에서 (1) 낙상/실족과 (2) PPE 미착용·통제구역 무단진입을 감지해서
자연어 알람을 만드는 파이프라인. 사람 탐지+추적을 공유 백본으로 두고, 낙상/PPE/구역
세 브랜치를 같은 트랙 위에서 병렬로 판단한 뒤 이벤트를 하나로 통합한다.

CV 파이프라인 말고 VLM(Gemini) 한 번 호출로 같은 걸 판단하는 버전도 따로 만들어서
비교했다(`scripts/vlm_safety_check.py`) — 결과는 `docs/ablation_studies.md` 참고.

## 아키텍처

```
Video Frame
   │
   ▼
[YOLO11n person detect + ByteTrack]         ← 공유 백본 (src/detection_tracking)
   │
   ├──▶ [낙상 감지]  src/fall_detection, fall_mode 3종 중 택1
   │      - bbox_heuristic: 추적 bbox 종횡비/수직속도/추적유실 휴리스틱 (기본값)
   │      - keypoint_heuristic: RTMPose로 뽑은 keypoint에 같은 휴리스틱 적용
   │      - hdgcn: 같은 keypoint를 HD-GCN(5-way)에 태움 — 오프라인 81.8%였지만
   │        실시간 keypoint와는 분포가 달라 검출 0건(ablation_studies.md 참고),
   │        지금은 keypoint_heuristic을 실사용 기본안으로 씀
   │
   ├──▶ [PPE 미착용 판정]  src/ppe_detection
   │      YOLO11n 4클래스(헬멧/조끼/벨트/안전화) + 간접연결(중심점+신체부위 매칭,
   │      전체 bbox IoU는 부분 박스엔 항상 0에 가까워 못 씀). track 첫 관찰
   │      시점에만 판정하고 이후 고정(재판정 없음, 깜빡임 방지)
   │
   └──▶ [통제구역 진입]  src/zone_intrusion
          고정 사각형 zone에 사람 발 위치가 들어오는 순간(전환 시점)만 이벤트화

   ▼
[Event Aggregator(쿨다운)] → [NLG 템플릿] → 알람
   │
   ▼
[이벤트 타임라인 JSON] → [Gemini VQA 웹앱] → 자연어 질의응답
```

## 실행 환경

- OS: Windows 11 (Git Bash / PowerShell)
- Python 3.11, GPU: RTX 4070 Laptop (8GB VRAM), CUDA 12.4
- 모델 2개를 동시에 GPU에 올리면 하드웨어가 불안정해서(반복 크래시 확인), 탐지→PPE→pose
  모델을 완전히 분리된 순차 패스로 실행한다(`src/pipeline.py` `run_offline()`)

## 설치

```bash
pip install -r requirements.txt
```

## 명령어 모음

**PPE 데이터 변환 + 학습**
```bash
python scripts/convert_aihub163_to_yolo.py --max-frames 8000
python scripts/train_ppe_yolo.py --epochs 40 --batch 16
```

**낙상 분류기(HD-GCN) 데이터 변환 + 학습** (실시간 배포엔 안 쓰지만 ablation 재현용)
```bash
python scripts/convert_aihub163_keypoints_to_pyskl_transition.py
python scripts/train_hdgcn_fall.py --epochs 15 --batch 32
```

**전체 파이프라인 데모** (탐지+추적 → 낙상 → PPE → 구역 → 오버레이 영상)
```bash
# 프레임 번호 간격이 큰 지점마다 자동으로 클립을 나눠서 처리한다(src/frame_clips.py)
python scripts/demo_full_pipeline.py                                    # 전체 소스, 기본(bbox_heuristic)
python scripts/demo_full_pipeline.py --source S2-N6001_trip --fall-mode keypoint_heuristic
python scripts/demo_full_pipeline.py --compare-all                      # 3가지 fall_mode 전부 비교 생성
```

**VLM(Gemini) 버전 — 같은 클립에 대해 CV 파이프라인과 비교**
```bash
export GEMINI_API_KEY=발급받은_키
python scripts/vlm_safety_check.py --clip S2-N6001_trip_clip00
```

**VQA 웹앱**
```bash
export GEMINI_API_KEY=발급받은_키
python webapp/app.py
# http://localhost:5050
```
데모 결과(위 `demo_full_pipeline.py`)가 있는 클립만 웹앱 드롭다운에 나온다. 데모가 없으면
드래그앤드랍 영역에 mp4와 `event_timeline_*.json`을 같이 넣어도 된다.

## 데이터셋

- **AIHub 163** (공사현장 안전장비 인식): https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=163
  - Bounding Box / Keypoint(낙상·부딪힘·넘어짐·물체에 맞음) 라벨 포함
  - 회원가입 및 이용 동의 후 수동 다운로드, `data/raw/ppe_construction_aihub163/`에 배치
  - PPE 학습 이미지와 낙상 데모 영상이 같은 데이터셋 안에서도 촬영 장소가 달라서,
    PPE 모델을 낙상 데모 영상에 그대로 적용하면 정확도가 떨어지는 도메인 갭이 있음(실측 확인)
- **AIHub 71850** (CCTV 이상행동 — 쓰러짐): https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71850
  - 위 163과 달리 프레임이 아니라 원본 영상 통째로 제공되고, 사고 발생 구간(프레임 범위)이
    정답으로 붙어있어서, 낙상 검출의 실제 지연시간(ground truth 대비 몇 프레임 늦게
    잡는지)을 재는 데 썼다. `data/raw/cctv_anomaly_aihub71850/`에 배치

데이터가 없는 상태에서 관련 스크립트를 실행하면 필요한 다운로드 경로를 안내하는 메시지를
출력하고 해당 단계를 건너뛴다.

## 진행 상황

- [x] 탐지+추적 공유 백본
- [x] 낙상 감지 (bbox/keypoint 휴리스틱 + HD-GCN 검토 후 제외, ablation 포함)
- [x] PPE 미착용 판정
- [x] 통제구역 무단진입 감지
- [x] pose 백엔드 비교(YOLO11n-pose → RTMPose) + fine-tuning 2회 시도(모두 실패, 원인 진단)
- [x] VLM(Gemini) 버전 비교 실험
- [x] 데모 영상 + VQA 웹앱

전체 흐름과 실측 수치는 `docs/FINAL_SUMMARY.md`, ablation 결과는 `docs/ablation_studies.md`에
정리되어 있다.

## 라이선스 유의사항

AIHub 원본 데이터/영상은 재배포하지 않는다(`data/raw/`, `outputs/`는 `.gitignore` 처리).
결과물의 스크린샷 프레임만 `results/figures/`에 포함한다.
