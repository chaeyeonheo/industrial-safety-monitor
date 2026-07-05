# industrial-safety-monitor

Real-time industrial safety monitoring system using a multi-task vision pipeline
(pose-based fall detection & PPE compliance monitoring).

## Known Blockers

- AIHub 데이터(163: 공사현장 안전장비 인식, 71850: CCTV 이상행동)는 회원가입/이용약관 동의가
  필요해 완전 자동 다운로드가 불가능합니다. `aihubshell` CLI + 발급받은 API 키로 다운로드를
  진행 중이며, 진행 상황은 이 섹션에 계속 갱신합니다.
- (진행 중인 항목은 아래 Phase 체크리스트 참고)

## 문제 정의

산업 현장(공사현장/물류센터)에서 카메라 영상으로부터 (1) 낙상/사고성 이상행동과
(2) 개인보호구(PPE) 미착용·위험구역 침입을 실시간으로 감지해 자연어 알람을 생성하는
시스템. 사람 탐지+추적을 공유 백본으로 두고, 두 시나리오(낙상 감지 / PPE·구역침입)를
같은 트랙 위에서 병렬로 판단한 뒤 이벤트를 하나로 통합한다.

## 아키텍처 개요

```
Video Frame
   │
   ▼
[YOLO11 person detect + ByteTrack]  ← 공유 백본 (src/detection_tracking)
   │
   ├──▶ [Fall Detection]            src/fall_detection
   │      Stage A: bbox heuristic trigger
   │      Stage B: pose extractor → ST-GCN/CTR-GCN or 1D-CNN-LSTM
   │
   ├──▶ [PPE Detection]             src/ppe_detection
   │      간접 연결(IoU 매칭) or 직접 클래스 모델
   │
   └──▶ [Zone Intrusion]            src/zone_intrusion
          ROI geometry / BEV homography

   ▼
[Event Aggregator] → [NLG template] → 알람
```

## 실행 환경

- OS: Windows 11 (Git Bash / PowerShell)
- Python: 3.11.15
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM), Driver 591.74, CUDA 12.4 (torch 2.5.1+cu124)
- `torch.cuda.is_available() == True` 확인 완료

## 설치

```bash
pip install -r requirements.txt
```

## 데이터셋

- **AIHub 163** (공사현장 안전장비 인식): https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=163
  - Bounding Box / Polygon / Keypoint(낙상·부딪힘·넘어짐·물체에 맞음) 라벨 포함
  - 회원가입 및 이용 동의 후 수동 다운로드, `data/raw/ppe_construction_aihub163/`에 배치
- **AIHub 71850** (CCTV 이상행동): https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71850
  - 구역침입 검증 및 NLG 어휘 캘리브레이션 참고용(도메인 일반화 테스트, 학습에는 미사용)
  - `data/raw/cctv_anomaly_aihub71850/`에 배치

데이터가 없는 상태에서 관련 스크립트를 실행하면 필요한 다운로드 경로를 안내하는 메시지를
출력하고 해당 단계를 건너뜁니다.

## 진행 상황 (Phase 체크리스트)

- [x] Phase 0: 환경 설정
- [~] Phase 1: 탐지+추적 공유 백본 (코드 완료, 정지이미지 sanity check 완료, 비디오 트랙 일관성 검증은 AIHub 데이터 다운로드 대기 중)
- [ ] Phase 2: 낙상 감지
- [ ] Phase 3: PPE / 구역 침입
- [ ] Phase 4: Weakly-supervised VAD 비교 (선택)
- [ ] Phase 5: Latency 벤치마크
- [ ] Phase 6: 데모 영상 생성
- [ ] Phase 7: 최종 문서화

## 라이선스 유의사항

AIHub 원본 데이터/영상은 재배포하지 않습니다(`data/raw/`, `outputs/`는 `.gitignore` 처리).
결과물의 스크린샷 프레임이나 짧은 GIF만 README에 포함합니다.
