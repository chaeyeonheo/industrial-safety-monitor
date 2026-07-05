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

### 비디오 기반 트랙 ID 일관성 검증

**미실행.** 지시문 5장 5절 rule 5에 따라 존재를 확인할 수 없는 데모 영상 URL을 임의로
지어내지 않았음. AIHub 163 키포인트 원천데이터(넘어짐/떨어짐, 실제 사람이 등장하는 영상
프레임 시퀀스) 다운로드가 진행 중이며, 완료되는 대로 `scripts/demo_tracking.py --source`로
재실행해 실측 결과를 이 절에 채운다.
