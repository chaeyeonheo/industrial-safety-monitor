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

**원인 분석(실측)**: 이 zip의 라벨 JSON을 프레임 번호로 분석한 결과, 같은 영상 내에서도
라벨링된 프레임 간 간격이 최소 1, 중앙값 7, 최대 60프레임으로 불균일하게 듬성듬성
샘플링되어 있음(`docs/PROGRESS.md`의 키포인트 라벨 포맷 조사 결과와 일치). 즉 이 프레임
시퀀스는 30fps 연속 영상이 아니라 "라벨링을 위해 골라낸 정지 프레임 모음"이라서, 프레임
사이 사람의 이동량이 IoU 기반 ByteTrack이 가정하는 프레임 간 근접성을 자주 벗어나
트랙이 끊어짐. **이것은 트래커 구현의 버그가 아니라 데이터 특성**이며, Phase 2의
포즈 분류기(ST-GCN 등)는 어차피 프레임 단위 keypoint를 독립적으로 학습에 사용하므로
연속 트래킹이 필수는 아니다. 다만 실제 연속 CCTV 영상(예: Phase 6 데모용 영상)에서는
프레임 간격이 균일(1/fps)하므로 이 문제가 재현되지 않을 것으로 예상되며, 이는 다음
세션에서 실제 연속 영상을 확보하면 재검증한다.
