# AIHub 163 keypoint 16개 점 — 관절 매핑 (실측 기반 추론, 공식 확정 아님)

## 배경

`docs/data_preprocessing.md`에서 "미해결"로 남겨뒀던 문제: AIHub 163 keypoint가
COCO-17 등 표준 레이아웃과 어떻게 대응되는지 알려주는 공식 문서를 확보하지
못했다. 사용자가 가진 AIHub 데이터 활용 안내 자료에는 "COCO-style의 **15개**
point"라고 되어 있는데, 실제 다운로드한 JSON에는 **16개** 점이 들어있어
숫자부터 안 맞는다. AIHub 페이지의 파일 트리를 다시 검색해봐도 별도 가이드
문서(가이드/안내서/매뉴얼 등 키워드)는 찾지 못했다.

그래서 **실제 이미지에 인덱스 번호를 찍어서 눈으로 확인하는 방식**으로
접근했다(사용자 제안). 아래는 그 결과이며, **공식 확정이 아니라 두 프레임의
시각적 대조로 추론한 best-effort 가설**이다.

## 방법

`data/raw/ppe_construction_aihub163/keypoints/val/labels/3.넘어짐.zip`의 라벨
JSON과, 미리 받아둔 같은 프레임의 원본 이미지(`keypoints/val/source/_frames_S2N6001/`)를
매칭해 16개 점을 이미지 위에 번호와 함께 그렸다.

- `results/figures/keypoint_index_mapping_frame7.png` — 뒷모습, point 0(코 추정)이
  안 보이는 프레임(visibility=0)
- `results/figures/keypoint_index_mapping_frame15.png` — 측후면, point 0이 보이는
  프레임(visibility=2) — 헬멧 위쪽에 점이 찍힘

## 추론된 매핑 (확정 아님)

| index | 추정 관절 | 근거 |
|---|---|---|
| 0 | 코/머리(nose/head) | frame15에서 헬멧 위, 얼굴 방향에 위치. 뒷모습(frame7)에서는 visibility=0(안 보임)으로 처리됨 — 얼굴 관련 점이라는 강한 정황 |
| 1 | 목(neck) | 0 바로 아래, 어깨 라인보다 위 |
| 2 | 가슴/상부 척추(chest/upper spine) | 1 바로 아래, 3·4(어깨)와 거의 같은 높이의 중앙 |
| 3 | 오른쪽 어깨(shoulder R) | 2 기준 이미지상 오른쪽, 4와 대칭 |
| 4 | 왼쪽 어깨(shoulder L) | 2 기준 이미지상 왼쪽, 3와 대칭 |
| 5 | 오른쪽 팔꿈치(elbow R) | 3 아래쪽 |
| 6 | 왼쪽 팔꿈치(elbow L) | 4 아래쪽 |
| 7 | 오른쪽 손목(wrist R) | 5 아래쪽, frame7에서 팔을 뻗은 자세일 때 팔 끝에 위치 |
| 8 | 왼쪽 손목(wrist L) | 6 아래쪽 |
| 9 | 골반 중앙(pelvis center) | 몸통 중앙, 10·11과 같은 높이 |
| 10 | 오른쪽 엉덩이(hip R) | 9 기준 오른쪽 |
| 11 | 왼쪽 엉덩이(hip L) | 9 기준 왼쪽 |
| 12 | 오른쪽 무릎(knee R) | 10 아래쪽 |
| 13 | 왼쪽 무릎(knee L) | 11 아래쪽 |
| 14 | 오른쪽 발목(ankle R) | 12 아래쪽 |
| 15 | 왼쪽 발목(ankle L) | 13 아래쪽 |

**패턴**: 코(1) + 목(1) + 척추중심(1) + 골반중심(1) = 중심선 4개, 나머지
12개가 어깨/팔꿈치/손목/엉덩이/무릎/발목의 좌우 쌍(6종류×2) — 정확히
4+12=16개로 맞아떨어진다. "COCO-style 15개"는 이 중 **점 0(코)을 얼굴
관련 별도 항목으로 세거나 아예 제외하고 몸통 15개만 셌을 가능성**이 있다고
추정한다(확인 안 됨).

## 신뢰도와 한계

- **두 프레임만으로 검증**했다. 다른 자세(정면, 앉은 자세, 낙상 도중 등)에서
  좌우가 실제로 이 순서와 일치하는지는 확인하지 않았다.
- 특히 1(목) vs 2(척추)의 구분은 두 점이 너무 가까이 붙어있어(frame15 기준
  픽셀 거리 약 33px) 순서가 바뀌었을 가능성을 배제할 수 없다.
- 9(골반중심)와 7(오른쪽 손목)이 시각적으로 겹쳐 보이는 프레임이 있어(frame15),
  팔이 몸통 옆에 붙어 있는 자세에서는 그림만으로 구분이 어렵다.
- **HD-GCN 그래프(인접행렬) 설정에 이 매핑을 쓰기 전에, 최소한 3~5개 프레임을
  더 다른 자세로 교차검증하는 것을 권장한다.** 지금은 시간 관계상 2프레임만
  확인한 상태로 다음 작업(HD-GCN 학습 통합)을 진행하되, 이 문서를 계속
  갱신할 것.

## 재현 방법

```python
import zipfile, json, cv2
z = zipfile.ZipFile("data/raw/ppe_construction_aihub163/keypoints/val/labels/3.넘어짐.zip")
with z.open("S2-N6001M00015.json") as f:
    d = json.load(f)
pts = d["annotations"][0]["point"]  # 16개, 각 [x, y, visibility(0/1/2)]
img = cv2.imread("data/raw/.../keypoints/val/source/_frames_S2N6001/S2-N6001M00015.jpg")
for i, (x, y, v) in enumerate(pts):
    cv2.circle(img, (int(x), int(y)), 6, (0,255,0), -1)
    cv2.putText(img, str(i), (int(x)+8, int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
```
