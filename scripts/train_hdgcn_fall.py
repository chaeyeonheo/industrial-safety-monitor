"""HD-GCN 낙상 분류기 소규모 학습.

사용자 요청("작게라도 해서 휴리스틱 단독 vs 학습 분류기 오탐률 비교 숫자 확보")에
따라 v2 데이터(전환감지 기반, data/processed/fall_keypoints_transition)로 작게
학습한다. GPU 크래시 이력 때문에 배치/에폭을 보수적으로 시작.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "HD-GCN"))

from model.HDGCN import Model  # noqa: E402  (third_party/HD-GCN)
import src.fall_detection.hdgcn_graph  # noqa: E402,F401  HD-GCN의 import_class가 getattr로 찾으므로 미리 import 필요
from src.fall_detection.hdgcn_dataset import FallWindowDataset  # noqa: E402

PICKLE_PATH = REPO_ROOT / "data/processed/fall_keypoints_transition/train_windows_transition.pkl"
NUM_CLASSES = 5  # falling_from_height/struck_by_collision/trip_and_fall/struck_by_object/normal
OUTPUT_DIR = REPO_ROOT / "outputs" / "hdgcn_runs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    if not PICKLE_PATH.exists():
        print(f"[train-hdgcn] 데이터가 없습니다: {PICKLE_PATH}. "
              f"먼저 scripts/convert_aihub163_keypoints_to_pyskl_transition.py를 실행하세요.")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[train-hdgcn] device={device} "
          f"({torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'})")

    dataset = FallWindowDataset(PICKLE_PATH)
    n_val = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"[train-hdgcn] 전체 {len(dataset)}개 중 train={n_train} val={n_val}")

    model = Model(
        num_class=NUM_CLASSES, num_point=16, num_person=1,
        graph="src.fall_detection.hdgcn_graph.Graph",
        graph_args={"labeling_mode": "spatial"},
        in_channels=3,
    ).to(device)

    if device != "cpu":
        free, total = torch.cuda.mem_get_info()
        print(f"[train-hdgcn] 모델 로드 후 GPU 메모리: 여유 {free/1e9:.2f}GB / 전체 {total/1e9:.2f}GB")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = ["epoch,train_loss,train_acc,val_acc"]

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for data, label, _ in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            correct += (out.argmax(1) == label).sum().item()
            total += data.size(0)
        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for data, label, _ in val_loader:
                data, label = data.to(device), label.to(device)
                out = model(data)
                val_correct += (out.argmax(1) == label).sum().item()
                val_total += data.size(0)
        val_acc = val_correct / val_total

        print(f"[train-hdgcn] epoch {epoch+1}/{args.epochs} "
              f"loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        log_lines.append(f"{epoch+1},{train_loss:.4f},{train_acc:.4f},{val_acc:.4f}")

    (OUTPUT_DIR / "results.csv").write_text("\n".join(log_lines))
    torch.save(model.state_dict(), OUTPUT_DIR / "hdgcn_fall_v2.pt")
    print(f"[train-hdgcn] 완료. 결과: {OUTPUT_DIR / 'results.csv'}, 가중치: {OUTPUT_DIR / 'hdgcn_fall_v2.pt'}")


if __name__ == "__main__":
    main()
