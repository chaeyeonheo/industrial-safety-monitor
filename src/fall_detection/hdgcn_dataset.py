"""v2(전환감지) pyskl 스타일 pickle을 HD-GCN Model이 기대하는 (C, T, V, M) 텐서로
변환하는 Dataset. HD-GCN은 in_channels=3을 하드코딩하므로(x, y, confidence) 3채널을
채운다."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FallWindowDataset(Dataset):
    def __init__(self, pickle_path: str | Path):
        with open(pickle_path, "rb") as f:
            self.samples = pickle.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]
        kp = s["keypoint"][0].astype(np.float32).copy()      # (T, V, 2)
        score = s["keypoint_score"][0].astype(np.float32)     # (T, V)
        h, w = s["img_shape"]
        kp[..., 0] /= w
        kp[..., 1] /= h

        data = np.concatenate([kp, score[..., None]], axis=-1)  # (T, V, 3)
        data = data.transpose(2, 0, 1)[..., None]                # (C, T, V, M=1)
        label = int(s["label"])
        return torch.from_numpy(data).float(), label, index
