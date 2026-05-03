"""PyTorch Dataset for pre-processed SF3 replays.

Each .npz file contains:
  flows      (T, 2, H, W) float32  — optical-flow frames (one per frame pair)
  movement   (T,)        int64     — direction index 0-8
  buttons    (T, 6)      float32   — button states {0, 1}
  health_p1  (T,)        float32   — P1 health 0-1
  health_p2  (T,)        float32   — P2 health 0-1
  audio      (T, 15)     float32   — audio features (required if use_audio=True)

The dataset yields sequences of *seq_len* flow frames where each LSTM
timestep receives one flow frame.  The reaction-time delay is baked into
the target offset:  target[t] = action at frame start + delay + t.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ReplayDataset(Dataset):
    def __init__(self,
                 data_dir: Path,
                 seq_len: int = 128,
                 total_delay: int = 12,
                 use_audio: bool = True,
                 ):
        self.seq_len = seq_len
        self.total_delay = total_delay
        self.use_audio = use_audio

        self.files = sorted(data_dir.glob("*.npz"))
        self._index: List[Tuple[int, int]] = []            # (file_idx, start)

        if not self.files:
            raise FileNotFoundError(f"No .npz files in {data_dir}")

        for fi, fp in enumerate(self.files):
            with np.load(fp) as d:
                n_frames = d["movement"].shape[0]
                n_flows = d["flows"].shape[0]

            # We need *seq_len* flows + *total_delay* frame offset for targets
            min_len = seq_len + total_delay
            if n_flows >= seq_len and n_frames >= min_len:
                limit = n_flows - seq_len + 1
                for start in range(0, limit):
                    if start + total_delay + seq_len <= n_frames:
                        self._index.append((fi, start))

        random.shuffle(self._index)

    # ------------------------------------------------------------------
    def file_of(self, idx: int) -> Path:
        """Return the .npz file that index *idx* belongs to."""
        fi, _ = self._index[idx]
        return self.files[fi]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        fi, start = self._index[idx]
        delay = self.total_delay
        L = self.seq_len

        with np.load(self.files[fi]) as data:
            fs = data["flows"][start : start + L].copy()              # (L, 2, H, W)

            t0 = start + delay
            mv = data["movement"][t0 : t0 + L].copy()                 # (L,)
            bt = data["buttons"][t0 : t0 + L].copy()                  # (L, 6)

            hp1 = data.get("health_p1", None)
            hp2 = data.get("health_p2", None)
            if hp1 is not None:
                hp1 = hp1[t0 : t0 + L].copy()
                hp2 = hp2[t0 : t0 + L].copy()
            else:
                hp1 = hp2 = np.zeros(L, dtype=np.float32)

            af = data.get("audio", None)
            if self.use_audio:
                if af is None:
                    raise ValueError(
                        f"use_audio=True but {self.files[fi].name} has no 'audio' array")
                af = af[t0 : t0 + L].copy()
            else:
                af = np.zeros((L, 0), dtype=np.float32)   # empty tensor

        return {
            "flows":    torch.from_numpy(fs.astype(np.float32, copy=False)),
            "movement": torch.from_numpy(mv.astype(np.int64, copy=False)),
            "buttons":  torch.from_numpy(bt.astype(np.float32, copy=False)),
            "hp1":      torch.from_numpy(hp1.astype(np.float32, copy=False)),
            "hp2":      torch.from_numpy(hp2.astype(np.float32, copy=False)),
            "audio":    torch.from_numpy(af.astype(np.float32, copy=False)),
        }
