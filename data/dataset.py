"""PyTorch Dataset for pre-processed SF3 replays.

Preferred input is a shard directory from scripts/collect_data.py:
  flow_u8.npy        (T-1, 2, H, W) uint8    — quantized optical-flow frames
  audio_f16.npy      (T, 15)        float16  — per-video-frame audio features
  movement_u8.npy    (T,)           uint8    — P1 direction index 0-8
  buttons_u8.npy     (T, 6)         uint8    — P1 button states {0, 1}

Legacy compressed .npz files remain supported for old processed data.

The dataset randomly selects an ego player (P1 or P2) per sample and returns
that player's actions as targets, with health/stun swapped for ego-consistency.
An ego_player indicator bit (0=P1, 1=P2) is included for the model input.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.shard import dequantize_flow


class ReplayDataset(Dataset):
    def __init__(self,
                 data_dir: Path,
                 seq_len: int = 128,
                 total_delay: int = 12,
                 use_audio: bool = True,
                 segment_stride: int = 1,
                 cache_size: int = 2,
                 ):
        self.seq_len = seq_len
        self.total_delay = total_delay
        self.use_audio = use_audio
        segment_stride = max(1, segment_stride)
        self.segment_stride = segment_stride
        self.cache_size = max(0, cache_size)
        self._cache: OrderedDict[int, Dict[str, np.ndarray]] = OrderedDict()

        shard_dirs = sorted(
            p for p in data_dir.iterdir()
            if p.is_dir() and (p / "flow_u8.npy").exists())
        npz_files = sorted(data_dir.glob("*.npz"))
        self.files = shard_dirs + npz_files
        self._index: List[Tuple[int, int]] = []            # (source_idx, start)

        if not self.files:
            raise FileNotFoundError(f"No shard directories or .npz files in {data_dir}")

        for fi, fp in enumerate(self.files):
            if fp.is_dir():
                n_frames = np.load(fp / "movement_u8.npy", mmap_mode="r").shape[0]
                n_flows = np.load(fp / "flow_u8.npy", mmap_mode="r").shape[0]
            else:
                with np.load(fp) as d:
                    n_frames = d["movement"].shape[0]
                    n_flows = d["flows"].shape[0]

            # We need *seq_len* flows + *total_delay* frame offset for targets
            min_len = seq_len + total_delay
            if n_flows >= seq_len and n_frames >= min_len:
                limit = n_flows - seq_len + 1
                for start in range(0, limit, segment_stride):
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

    def _load_file(self, fi: int) -> Dict[str, np.ndarray]:
        cached = self._cache.get(fi)
        if cached is not None:
            self._cache.move_to_end(fi)
            return cached

        fp = self.files[fi]
        if fp.is_dir():
            loaded = {
                "_format": "shard",
                "flow_u8": np.load(fp / "flow_u8.npy", mmap_mode="r"),
                "audio": np.load(fp / "audio_f16.npy", mmap_mode="r"),
                "movement": np.load(fp / "movement_u8.npy", mmap_mode="r"),
                "buttons": np.load(fp / "buttons_u8.npy", mmap_mode="r"),
                "p2_movement": np.load(fp / "p2_movement_u8.npy", mmap_mode="r"),
                "p2_buttons": np.load(fp / "p2_buttons_u8.npy", mmap_mode="r"),
                "health_p1": np.load(fp / "health_p1_u8.npy", mmap_mode="r"),
                "health_p2": np.load(fp / "health_p2_u8.npy", mmap_mode="r"),
                "stun_p1": np.load(fp / "stun_p1_u8.npy", mmap_mode="r"),
                "stun_p2": np.load(fp / "stun_p2_u8.npy", mmap_mode="r"),
            }
        else:
            with np.load(fp) as data:
                loaded = {k: data[k] for k in data.files}
                loaded["_format"] = "npz"
        self._cache[fi] = loaded
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return loaded

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        fi, start = self._index[idx]
        delay = self.total_delay
        L = self.seq_len
        ego = random.randint(0, 1)  # 0 = P1, 1 = P2

        data = self._load_file(fi)
        if data["_format"] == "shard":
            fs = dequantize_flow(data["flow_u8"][start : start + L])
        else:
            fs = data["flows"][start : start + L].copy()          # (L, 2, H, W)

        t0 = start + delay
        if ego == 0:
            mv = data["movement"][t0 : t0 + L].copy()
            bt = data["buttons"][t0 : t0 + L].copy()
        else:
            mv = data["p2_movement"][t0 : t0 + L].copy()
            bt = data["p2_buttons"][t0 : t0 + L].copy()

        hp1 = data.get("health_p1", None)
        hp2 = data.get("health_p2", None)
        st1 = data.get("stun_p1", None)
        st2 = data.get("stun_p2", None)
        if hp1 is not None:
            if ego == 0:
                hp1 = hp1[t0 : t0 + L].copy()
                hp2 = hp2[t0 : t0 + L].copy()
                st1 = st1[t0 : t0 + L].copy() if st1 is not None else np.zeros(L, dtype=np.float32)
                st2 = st2[t0 : t0 + L].copy() if st2 is not None else np.zeros(L, dtype=np.float32)
            else:
                # Swap: our health = P2's, opponent = P1's
                hp1, hp2 = hp2[t0 : t0 + L].copy(), hp1[t0 : t0 + L].copy()
                st1, st2 = (st2[t0 : t0 + L].copy() if st2 is not None else np.zeros(L, dtype=np.float32),
                            st1[t0 : t0 + L].copy() if st1 is not None else np.zeros(L, dtype=np.float32))
            if data["_format"] == "shard":
                hp1 = hp1.astype(np.float32) / 255.0
                hp2 = hp2.astype(np.float32) / 255.0
        else:
            hp1 = hp2 = np.zeros(L, dtype=np.float32)
            st1 = st2 = np.zeros(L, dtype=np.float32)

        af = data.get("audio", None)
        if self.use_audio:
            if af is None:
                raise ValueError(
                    f"use_audio=True but {self.files[fi].name} has no 'audio' array")
            af = af[t0 : t0 + L].astype(np.float32, copy=True)
        else:
            af = np.zeros((L, 0), dtype=np.float32)

        return {
            "flows":      torch.from_numpy(fs.astype(np.float32, copy=False)),
            "movement":   torch.from_numpy(mv.astype(np.int64, copy=False)),
            "buttons":    torch.from_numpy(bt.astype(np.float32, copy=False)),
            "hp1":        torch.from_numpy(hp1.astype(np.float32, copy=False)),
            "hp2":        torch.from_numpy(hp2.astype(np.float32, copy=False)),
            "stun1":      torch.from_numpy(st1.astype(np.float32, copy=False)),
            "stun2":      torch.from_numpy(st2.astype(np.float32, copy=False)),
            "ego_player": ego,
            "audio":      torch.from_numpy(af.astype(np.float32, copy=False)),
        }
