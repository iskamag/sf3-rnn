"""PyTorch Dataset for pre-processed SF3 replays.

Input is a shard directory from scripts/collect_data.py:
  flow_u8.npy        (T-1, 2, H, W) uint8    — quantized optical-flow frames
  audio_f16.npy      (T, 15)        float16  — per-video-frame audio features
  movement_u8.npy    (T,)           uint8    — P1 direction index 0-8
  buttons_u8.npy     (T, 6)         uint8    — P1 button states {0, 1}

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
                 audio_offset: int = 0,
                 use_audio: bool = True,
                 segment_stride: int = 1,
                 cache_size: int = 2,
                 ):
        self.seq_len = seq_len
        self.total_delay = total_delay
        self.audio_offset = audio_offset
        self.use_audio = use_audio
        segment_stride = max(1, segment_stride)
        self.segment_stride = segment_stride
        self.cache_size = max(0, cache_size)
        self._cache: OrderedDict[int, Dict[str, np.ndarray]] = OrderedDict()

        shard_dirs = sorted(
            p for p in data_dir.iterdir()
            if p.is_dir() and (p / "flow_u8.npy").exists())
        self.files = shard_dirs
        self._index: List[Tuple[int, int, int]] = []       # (source_idx, start, jitter_end)
        self._sequences_by_file: List[List[Tuple[int, int]]] = [
            [] for _ in self.files]

        if not self.files:
            raise FileNotFoundError(f"No shard directories in {data_dir}")

        for fi, fp in enumerate(self.files):
            n_frames = np.load(fp / "movement_u8.npy", mmap_mode="r").shape[0]
            n_flows = np.load(fp / "flow_u8.npy", mmap_mode="r").shape[0]

            # Need seq_len visual flows, delayed targets, and audio aligned to
            # its shorter sensory path when audio is enabled.
            min_start = max(0, -audio_offset if use_audio else 0)
            max_start = min(n_flows - seq_len,
                            n_frames - total_delay - seq_len)
            if use_audio:
                max_start = min(max_start, n_frames - audio_offset - seq_len)

            if max_start >= min_start:
                for start in range(min_start, max_start + 1, segment_stride):
                    jitter_end = min(start + segment_stride - 1, max_start)
                    self._index.append((fi, start, jitter_end))
                    self._sequences_by_file[fi].append((start, jitter_end))

        random.shuffle(self._index)

    # ------------------------------------------------------------------
    def file_of(self, idx: int) -> Path:
        """Return the shard directory that index *idx* belongs to."""
        fi, _, _ = self._index[idx]
        return self.files[fi]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def num_sources(self) -> int:
        return len(self.files)

    def sequences_for_file(self, fi: int) -> List[Tuple[int, int]]:
        return self._sequences_by_file[fi]

    def _load_file(self, fi: int) -> Dict[str, np.ndarray]:
        cached = self._cache.get(fi)
        if cached is not None:
            self._cache.move_to_end(fi)
            return cached

        fp = self.files[fi]
        loaded = {
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
        self._cache[fi] = loaded
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return loaded

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        fi, start, _ = self._index[idx]
        ego = random.randint(0, 1)  # 0 = P1, 1 = P2
        return self.get_sequence(fi, start, ego)

    def get_sequence(self, fi: int, start: int, ego: int) -> dict[str, torch.Tensor]:
        delay = self.total_delay
        L = self.seq_len

        data = self._load_file(fi)
        fs = dequantize_flow(data["flow_u8"][start : start + L])

        t0 = start + delay
        if ego == 0:
            mv = data["movement"][t0 : t0 + L]
            bt = data["buttons"][t0 : t0 + L]
        else:
            mv = data["p2_movement"][t0 : t0 + L]
            bt = data["p2_buttons"][t0 : t0 + L]

        hp1 = data.get("health_p1", None)
        hp2 = data.get("health_p2", None)
        st1 = data.get("stun_p1", None)
        st2 = data.get("stun_p2", None)
        if hp1 is not None:
            if ego == 0:
                hp1 = hp1[t0 : t0 + L]
                hp2 = hp2[t0 : t0 + L]
                st1 = st1[t0 : t0 + L] if st1 is not None else np.zeros(L, dtype=np.float32)
                st2 = st2[t0 : t0 + L] if st2 is not None else np.zeros(L, dtype=np.float32)
            else:
                hp1, hp2 = hp2[t0 : t0 + L], hp1[t0 : t0 + L]
                st1, st2 = (st2[t0 : t0 + L] if st2 is not None else np.zeros(L, dtype=np.float32),
                            st1[t0 : t0 + L] if st1 is not None else np.zeros(L, dtype=np.float32))
            hp1 = hp1.astype(np.float32).copy() / 255.0
            hp2 = hp2.astype(np.float32).copy() / 255.0
        else:
            hp1 = hp2 = np.zeros(L, dtype=np.float32)
            st1 = st2 = np.zeros(L, dtype=np.float32)

        af = data.get("audio", None)
        if self.use_audio:
            if af is None:
                raise ValueError(
                    f"use_audio=True but {self.files[fi].name} has no 'audio' array")
            audio_start = start + self.audio_offset
            af = af[audio_start : audio_start + L].astype(np.float32)
        else:
            af = np.zeros((L, 0), dtype=np.float32)

        return {
            "flows":      torch.from_numpy(fs),
            "movement":   torch.from_numpy(np.asarray(mv, dtype=np.int64)),
            "buttons":    torch.from_numpy(np.asarray(bt, dtype=np.float32)),
            "hp1":        torch.from_numpy(np.asarray(hp1, dtype=np.float32)),
            "hp2":        torch.from_numpy(np.asarray(hp2, dtype=np.float32)),
            "stun1":      torch.from_numpy(np.asarray(st1, dtype=np.float32)),
            "stun2":      torch.from_numpy(np.asarray(st2, dtype=np.float32)),
            "ego_player": ego,
            "audio":      torch.from_numpy(af),
        }
