"""Audio reaction buffer — shorter sensory lag than vision.

Collects raw audio samples from a running capture backend.  At each game
step returns a 15-d feature vector delayed by `audio_lag_frames` (default 3)
— ~50 ms, matching the faster auditory pathway.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import torch

from vision.audio_features import extract_features


class AudioBuffer:
    def __init__(self,
                 audio_lag: int = 3,
                 sample_rate: int = 16000,
                 feat_dim: int = 15,
                 device: str = "cuda",
                 ):
        self.audio_lag = audio_lag
        self.sample_rate = sample_rate
        self.feat_dim = feat_dim
        self.device = device

        # 256 samples @ 16 kHz ≈ 16 ms  (close to 1/60 s, power-of-2 FFT)
        self.spf = 256
        self._chunks: deque[np.ndarray] = deque(maxlen=audio_lag + 2)

    # ------------------------------------------------------------------
    def add_samples(self, samples: np.ndarray):               # (N,) float32
        """Push raw audio samples.  Internally splits into 256-sample chunks."""
        spf = self.spf
        for i in range(0, len(samples) - spf + 1, spf):
            self._chunks.append(samples[i : i + spf])

    def ready(self) -> bool:
        return len(self._chunks) >= self.audio_lag + 1

    def get_features(self) -> Optional[torch.Tensor]:
        """Return delayed audio features, (1, feat_dim) on device."""
        if not self.ready():
            return None

        # The chunk at position -audio_lag - 1 is the one the auditory
        # pathway would be delivering right now.
        chunk = self._chunks[-self.audio_lag - 1]
        feats = extract_features(chunk, self.sample_rate)
        return torch.from_numpy(feats).unsqueeze(0).to(self.device)
