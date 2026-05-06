"""Audio reaction buffer — shorter sensory lag than vision.

Collects raw audio samples from a running capture backend.  At each game
step returns a 15-d feature vector delayed by `audio_lag_frames` (default 3)
— ~50 ms, matching the faster auditory pathway.
"""

from __future__ import annotations

from collections import deque
import threading
from typing import Optional

import numpy as np
import torch

from vision.audio_features import extract_features


class AudioBuffer:
    def __init__(self,
                 audio_lag: int = 3,
                 sample_rate: int = 16000,
                 feat_dim: int = 15,
                 fps: int = 60,
                 device: str = "cuda",
                 ):
        self.audio_lag = audio_lag
        self.sample_rate = sample_rate
        self.feat_dim = feat_dim
        self.fps = fps
        self.device = device

        # Keep a power-of-two FFT window but advance on the video frame clock.
        self.fft_window = 256
        self._half_window = self.fft_window // 2
        self._samples_per_frame = sample_rate / fps
        self._features: deque[np.ndarray] = deque(maxlen=audio_lag + 8)
        self._samples = np.zeros(0, dtype=np.float32)
        self._sample_offset = 0
        self._next_frame = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def add_samples(self, samples: np.ndarray):               # (N,) float32
        """Push raw audio samples and emit features on the 60 FPS timeline."""
        if len(samples) == 0:
            return

        with self._lock:
            samples = np.asarray(samples, dtype=np.float32).reshape(-1)
            self._samples = np.concatenate([self._samples, samples])

            available_end = self._sample_offset + len(self._samples)
            while True:
                center = int(round((self._next_frame + 0.5) * self._samples_per_frame))
                start = center - self._half_window
                end = start + self.fft_window
                if end > available_end:
                    break

                local_start = max(start - self._sample_offset, 0)
                local_end = end - self._sample_offset
                chunk = np.zeros(self.fft_window, dtype=np.float32)
                dst_start = local_start - (start - self._sample_offset)
                chunk[dst_start:dst_start + (local_end - local_start)] = \
                    self._samples[local_start:local_end]
                self._features.append(extract_features(chunk, self.sample_rate))
                self._next_frame += 1

            keep_from = max(
                0,
                int(round((self._next_frame + 0.5) * self._samples_per_frame))
                - self._half_window,
            )
            drop = min(max(keep_from - self._sample_offset, 0), len(self._samples))
            if drop:
                self._samples = self._samples[drop:]
                self._sample_offset += drop

    def reset(self):
        with self._lock:
            self._features.clear()
            self._samples = np.zeros(0, dtype=np.float32)
            self._sample_offset = 0
            self._next_frame = 0

    def ready(self) -> bool:
        with self._lock:
            return len(self._features) >= self.audio_lag + 1

    def get_features(self) -> Optional[torch.Tensor]:
        """Return delayed audio features, (1, feat_dim) on device."""
        with self._lock:
            if len(self._features) < self.audio_lag + 1:
                return None

            # The feature at position -audio_lag - 1 is the one the auditory
            # pathway would be delivering right now.
            feats = self._features[-self.audio_lag - 1].copy()
        return torch.from_numpy(feats).unsqueeze(0).to(self.device)
