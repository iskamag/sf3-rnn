"""Frame buffer enforcing human-like **sensory** reaction-time delay.

Accumulates greyscale frames.  At each game step returns the optical-flow
frame from `sensory_lag` frames ago — the stale visual input the human
visual pipeline would be delivering right now.

This handles one half of the latency model; the other half (motor lag)
is handled by `OutputDelayBuffer`.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import cv2
import numpy as np
import torch

from vision.optical_flow import compute_flow, preprocess_flow


class ReactionBuffer:
    def __init__(self,
                 sensory_lag: int = 8,       # frames of input latency
                 frame_size: int = 128,
                 flow_norm: float = 20.0,
                 device: str = "cuda",
                 ):
        self.sensory_lag = sensory_lag
        self.frame_size = frame_size
        self.flow_norm = flow_norm
        self.device = device

        # Need sensory_lag + 1 frames to produce a flow pair centred at
        # time t − sensory_lag given the current buffer at time t.
        self._frames: deque[np.ndarray] = deque(maxlen=sensory_lag + 2)

    # ------------------------------------------------------------------
    def add_frame(self, frame_rgb: np.ndarray):                # (H, W, 3) uint8
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.frame_size, self.frame_size))
        self._frames.append(gray)

    def ready(self) -> bool:
        return len(self._frames) >= self.sensory_lag + 2

    def get_flow(self) -> Optional[torch.Tensor]:
        """Delayed optical-flow frame, shape (1, 1, 2, H, W) on device."""
        if not self.ready():
            return None

        # At buffer time t we look at the frame pair centred on t − sensory_lag.
        prev = self._frames[-self.sensory_lag - 2]
        curr = self._frames[-self.sensory_lag - 1]

        raw = compute_flow(prev, curr)
        tensor = preprocess_flow(raw, self.frame_size, self.flow_norm)
        return tensor.unsqueeze(0).unsqueeze(0).to(self.device)

    def get_flow_at(self, offset: int) -> Optional[torch.Tensor]:
        """Return the flow frame from *offset* frames ago (offset ≥ 1)."""
        need = offset + 2
        if len(self._frames) < need:
            return None
        prev = self._frames[-offset - 2]
        curr = self._frames[-offset - 1]
        raw = compute_flow(prev, curr)
        tensor = preprocess_flow(raw, self.frame_size, self.flow_norm)
        return tensor.unsqueeze(0).unsqueeze(0).to(self.device)
