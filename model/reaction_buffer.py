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
        # Need sensory_lag + 2 frames for a single flow pair; add extra
        # headroom for multi-frame stride-N models (default 4 extra slots).
        extra: int = 4,
        frame_width: int = 160,
        frame_height: int = 120,
        flow_norm: float = 20.0,
        device: str = "cuda",
                 ):
        self.sensory_lag = sensory_lag
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.flow_norm = flow_norm
        self.device = device

        self._frames: deque[np.ndarray] = deque(maxlen=sensory_lag + extra + 2)

    # ------------------------------------------------------------------
    def add_frame(self, frame_rgb: np.ndarray):                # (H, W, 3) uint8
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.frame_width, self.frame_height))
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
        tensor = preprocess_flow(raw, self.frame_width, self.frame_height, self.flow_norm)
        return tensor.unsqueeze(0).unsqueeze(0).to(self.device)

