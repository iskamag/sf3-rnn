"""SF3 environment for Fightcade FBNeo.

Screen frames are captured from the desktop with mss. Inputs are sent through
the FBNeo Lua bridge (`emulator/bot_bridge.lua`) by writing CPS3 bitfields to a
command file that Lua polls once per frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from config import Config
from model.action_space import Direction, Button, action_to_bitfield


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------
class SF3Env(ABC):
    """Minimal gym-like interface for SF3."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Start a new match and return the first RGB frame.  (H, W, 3) uint8."""

    @abstractmethod
    def step(self, direction: Direction,
             buttons: list[Button]) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action, return (frame, reward, done, info)."""

    @abstractmethod
    def close(self):
        """Tear down resources."""

    @abstractmethod
    def is_p1(self) -> bool:
        """Which side the bot controls."""

    # ------------------------------------------------------------------
    # Helper that most backends can share
    # ------------------------------------------------------------------
    @staticmethod
    def resize_frame(raw: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        # Crop 4:3 content area if needed (depends on capture source)
        h, w = raw.shape[:2]
        # Heuristic: assume a 4:3 game area centred in the capture
        aspect = 4 / 3
        if w / h > aspect:
            new_w = int(h * aspect)
            offset = (w - new_w) // 2
            raw = raw[:, offset:offset + new_w]
        elif w / h < aspect:
            new_h = int(w / aspect)
            offset = (h - new_h) // 2
            raw = raw[offset:offset + new_h, :]
        return cv2.resize(raw, (target_w, target_h))


# ---------------------------------------------------------------------------
# Screen-capture backend for Fightcade / FBNeo windows
# ---------------------------------------------------------------------------
class MssScreenEnv(SF3Env):
    """Capture the game window and send inputs through the FBNeo Lua bridge."""

    def __init__(self, cfg: Config, monitor: dict | None = None):
        self.cfg = cfg
        self.monitor = monitor or {"top": 0, "left": 0, "width": 1280, "height": 960}
        self._p1 = True
        self._mss = None
        self._input_path = Path(cfg.fbneo_input_path)

    def reset(self) -> np.ndarray:
        import mss
        self._mss = mss.mss()
        return self._capture()

    def step(self, direction: Direction,
             buttons: list[Button]) -> Tuple[np.ndarray, float, bool, dict]:
        self._send_input(direction, buttons)
        frame = self._capture()
        # Basic reward can be filled in by external orchestrator
        return frame, 0.0, False, {}

    def close(self):
        if self._mss:
            self._mss.close()
            self._mss = None

    def is_p1(self) -> bool:
        return self._p1

    # ------------------------------------------------------------------
    def _capture(self) -> np.ndarray:
        img = np.array(self._mss.grab(self.monitor))
        # mss returns BGRA → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return SF3Env.resize_frame(img, self.cfg.native_height, self.cfg.native_width)

    def _send_input(self, direction: Direction, buttons: list[Button]):
        bits = action_to_bitfield(direction, buttons)
        self._input_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._input_path.with_suffix(self._input_path.suffix + ".tmp")
        tmp.write_text(f"{bits}\n")
        tmp.replace(self._input_path)
