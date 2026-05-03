"""SF3 environment — abstracts the emulator / screen-capture backend.

Two backends included:
  1.  MssScreenEnv   — screen capture via mss + input via pynput
  2.  RetroArchEnv   — libretro FBNeo core via subprocess (retroarch + netplay stub)

Both expose the same interface so the training / inference code doesn't care
which one is wired up.
"""

from __future__ import annotations

import subprocess
import time
from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import numpy as np
import torch

from config import Config
from model.action_space import Direction, Button, bitfield_to_action


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
# Screen-capture backend (works with Fightcade / RetroArch desktop windows)
# ---------------------------------------------------------------------------
class MssScreenEnv(SF3Env):
    """Captures the desktop region containing the game window and injects
    virtual-keyboard / virtual-joystick inputs via pynput."""

    def __init__(self, cfg: Config, monitor: dict | None = None):
        self.cfg = cfg
        self.monitor = monitor or {"top": 0, "left": 0, "width": 1280, "height": 960}
        self._p1 = True
        self._mss = None
        self._input_device = None

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
        # Implement with pynput / vgamepad / keyboard according to your setup.
        # Placeholder — the actual mapping depends on emulator keybindings.
        pass


# ---------------------------------------------------------------------------
# libretro / RetroArch backend
# ---------------------------------------------------------------------------
class RetroArchEnv(SF3Env):
    """Launch RetroArch with the FBNeo core and communicate via stdio / TCP.

    Requires `retroarch` on PATH and a valid `fbneo` core library.
    """

    def __init__(self, cfg: Config,
                 core_path: str = "/usr/lib/libretro/fbneo_libretro.so",
                 rom_path: str = "sfiii3.zip",
                 port: int = 55355):
        self.cfg = cfg
        self.core_path = core_path
        self.rom_path = rom_path
        self.port = port
        self.proc = None
        self._p1 = True

    def reset(self) -> np.ndarray:
        self._launch()
        # Wait for game to boot …
        time.sleep(3)
        return self._read_frame()

    def step(self, direction: Direction,
             buttons: list[Button]) -> Tuple[np.ndarray, float, bool, dict]:
        self._write_input(direction, buttons)
        frame = self._read_frame()
        return frame, 0.0, False, {}

    def close(self):
        if self.proc:
            self.proc.kill()
            self.proc = None

    def is_p1(self) -> bool:
        return self._p1

    # ------------------------------------------------------------------
    def _launch(self):
        self.proc = subprocess.Popen([
            "retroarch",
            "-L", self.core_path,
            self.rom_path,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _read_frame(self) -> np.ndarray:
        # Stub — real implementation reads from libretro video buffer
        # via the network bridge or shared memory.
        return np.zeros((self.cfg.native_height, self.cfg.native_width, 3),
                        dtype=np.uint8)

    def _write_input(self, direction: Direction, buttons: list[Button]):
        # Stub — send via RetroArch network commands
        pass
