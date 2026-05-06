"""SF3 environment for Fightcade FBNeo — gymnasium interface.

The environment reads actual game state (health, wins) from the FBNeo Lua
bridge instead of peeping at pixels.  Reward is computed from real memory
values: damage dealt/taken per step, round/match win bonuses, and a bonus
for remaining health when winning a round.

Sensory lag (ReactionBuffer) lives in the env; motor lag lives in the
training loop so that stored (action, log_prob, reward) triples align.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from config import Config
from model.action_space import Direction, Button, action_to_bitfield
from model.reaction_buffer import ReactionBuffer


@dataclass
class GameState:
    """Canonical game state from the Lua bridge."""
    p1_health: float = 0.0
    p2_health: float = 0.0
    p1_wins: int = 0
    p2_wins: int = 0
    p1_stun: int = 0
    p2_stun: int = 0
    timer: int = 0

    @staticmethod
    def parse(line: str) -> GameState:
        parts = line.strip().split()
        if len(parts) < 7:
            return GameState()
        return GameState(
            p1_health=float(parts[0]),
            p2_health=float(parts[1]),
            p1_wins=int(parts[2]),
            p2_wins=int(parts[3]),
            p1_stun=int(parts[4]),
            p2_stun=int(parts[5]),
            timer=int(parts[6]),
        )


class SF3Env(gym.Env):
    """Gymnasium environment wrapping Fightcade FBNeo.

    ``step(action)`` sends the motor-delayed action to the emulator, reads
    actual game state from the Lua bridge, and returns a reward based on
    real health delta, round/match wins, and remaining health.

    The observation is a delayed optical-flow frame from the ReactionBuffer
    (sensory lag). Reward comes from the Lua bridge game state.
    """

    def __init__(self, cfg: Config, monitor: dict | None = None):
        super().__init__()
        self.cfg = cfg
        self.monitor = monitor or {"top": 0, "left": 0, "width": 1280, "height": 960}
        self._p1 = True

        # ── spaces ──────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(2, cfg.frame_height, cfg.frame_width),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([9, 2, 2, 2, 2, 2, 2])

        # ── sensory lag ─────────────────────────────────────────────────
        self._sensory_buf: ReactionBuffer | None = None

        # ── screen capture / Lua bridge ─────────────────────────────────
        self._mss: Optional["mss.mss"] = None
        self._input_path = Path(cfg.fbneo_input_path)
        self._state_path = Path(cfg.fbneo_state_path)

        # ── reward tracking ─────────────────────────────────────────────
        self._prev_hp_self: float = 1.0
        self._prev_hp_opp: float = 1.0
        self._prev_wins_self: int = 0

    # ------------------------------------------------------------------
    # gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        import mss
        if self._mss is not None:
            self._mss.close()
        self._mss = mss.mss()

        device_str = self.cfg.device if torch.cuda.is_available() else "cpu"
        self._sensory_buf = ReactionBuffer(
            sensory_lag=self.cfg.sensory_lag_frames,
            extra=self.cfg.temporal_stride,
            frame_width=self.cfg.frame_width,
            frame_height=self.cfg.frame_height,
            flow_norm=self.cfg.flow_norm,
            device=device_str,
        )

        self._prev_hp_self = 1.0
        self._prev_hp_opp = 1.0
        self._prev_wins_self = 0

        # Prime the reaction buffer.
        prime_frames = self.cfg.sensory_lag_frames + 2
        frame = self._capture()
        for _ in range(prime_frames):
            self._sensory_buf.add_frame(frame)
            frame = self._capture()

        # First valid (delayed) observation.
        obs = self._sensory_buf.get_flow()
        if obs is None:
            obs = np.zeros((2, self.cfg.frame_height, self.cfg.frame_width), dtype=np.float32)
        else:
            obs = obs.squeeze().cpu().numpy()

        return obs, {}

    def step(self, action: np.ndarray
             ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute *action* and return (obs, reward, terminated, truncated, info).

        *action* is the motor-delayed action. Reward is computed from the
        game state read from the Lua bridge.
        """
        dir_int = int(action[0])
        buttons = [Button(i) for i in range(6) if action[i + 1] == 1]

        # Send the (already delayed) action to the emulator.
        if dir_int != 0 or buttons:
            self._send_input(Direction(dir_int), buttons)
        else:
            self._send_input(Direction.NEUTRAL, [])

        # Capture frame for optical flow observation.
        frame = self._capture()
        self._sensory_buf.add_frame(frame)

        # Read actual game state from Lua bridge.
        gs = self._read_state()

        # Compute reward from real values.
        reward = self._compute_reward(gs, bool(dir_int != 0 or buttons))

        # Sensory-delayed flow observation.
        obs_flow = self._sensory_buf.get_flow()
        if obs_flow is not None:
            obs = obs_flow.squeeze().cpu().numpy()
        else:
            obs = np.zeros((2, self.cfg.frame_height, self.cfg.frame_width), dtype=np.float32)

        # Terminate when the health bars disappear.
        terminated = self._health_bars_hidden(gs)
        truncated = False

        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------

    def render(self):
        pass

    def close(self):
        if self._mss is not None:
            self._mss.close()
            self._mss = None

    # ------------------------------------------------------------------
    # reward
    # ------------------------------------------------------------------

    def _compute_reward(self, gs: GameState, any_button: bool) -> float:
        cfg = self.cfg
        if self._p1:
            hp_self, hp_opp = gs.p1_health, gs.p2_health
            wins_self = gs.p1_wins
        else:
            hp_self, hp_opp = gs.p2_health, gs.p1_health
            wins_self = gs.p2_wins

        dmg_dealt = max(0.0, self._prev_hp_opp - hp_opp)
        dmg_taken = max(0.0, self._prev_hp_self - hp_self)
        rounds_won = max(0, wins_self - self._prev_wins_self)
        match_won = 1 if wins_self >= 2 and self._prev_wins_self < 2 else 0

        reward = cfg.reward_health_delta * (dmg_dealt - dmg_taken)
        reward += cfg.reward_round_win * rounds_won
        reward += cfg.reward_match_win * match_won
        if rounds_won > 0:
            reward += cfg.reward_remaining_health * hp_self
        reward += cfg.reward_idle_penalty if not any_button else 0.0

        self._prev_hp_self = hp_self
        self._prev_hp_opp = hp_opp
        self._prev_wins_self = wins_self
        return reward

    # ------------------------------------------------------------------
    # Lua bridge state file
    # ------------------------------------------------------------------

    def _read_state(self) -> GameState:
        try:
            text = self._state_path.read_text().strip()
            if not text:
                return GameState()
            return GameState.parse(text)
        except (FileNotFoundError, OSError, ValueError, IndexError):
            return GameState()

    @staticmethod
    def _health_bars_hidden(gs: GameState) -> bool:
        """Return True when the HUD health bars are absent."""
        return gs.p1_health < 0.05 and gs.p2_health < 0.05

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _capture(self) -> np.ndarray:
        img = np.array(self._mss.grab(self.monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return self._resize_frame(img, self.cfg.native_height, self.cfg.native_width)

    def _send_input(self, direction: Direction, buttons: list[Button]):
        bits = action_to_bitfield(direction, buttons)
        self._input_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._input_path.with_suffix(self._input_path.suffix + ".tmp")
        tmp.write_text(f"{bits}\n")
        tmp.replace(self._input_path)

    def _resize_frame(raw: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h, w = raw.shape[:2]
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


class MssScreenEnv:
    """Minimal screen-capture environment for ``scripts/run_bot.py``.

    Captures the game window and sends inputs via the Lua bridge.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._mss = None
        self._input_path = Path(cfg.fbneo_input_path)

    def reset(self) -> np.ndarray:
        import mss
        self._mss = mss.mss()
        return self._capture()

    def step(self, direction: Direction, buttons: list[Button]) -> tuple:
        bits = action_to_bitfield(direction, buttons)
        self._input_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._input_path.with_suffix(self._input_path.suffix + ".tmp")
        tmp.write_text(f"{bits}\n")
        tmp.replace(self._input_path)
        frame = self._capture()
        return frame, 0.0, False, False, {}

    def close(self):
        if self._mss is not None:
            self._mss.close()
            self._mss = None

    def _capture(self) -> np.ndarray:
        import mss
        if self._mss is None:
            self._mss = mss.mss()
        monitor = {"top": 0, "left": 0, "width": 1280, "height": 960}
        img = np.array(self._mss.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return SF3Env._resize_frame(img, self.cfg.native_height, self.cfg.native_width)
