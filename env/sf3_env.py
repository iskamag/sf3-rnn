"""SF3 environment for Fightcade FBNeo — gymnasium interface.

The environment models sensory lag (ReactionBuffer delays observations).
Motor lag is handled externally by the training loop so that the stored
(log_prob, action, reward) triples remain properly aligned.

Screen frames are captured with mss. Inputs are sent through the FBNeo Lua
bridge by writing CPS3 bitfields to a command file.
"""

from __future__ import annotations

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
from train.reward import compute_step_reward, RewardConfig
from vision.game_state import (
    extract_game_state, GameState, WinTracker, has_health_bar,
)


class SF3Env(gym.Env):
    """Gymnasium environment wrapping Fightcade FBNeo screen capture.

    ``step(action)`` immediately sends *action* to the emulator.  The
    returned observation is a delayed optical-flow frame from the
    ReactionBuffer (sensory lag).  Reward is computed from health-bar
    pixel analysis.
    """

    def __init__(
        self,
        cfg: Config,
        monitor: dict | None = None,
        reward_cfg: RewardConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.monitor = monitor or {"top": 0, "left": 0, "width": 1280, "height": 960}
        self._p1 = True
        self._reward_cfg = reward_cfg or RewardConfig(
            health_delta=cfg.reward_health_delta,
            round_win=cfg.reward_round_win,
            match_win=cfg.reward_match_win,
            idle=cfg.reward_idle_penalty,
        )

        # ── spaces ──────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(2, cfg.frame_height, cfg.frame_width),
            dtype=np.float32,
        )
        # Action received here is the *motor-delayed* action (already through
        # OutputDelayBuffer), not the raw policy output.
        self.action_space = spaces.MultiDiscrete([9, 2, 2, 2, 2, 2, 2])

        # ── sensory lag ─────────────────────────────────────────────────
        self._sensory_buf: ReactionBuffer | None = None

        # ── screen capture ──────────────────────────────────────────────
        self._mss: Optional["mss.mss"] = None
        self._input_path = Path(cfg.fbneo_input_path)

        # ── health / reward tracking ────────────────────────────────────
        self._prev_gs: Optional[GameState] = None
        self._win_tracker = WinTracker()
        self._roi = {
            "p1_health": cfg.p1_health_roi,
            "p2_health": cfg.p2_health_roi,
            "p1_stun":   cfg.p1_stun_roi,
            "p2_stun":   cfg.p2_stun_roi,
        }

    # ------------------------------------------------------------------
    # gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        import mss
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
        self._prev_gs = None
        self._win_tracker = WinTracker()

        # Navigate menus until gameplay starts.
        frame = self._navigate_menu()

        # Prime the reaction buffer so get_flow() returns a valid result.
        prime_frames = self.cfg.sensory_lag_frames + 2
        for _ in range(prime_frames):
            self._sensory_buf.add_frame(frame)
            frame = self._capture()

        # Initial health state.
        self._prev_gs = self._win_tracker.update(extract_game_state(frame, self._roi))

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

        *action* is the motor-delayed action (direction + 6 buttons) —
        the training loop applies OutputDelayBuffer before calling step().
        The observation is the delayed optical flow from the ReactionBuffer.
        """
        dir_int = int(action[0])
        buttons = [Button(i) for i in range(6) if action[i + 1] == 1]

        # Send the (already delayed) action to the emulator.
        if dir_int != 0 or buttons:
            d = Direction(dir_int)
            self._send_input(d, buttons)
        else:
            self._send_input(Direction.NEUTRAL, [])

        # Capture frame and push through sensory delay.
        frame = self._capture()
        self._sensory_buf.add_frame(frame)

        obs_flow = self._sensory_buf.get_flow()
        if obs_flow is not None:
            obs = obs_flow.squeeze().cpu().numpy()
        else:
            obs = np.zeros((2, self.cfg.frame_height, self.cfg.frame_width), dtype=np.float32)

        # Reward from health-delta between frames.
        curr_gs = self._win_tracker.update(extract_game_state(frame, self._roi))
        any_button = bool(dir_int != 0 or len(buttons) > 0)
        reward = compute_step_reward(
            self._prev_gs, curr_gs, self._p1, self._reward_cfg, any_button,
        ) if self._prev_gs is not None else 0.0
        self._prev_gs = curr_gs

        # Match over when health bars disappear (back to menu).
        terminated = not has_health_bar(frame, self._roi)
        truncated = False

        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        if self._mss is not None:
            self._mss.close()
            self._mss = None

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

    def _navigate_menu(self) -> np.ndarray:
        """Send menu-navigation actions until health bars are visible.

        Returns the last captured frame.
        """
        seq = self.cfg.char_select_seq
        post_stall = self.cfg.post_match_stall
        menu_detect = self.cfg.menu_detect_frames

        for _ in range(post_stall):
            self._send_input(Direction.NEUTRAL, [])
            self._capture()

        health_seen = 0
        last_frame = None

        for direction_name, button_names, hold in seq:
            d = getattr(Direction, direction_name.upper())
            b = [getattr(Button, n.upper()) for n in button_names]
            for _ in range(hold):
                self._send_input(d, b)
                last_frame = self._capture()
                if has_health_bar(last_frame, self._roi):
                    health_seen += 1
                else:
                    health_seen = 0

        for _ in range(menu_detect):
            self._send_input(Direction.NEUTRAL, [])
            last_frame = self._capture()
            if has_health_bar(last_frame, self._roi):
                health_seen += 1
                if health_seen >= 6:
                    break
            else:
                health_seen = 0

        return last_frame

    @staticmethod
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
