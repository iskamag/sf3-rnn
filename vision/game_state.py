"""Extract health, stun, super-meter and timer from SF3 HUD regions."""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass(slots=True)
class GameState:
    p1_health: float = 1.0
    p2_health: float = 1.0
    p1_stun: float = 0.0
    p2_stun: float = 0.0
    p1_super: float = 0.0
    p2_super: float = 0.0
    round_timer: int = 99
    p1_wins: int = 0
    p2_wins: int = 0

    def to_tensor(self) -> "torch.Tensor":  # deferred import
        import torch
        return torch.tensor([
            self.p1_health, self.p2_health,
            self.p1_stun, self.p2_stun,
            self.p1_super, self.p2_super,
            self.round_timer / 99.0,
            self.p1_wins / 2.0, self.p2_wins / 2.0,
        ], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Health bar reading
# ---------------------------------------------------------------------------
def _colour_mask_ratio(bar: np.ndarray,                      # (h, w, 3) BGR
                       lower: tuple,
                       upper: tuple) -> float:
    """Fraction of pixels inside *bar* that fall within the [lower, upper]
    HSV colour range."""
    hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return mask.sum() / bar.size * 3  # bar.size = h * w * 3


# Health-bar colours in SF3 are warm (green → yellow → red).
# We catch all of them with a wide saturation/value window and a hue envelope.
_HEALTH_LOWER = np.array([10, 80, 80])    # H, S, V
_HEALTH_UPPER = np.array([25, 255, 255])

# Stun bar is white → grey gradient
_STUN_LOWER = np.array([0, 0, 180])
_STUN_UPPER = np.array([180, 30, 255])

# Super bar is blue
_SUPER_LOWER = np.array([100, 80, 80])
_SUPER_UPPER = np.array([130, 255, 255])


def read_health(bar: np.ndarray) -> float:
    """Return health fraction (0 – 1) from a cropped health-bar image."""
    return _colour_mask_ratio(bar, _HEALTH_LOWER, _HEALTH_UPPER)


def read_stun(bar: np.ndarray) -> float:
    return _colour_mask_ratio(bar, _STUN_LOWER, _STUN_UPPER)


def read_super(bar: np.ndarray) -> float:
    return _colour_mask_ratio(bar, _SUPER_LOWER, _SUPER_UPPER)


# ---------------------------------------------------------------------------
# High-level extraction
# ---------------------------------------------------------------------------

def has_health_bar(frame: np.ndarray, roi: dict,
                   threshold: float = 0.08) -> bool:
    """Quick check — are health bars visible on this frame?  True during
    gameplay, False during character select, intros, and victory screens."""
    def crop(r):
        x, y, w, h = r
        return frame[y:y + h, x:x + w]
    p1 = read_health(crop(roi["p1_health"]))
    p2 = read_health(crop(roi["p2_health"]))
    return (p1 > threshold) and (p2 > threshold)


def find_match_boundaries(
    health_signal: "np.ndarray",           # (T,) bool — True = health visible
    min_round_gap: int = 30,               # frames of health-loss that are just a round reset
    min_menu_gap: int = 180,               # frames of health-loss that signal a menu
    min_match_len: int = 120,              # ignore very short segments
) -> "list[tuple[int, int]]":
    """Split a boolean health signal into (start, end) match windows.

    Short health-absent gaps (< min_round_gap) are absorbed into the
    current match.  Long gaps (>= min_menu_gap) signal character select
    or a return to the main menu, splitting the replay into separate
    matches.
    """
    T = len(health_signal)
    if T == 0:
        return []

    # ---- greedy forward scan ----
    matches: list[tuple[int, int]] = []
    i = 0
    while i < T:
        # fast-forward to health
        while i < T and not health_signal[i]:
            i += 1
        if i >= T:
            break
        start = i

        # consume until we hit a long gap
        gap_counter = 0
        while i < T:
            if health_signal[i]:
                gap_counter = 0
                i += 1
            else:
                gap_counter += 1
                i += 1
                if gap_counter >= min_menu_gap:
                    break

        end = i - gap_counter
        if end - start >= min_match_len:
            matches.append((start, end))

        # skip the remaining gap
        while i < T and not health_signal[i]:
            i += 1

    return matches


def health_signal_from_array(
    health: "np.ndarray",                  # (T,) float, health_p1 or health_p2
    threshold: float = 0.08,
) -> "np.ndarray":                         # (T,) bool
    """Convert a health trace into a boolean gameplay signal."""
    return health > threshold


def health_signal_from_inputs(
    buttons: "np.ndarray",                 # (T, 6) float
    activity_window: int = 60,
    threshold: float = 0.01,
) -> "np.ndarray":                         # (T,) bool
    """Approximate gameplay frames from button activity.

    Fallback when screen captures are unavailable.  Menus have sparse
    directional inputs; gameplay has frequent button presses.
    """
    T = len(buttons)
    # rolling sum of button presses within window
    kernel = np.ones(activity_window)
    active = buttons.sum(axis=1)                     # (T,)  total buttons pressed per frame
    convolved = np.convolve(active, kernel, mode="same")
    return convolved > threshold


def extract_game_state(frame: np.ndarray,                     # (H, W, 3) BGR
                       roi: dict,
                       ) -> GameState:
    """Extract full GameState from a native-resolution frame."""
    def crop(r):
        x, y, w, h = r
        return frame[y:y + h, x:x + w]

    return GameState(
        p1_health=read_health(crop(roi["p1_health"])),
        p2_health=read_health(crop(roi["p2_health"])),
        p1_stun=read_stun(crop(roi["p1_stun"])),
        p2_stun=read_stun(crop(roi["p2_stun"])),
    )
