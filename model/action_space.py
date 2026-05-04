"""SF3 action encoding — 8-way joystick + 6 buttons."""

from enum import IntEnum
from typing import List, Tuple, Dict

import torch

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Direction(IntEnum):
    NEUTRAL = 0
    UP = 1
    UP_FORWARD = 2
    FORWARD = 3
    DOWN_FORWARD = 4
    DOWN = 5
    DOWN_BACK = 6
    BACK = 7
    UP_BACK = 8


class Button(IntEnum):
    LP = 0
    MP = 1
    HP = 2
    LK = 3
    MK = 4
    HK = 5


# Human-readable labels
DIR_NAMES = {
    Direction.NEUTRAL: "5",
    Direction.UP: "8", Direction.UP_FORWARD: "9",
    Direction.FORWARD: "6", Direction.DOWN_FORWARD: "3",
    Direction.DOWN: "2", Direction.DOWN_BACK: "1",
    Direction.BACK: "4", Direction.UP_BACK: "7",
}

BTN_NAMES = {
    Button.LP: "LP", Button.MP: "MP", Button.HP: "HP",
    Button.LK: "LK", Button.MK: "MK", Button.HK: "HK",
}

# ---------------------------------------------------------------------------
# Action ↔ Tensor conversion
# ---------------------------------------------------------------------------

def encode_action(direction: Direction,
                  active_buttons: List[Button]) -> Dict[str, torch.Tensor]:
    mov = torch.tensor(int(direction), dtype=torch.long)
    btn = torch.zeros(6, dtype=torch.float32)
    for b in active_buttons:
        btn[int(b)] = 1.0
    return {"movement": mov, "buttons": btn}


def decode_action(movement_logits: torch.Tensor,
                  button_logits: torch.Tensor,
                  threshold: float = 0.5) -> Tuple[Direction, List[Button]]:
    d = Direction(movement_logits.argmax(dim=-1).item())
    b = [Button(i) for i, logit in enumerate(button_logits)
         if torch.sigmoid(logit).item() > threshold]
    return d, b


def action_to_str(direction: Direction,
                  buttons: List[Button]) -> str:
    d = DIR_NAMES[direction]
    b = "+".join(BTN_NAMES[x] for x in buttons)
    return f"{d}{ '+' + b if b else ''}"

# ---------------------------------------------------------------------------
# Mirroring — for training from P2's perspective
# ---------------------------------------------------------------------------

_MIRROR_DIR = {
    Direction.NEUTRAL: Direction.NEUTRAL,
    Direction.UP: Direction.UP,
    Direction.UP_FORWARD: Direction.UP_BACK,
    Direction.FORWARD: Direction.BACK,
    Direction.DOWN_FORWARD: Direction.DOWN_BACK,
    Direction.DOWN: Direction.DOWN,
    Direction.DOWN_BACK: Direction.DOWN_FORWARD,
    Direction.BACK: Direction.FORWARD,
    Direction.UP_BACK: Direction.UP_FORWARD,
}


def mirror_direction(d: Direction) -> Direction:
    """Flip left/right in a Direction enum (e.g. FORWARD → BACK)."""
    return _MIRROR_DIR[d]


def mirror_bitfield(bits: int) -> int:
    """Flip Left (bit 2) ↔ Right (bit 3) in a CPS3 input bitfield."""
    left = (bits >> 2) & 1
    right = (bits >> 3) & 1
    bits &= ~(1 << 2 | 1 << 3)          # clear both
    bits |= (left << 3) | (right << 2)  # swap
    return bits

# ---------------------------------------------------------------------------
# FBNeo bitfield decoding (CPS3: sfiii3 / sfiii3n)
# ---------------------------------------------------------------------------
# CPS3 input byte layout used by FBNeo / Fightcade replays:
#   bit 0 – Up
#   bit 1 – Down
#   bit 2 – Left
#   bit 3 – Right
#   bit 4 – LP  (button 1 / jab)
#   bit 5 – MP  (button 2 / strong)
#   bit 6 – HP  (button 3 / fierce)
#   bit 7 – LK  (button 4 / short)
#   bit 8 – MK  (button 5 / forward)
#   bit 9 – HK  (button 6 / roundhouse)
#   bit 10 – Start
#   bits 11+ — Coin / Service (ignored)

def bitfield_to_action(bits: int) -> Tuple[Direction, List[Button]]:
    """Convert a CPS3 input bitfield to Direction + Buttons."""
    up    = (bits >> 0) & 1
    down  = (bits >> 1) & 1
    left  = (bits >> 2) & 1
    right = (bits >> 3) & 1
    lp    = (bits >> 4) & 1
    mp    = (bits >> 5) & 1
    hp    = (bits >> 6) & 1
    lk    = (bits >> 7) & 1
    mk    = (bits >> 8) & 1
    hk    = (bits >> 9) & 1

    if up and right:           d = Direction.UP_FORWARD
    elif up and left:          d = Direction.UP_BACK
    elif down and right:       d = Direction.DOWN_FORWARD
    elif down and left:        d = Direction.DOWN_BACK
    elif up:                   d = Direction.UP
    elif down:                 d = Direction.DOWN
    elif left:                 d = Direction.BACK
    elif right:                d = Direction.FORWARD
    else:                      d = Direction.NEUTRAL

    buttons = []
    if lp: buttons.append(Button.LP)
    if mp: buttons.append(Button.MP)
    if hp: buttons.append(Button.HP)
    if lk: buttons.append(Button.LK)
    if mk: buttons.append(Button.MK)
    if hk: buttons.append(Button.HK)

    return d, buttons


def action_to_bitfield(direction: Direction,
                       buttons: List[Button]) -> int:
    """Encode Direction + Buttons back to a CPS3 input bitfield."""
    bits = 0
    if direction in (Direction.UP, Direction.UP_FORWARD, Direction.UP_BACK):
        bits |= 1 << 0
    if direction in (Direction.DOWN, Direction.DOWN_FORWARD, Direction.DOWN_BACK):
        bits |= 1 << 1
    if direction in (Direction.BACK, Direction.DOWN_BACK, Direction.UP_BACK):
        bits |= 1 << 2
    if direction in (Direction.FORWARD, Direction.DOWN_FORWARD, Direction.UP_FORWARD):
        bits |= 1 << 3
    btn_bits = [4, 5, 6, 7, 8, 9]
    for b in buttons:
        bits |= 1 << btn_bits[int(b)]
    return bits
