"""Output-delay buffer — models the motor side of human reaction time.

After the policy produces a prediction, the action must wait *motor_lag*
frames before it is physically executed (neural transmission + muscle
contraction + USB polling + game-engine input sampling).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, List

import torch

from model.action_space import Direction, Button


@dataclass
class DelayedAction:
    movement: int                     # Direction enum value
    buttons: torch.Tensor             # (6,) binary
    log_prob: torch.Tensor            # scalar
    value: torch.Tensor               # scalar
    payload: object = None            # opaque decision-time context


class OutputDelayBuffer:
    def __init__(self, motor_lag: int = 4):
        self.motor_lag = motor_lag
        self._queue: deque[DelayedAction] = deque(maxlen=motor_lag + 1)

    def push(self, movement: int, buttons: torch.Tensor,
             log_prob: torch.Tensor, value: torch.Tensor,
             payload: object = None) -> Optional[DelayedAction]:
        """Push a fresh prediction.  Returns the stale action ready to execute
        (or *None* while the pipeline is priming).

        *payload* rides along untouched — callers keep the decision context
        (e.g. the observation the action was based on) attached to the action
        until it executes.
        """
        self._queue.append(
            DelayedAction(movement=movement, buttons=buttons,
                          log_prob=log_prob, value=value, payload=payload))
        if len(self._queue) > self.motor_lag:
            return self._queue.popleft()
        return None

    def reset(self):
        self._queue.clear()

    def ready(self) -> bool:
        return len(self._queue) > self.motor_lag
