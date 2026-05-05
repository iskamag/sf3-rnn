"""Reward function that can be computed from screen pixels alone.

Every reward component is derived from observable game-state (health bars,
round indicators).  No game-internal memory access needed — this is what
lets us use the same reward for BC replay scoring and RL fine-tuning.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from vision.game_state import GameState


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class RewardConfig:
    health_delta: float = 1.0       # weight on (damage done – damage taken)
    round_win: float = 10.0         # bonus for taking a round
    match_win: float = 50.0         # bonus for winning the match
    idle: float = -0.01             # penalty per step with no inputs


# ---------------------------------------------------------------------------
# Step-level reward
# ---------------------------------------------------------------------------
def compute_step_reward(prev: GameState,
                        curr: GameState,
                        p1: bool,
                        cfg: RewardConfig = RewardConfig(),
                        any_button: bool = True,
                        ) -> float:
    """Reward for transitioning from *prev* → *curr* state as player *p1*."""
    if p1:
        hp_self, hp_opp = curr.p1_health, curr.p2_health
        hp_self_prev, hp_opp_prev = prev.p1_health, prev.p2_health
        wins_self, wins_self_prev = curr.p1_wins, prev.p1_wins
    else:
        hp_self, hp_opp = curr.p2_health, curr.p1_health
        hp_self_prev, hp_opp_prev = prev.p2_health, prev.p1_health
        wins_self, wins_self_prev = curr.p2_wins, prev.p2_wins

    dmg_dealt = max(0.0, hp_opp_prev - hp_opp)
    dmg_taken = max(0.0, hp_self_prev - hp_self)
    rounds_won = max(0, wins_self - wins_self_prev)
    match_won = int(wins_self >= 2 and wins_self_prev < 2)

    reward = cfg.health_delta * (dmg_dealt - dmg_taken)
    reward += cfg.round_win * rounds_won
    reward += cfg.match_win * match_won
    reward += cfg.idle if not any_button else 0.0

    return reward


# ---------------------------------------------------------------------------
# Episode-level reward (for PPO return calculation)
# ---------------------------------------------------------------------------
def compute_returns(rewards: torch.Tensor,               # (T,)
                    dones: torch.Tensor,                 # (T,) bool
                    gamma: float = 0.99,
                    gae_lambda: float = 0.95,
                    values: torch.Tensor | None = None,  # (T,)
                    ) -> torch.Tensor:                    # (T,) advantages
    """Compute GAE advantages and returns.

    If *values* is None returns simple discounted returns.
    Returns tensor of advantages (same shape as *rewards*).
    """
    T = len(rewards)
    device = rewards.device

    if values is None:
        values = torch.zeros_like(rewards)

    advantages = torch.zeros(T, device=device)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * next_val * (1 - float(dones[t])) - values[t]
        gae = delta + gamma * gae_lambda * (1 - float(dones[t])) * gae
        advantages[t] = gae

    returns = advantages + values
    # Normalise advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns
