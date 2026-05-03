"""PPO fine-tuning on top of behaviour-cloned policy.

Self-play or environment returns frames.  The agent processes one delayed
optical-flow frame per LSTM step.  PPO updates run over the full episode
sequence so the LSTM's hidden state is respected during the backward pass.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from config import Config
from model.rnn import SF3RNN
from train.reward import compute_returns, RewardConfig
from vision.game_state import GameState


# ---------------------------------------------------------------------------
# Rollout buffer — stores one flow frame + action per game step
# ---------------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.use_audio = cfg.use_audio
        self.reset()

    def reset(self):
        self.flows: list[torch.Tensor] = []           # each (2, H, W) on CPU
        self.audios: list[torch.Tensor] = []           # each (15,) on CPU  (if use_audio)
        self.movements: list[int] = []
        self.button_masks: list[torch.Tensor] = []    # each (6,) float
        self.log_probs: list[torch.Tensor] = []       # scalar
        self.values: list[torch.Tensor] = []          # scalar
        self.rewards: list[float] = []
        self.dones: list[bool] = []

    def add(self, flow: torch.Tensor, movement: int, buttons: torch.Tensor,
            log_prob: torch.Tensor, value: torch.Tensor,
            reward: float, done: bool,
            audio_feats: torch.Tensor | None = None):
        self.flows.append(flow.squeeze(0).squeeze(0).cpu())
        if self.use_audio:
            assert audio_feats is not None, "use_audio=True but no audio_feats in buffer.add()"
            self.audios.append(audio_feats.cpu())
        self.movements.append(movement)
        self.button_masks.append(buttons.cpu())
        self.log_probs.append(log_prob.detach().cpu())
        self.values.append(value.detach().cpu())
        self.rewards.append(reward)
        self.dones.append(done)

    def get_batch(self, device: torch.device) -> dict[str, torch.Tensor]:
        batch = {
            "flows":         torch.stack(self.flows).to(device),
            "movement":      torch.tensor(self.movements, dtype=torch.long, device=device),
            "buttons":       torch.stack(self.button_masks).to(device),
            "old_log_probs": torch.stack(self.log_probs).to(device),
            "values":        torch.stack(self.values).to(device),
            "rewards":       torch.tensor(self.rewards, dtype=torch.float32, device=device),
            "dones":         torch.tensor(self.dones, dtype=torch.bool, device=device),
        }
        if self.use_audio:
            batch["audio"] = torch.stack(self.audios).to(device)
        return batch

    def __len__(self) -> int:
        return len(self.rewards)


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------
class PPOTrainer:
    def __init__(self, cfg: Config, bc_checkpoint: Optional[Path] = None):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.model = SF3RNN(
            feature_dim=cfg.feature_dim,
            use_audio=cfg.use_audio,
            audio_dim=cfg.audio_dim,
            context_dim=cfg.context_dim,
            lstm_hidden=cfg.lstm_hidden,
            lstm_layers=cfg.lstm_layers,
            lstm_dropout=cfg.lstm_dropout,
            num_directions=cfg.num_directions,
            num_buttons=cfg.num_buttons,
        ).to(self.device)

        if bc_checkpoint is not None:
            ckpt = torch.load(bc_checkpoint, map_location=self.device)
            self.model.load_state_dict(ckpt["model"], strict=False)
            print(f"[PPO] Loaded BC weights from {bc_checkpoint}")

        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr * 0.1)

        self.buffer = RolloutBuffer(cfg)
        self.writer = SummaryWriter(cfg.log_dir / "ppo")
        self.step = 0

    # ------------------------------------------------------------------
    def update(self) -> dict[str, float]:
        """One PPO epoch over the collected buffer."""
        batch = self.buffer.get_batch(self.device)
        cfg = self.cfg

        # ---- Run model over the full episode ----
        kw = {}
        if cfg.use_audio:
            kw["audio"] = batch["audio"].unsqueeze(0)
        out = self.model(batch["flows"].unsqueeze(0), context=None, **kw)
        new_mov = out["movement"].squeeze(0)          # (T, 9)
        new_btn = out["buttons"].squeeze(0)           # (T, 6)
        new_val = out["value"].squeeze(0)             # (T,)

        # ---- Advantages & returns (GAE) ----
        advantages, returns = compute_returns(
            batch["rewards"], batch["dones"],
            gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
            values=torch.cat([batch["values"], new_val[-1:]]),
        )

        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0

        for _ in range(cfg.ppo_updates):
            # ---- Policy loss (movement + buttons) ----
            mov_dist = torch.distributions.Categorical(logits=new_mov)
            mov_logp = mov_dist.log_prob(batch["movement"])

            # Buttons: independent Bernoulli log-prob
            btn_probs = torch.sigmoid(new_btn)
            btn_logp = torch.distributions.Bernoulli(btn_probs).log_prob(
                batch["buttons"].clamp(0, 1)
            ).sum(-1)

            logp = mov_logp + btn_logp
            entropy = mov_dist.entropy().mean()

            ratio = torch.exp(logp - batch["old_log_probs"])
            clipped = torch.clamp(ratio, 1 - cfg.ppo_clip_eps, 1 + cfg.ppo_clip_eps)
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

            value_loss = F.mse_loss(new_val, returns)

            loss = (policy_loss +
                    cfg.vf_coef * value_loss -
                    cfg.ent_coef * entropy)

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            self.optim.step()

            total_loss += loss.item()
            total_policy += policy_loss.item()
            total_value += value_loss.item()
            total_entropy += entropy.item()

        self.buffer.reset()
        n = cfg.ppo_updates
        return {
            "loss": total_loss / n,
            "policy_loss": total_policy / n,
            "value_loss": total_value / n,
            "entropy": total_entropy / n,
        }

    # ------------------------------------------------------------------
    def save(self, tag: str = ""):
        p = self.cfg.ckpt_dir / f"ppo_step{self.step:06d}{tag}.pt"
        torch.save({
            "step": self.step,
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
        }, p)
