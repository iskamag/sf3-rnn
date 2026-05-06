"""PPO fine-tuning with sequence-aware temporal-stride updates.

The model runs once per *temporal_stride* (N) frames and produces N action
sets in one recurrence step.  This implementation preserves the full
reaction-time pipeline:
  - sensory lag (delayed observations)
  - motor lag (delayed action execution)
  - temporal stride (N actions per model forward)
  - hidden state / context propagation across episodes

Rollouts are collected at 60 fps; updates process full trajectories in
sequence so the minGRU hidden state and MatchContext are correctly restored.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical, Bernoulli

from config import Config
from model.rnn import SF3RNN, HiddenState
from train.muon import make_muon_optimizer


# ---------------------------------------------------------------------------
# Custom policy — wraps SF3RNN with stride-N accumulation
# ---------------------------------------------------------------------------

class _StrideBuffer:
    """Accumulates N flow frames then runs the model, producing N actions.

    Between model executions the queued actions are returned one per call
    to ``step()``.  The most recent N *returned* actions are tracked for
    teacher-forced prev_action embeddings on the next model forward.
    """

    def __init__(self, model: SF3RNN):
        self.model = model
        self.N = model.temporal_stride

        # Sensory accumulation (for the next model forward)
        self._flows: list[torch.Tensor] = []
        self._audios: list[torch.Tensor] = []

        # Output action queue from the most recent model forward
        self._actions: list[dict] = []

        # The last N actions actually returned by step() — used as prev_action
        self._prev_N: deque[dict] = deque(maxlen=self.N)

        # Persistent hidden / context across the episode
        self.hidden: HiddenState = None
        self.context: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def reset(self):
        self._flows.clear()
        self._audios.clear()
        self._actions.clear()
        self._prev_N.clear()
        self.hidden = None
        self.context = None
        self.model.reset_hidden()

    # ------------------------------------------------------------------
    def step(self, flow: torch.Tensor,
             audio: Optional[torch.Tensor] = None,
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Accumulate one sensor frame; returns (mov_int, btn_mask, logp, value).

        Every N-th call runs the model and queues N fresh actions.
        Returns neutral / zero log prob while the buffer is priming.
        """
        self._flows.append(flow.squeeze().cpu())
        if audio is not None:
            self._audios.append(audio.squeeze().cpu())

        if len(self._flows) >= self.N and len(self._actions) == 0:
            self._forward()

        if self._actions:
            a = self._actions.pop(0)
            self._prev_N.append(a)
            return (a["mov_int"], a["btn_mask"],
                    a["log_prob"], a["value"])

        # Priming: return neutral action
        return (torch.tensor(0), torch.zeros(6, dtype=torch.float32),
                torch.tensor(0.0), torch.tensor(0.0))

    # ------------------------------------------------------------------
    def _forward(self):
        """Run model on accumulated stride and queue N actions."""
        N = self.N
        device = next(self.model.parameters()).device

        # Model expects (B, T, N, 2, H, W) with T=1
        flows_in = torch.stack(self._flows[-N:], dim=0)
        flows_in = flows_in.unsqueeze(0).unsqueeze(0).to(device)

        audio_in = None
        if self.model.use_audio and len(self._audios) >= N:
            audio_in = torch.stack(self._audios[-N:], dim=0)  # (N, 15)
            audio_in = audio_in.unsqueeze(0).to(device)       # (1, N, 15) → step() expects (B, N, 15)

        actions, new_hidden = self.model.step(
            flows_in, self.hidden, audio_feats=audio_in,
        )
        self.hidden = new_hidden

        # Trim buffers (keep last frame for inter-stride continuity).
        self._flows = self._flows[-1:]
        self._audios = self._audios[-1:]

        # Build teacher-forced prev_action from the last N *returned* actions.
        if self._prev_N:
            mov_oh = F.one_hot(
                torch.tensor([a["mov_int"] for a in self._prev_N],
                             device=device),
                num_classes=self.model.num_directions,
            ).float().unsqueeze(0)
            btn_stack = torch.stack([a["btn_mask"].to(device)
                                     for a in list(self._prev_N)]).unsqueeze(0)
            self.model._prev_action = torch.cat([mov_oh, btn_stack], dim=-1)

        # Sample and queue N actions from model output.
        for i in range(N):
            mov_log = actions["movement"][i]
            btn_log = actions["buttons"][i]
            val = actions["value"][i]

            mov_dist = Categorical(logits=mov_log)
            mov_int = mov_dist.sample()
            btn_dist = Bernoulli(logits=torch.sigmoid(btn_log))
            btn_mask = btn_dist.sample()
            logp = mov_dist.log_prob(mov_int) + btn_dist.log_prob(btn_mask).sum()

            self._actions.append({
                "mov_int":  mov_int.detach().cpu(),
                "btn_mask": btn_mask.detach().cpu(),
                "log_prob": logp.detach().cpu(),
                "value":    val.detach().cpu(),
            })


# ---------------------------------------------------------------------------
# Rollout buffer — stores per-step transitions with trajectory tracking
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    flow: torch.Tensor
    audio: Optional[torch.Tensor]
    movement: int
    buttons: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: float
    done: bool


class RolloutBuffer:
    """Simple list-based buffer.  ``get_trajectories()`` splits by done flag."""

    def __init__(self):
        self._transitions: list[Transition] = []

    def add(self, t: Transition):
        self._transitions.append(t)

    def reset(self):
        self._transitions.clear()

    def __len__(self) -> int:
        return len(self._transitions)

    def get_trajectories(self) -> list[list[Transition]]:
        trajs: list[list[Transition]] = []
        cur: list[Transition] = []
        for t in self._transitions:
            cur.append(t)
            if t.done:
                trajs.append(cur)
                cur = []
        if cur:
            trajs.append(cur)
        return trajs


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    T = len(rewards)
    device = rewards.device
    advantages = torch.zeros(T, device=device)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * next_val * (1 - float(dones[t])) - values[t]
        gae = delta + gamma * gae_lambda * (1 - float(dones[t])) * gae
        advantages[t] = gae
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class SF3PPO:
    """PPO fine-tuning with sequence-aware, stride-N updates.

    Collects rollouts through a gymnasium environment, preserving hidden
    state across the full episode.  The update processes each trajectory
    in sequence so that the minGRU hidden state is correctly restored.
    """

    def __init__(self, cfg: Config, bc_checkpoint: Optional[Path] = None):
        self.cfg = cfg
        self.N = cfg.temporal_stride
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # ── model ──────────────────────────────────────────────────
        self.model = SF3RNN(
            feature_dim=cfg.feature_dim,
            use_audio=cfg.use_audio,
            audio_dim=cfg.audio_dim,
            context_dim=cfg.context_dim,
            context_layers=cfg.context_layers,
            action_embed_dim=cfg.action_embed_dim,
            temporal_stride=cfg.temporal_stride,
            mingru_hidden=cfg.mingru_hidden,
            mingru_layers=cfg.mingru_layers,
            mingru_dropout=cfg.mingru_dropout,
            num_directions=cfg.num_directions,
            num_buttons=cfg.num_buttons,
            cnn_channels=cfg.cnn_channels,
        ).to(self.device)

        if bc_checkpoint is not None:
            ckpt = torch.load(bc_checkpoint, map_location=self.device)
            self.model.load_state_dict(ckpt["model"], strict=False)
            print(f"[PPO] Loaded BC weights from {bc_checkpoint}")

        # ── optimizer ──────────────────────────────────────────────
        self.optim = make_muon_optimizer(
            self.model,
            lr=cfg.muon_lr * 0.1,
            adamw_lr=cfg.lr * 0.1,
            weight_decay=cfg.weight_decay,
            momentum=cfg.muon_momentum,
            ns_steps=cfg.muon_ns_steps,
        )

        # ── stride-N buffer (per-episode state) ────────────────────
        self._stride = _StrideBuffer(self.model)

        # ── rollout buffer ─────────────────────────────────────────
        self.buffer = RolloutBuffer()

        # ── logging ────────────────────────────────────────────────
        self.writer = SummaryWriter(cfg.log_dir / "ppo")
        self.total_steps = 0
        self.episode = 0

    # ------------------------------------------------------------------
    def reset(self):
        """Reset episode state: hidden, stride buffer, model."""
        self._stride.reset()
        self.model.reset_hidden()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, flow: torch.Tensor,
            audio: Optional[torch.Tensor] = None,
            ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step forward through stride-N accumulation.

        Returns (movement, buttons, log_prob, value) with the action
        and log_prob sampled consistently inside the stride buffer.
        """
        mov_int, btn_mask, logp, val = self._stride.step(flow, audio)
        return int(mov_int.item()), btn_mask, logp, val

    # ------------------------------------------------------------------
    def _process_trajectory(self, traj: list[Transition],
                            ) -> dict[str, float]:
        """Run PPO updates on one trajectory, replaying in sequence.

        The model hidden state is reset at the start and propagated
        forward through each stride window.
        """
        T = len(traj)
        if T < self.N:
            return {"loss": 0.0, "policy_loss": 0.0,
                    "value_loss": 0.0, "entropy": 0.0}

        cfg = self.cfg
        device = self.device

        # ── trajectory tensors ──────────────────────────────────
        flows = torch.stack([t.flow for t in traj]).to(device)
        movements = torch.tensor([t.movement for t in traj], device=device)
        buttons = torch.stack([t.buttons for t in traj]).to(device)
        old_log_probs = torch.stack([t.log_prob for t in traj]).to(device)
        old_values = torch.stack([t.value for t in traj]).to(device)
        rewards = torch.tensor([t.reward for t in traj],
                               dtype=torch.float32, device=device)

        if cfg.use_audio:
            feat_dim = self.model.audio_encoder.net[0].in_features
            audio_ft = torch.stack([
                t.audio if t.audio is not None else torch.zeros(feat_dim)
                for t in traj
            ]).to(device)                                                           # (T, 15)
        dones = torch.tensor([t.done for t in traj],
                             dtype=torch.bool, device=device)

        advantages, returns = compute_gae(
            rewards, dones, old_values,
            gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
        )

        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(cfg.ppo_updates):
            self.model.reset_hidden()
            hidden: HiddenState = None
            context: Optional[torch.Tensor] = None

            # Align to stride boundaries.
            T_recur = T // self.N
            eff_T = T_recur * self.N

            flows_chunked = flows[:eff_T].reshape(
                1, T_recur, self.N, 2, *flows.shape[2:])
            movements_chunked = movements[:eff_T].reshape(
                1, T_recur, self.N)
            buttons_chunked = buttons[:eff_T].reshape(
                1, T_recur, self.N, 6)

            # Teacher-forced prev_actions (shifted by one stride).
            mov_oh = F.one_hot(
                movements_chunked, num_classes=cfg.num_directions).float()
            shifted_mov = torch.cat([
                torch.zeros(1, 1, self.N, cfg.num_directions, device=device),
                mov_oh[:, :-1],
            ], dim=1)
            shifted_btn = torch.cat([
                torch.zeros(1, 1, self.N, cfg.num_buttons, device=device),
                buttons_chunked[:, :-1],
            ], dim=1)
            prev_actions = torch.cat([shifted_mov, shifted_btn], dim=-1)

            losses = []
            policy_losses = []
            value_losses = []
            entropies = []

            if cfg.use_audio:
                audio_chunked = audio_ft[:eff_T].reshape(
                    1, T_recur, self.N, -1)

            for t_step in range(T_recur):
                f_in = flows_chunked[:, t_step:t_step + 1].contiguous()
                ego = torch.zeros(1, 1, device=device)

                kw = {}
                if cfg.use_audio:
                    kw["audio"] = audio_chunked[:, t_step:t_step + 1]

                out = self.model(
                    f_in, ego,
                    hidden=hidden,
                    context=context,
                    prev_actions=prev_actions[:, t_step:t_step + 1],
                    **kw,
                )
                hidden = out["hidden"]
                context = out["context"]

                # out["movement"] is (1, 1, N, 9); squeeze B & T → (N, 9)
                mov_log = out["movement"][0, 0]
                btn_log = out["buttons"][0, 0]
                vals = out["value"][0, 0]

                for i in range(self.N):
                    idx = t_step * self.N + i
                    if idx >= eff_T:
                        break

                    mov_dist = Categorical(logits=mov_log[i])
                    btn_dist = Bernoulli(logits=torch.sigmoid(btn_log[i]))

                    new_logp = (
                        mov_dist.log_prob(movements_chunked[0, t_step, i])
                        + btn_dist.log_prob(
                            buttons_chunked[0, t_step, i]).sum()
                    )
                    entropy = (
                        mov_dist.entropy()
                        + btn_dist.entropy().sum()
                    )

                    ratio = torch.exp(new_logp - old_log_probs[idx])
                    adv = advantages[idx]
                    clipped = torch.clamp(
                        ratio, 1 - cfg.ppo_clip_eps, 1 + cfg.ppo_clip_eps)
                    p_loss = -min(ratio * adv, clipped * adv)

                    v_loss = F.mse_loss(vals[i], returns[idx])

                    losses.append(
                        p_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy)
                    policy_losses.append(p_loss.detach())
                    value_losses.append(v_loss.detach())
                    entropies.append(entropy.detach())

            if not losses:
                continue

            loss = torch.stack(losses).mean()

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.max_grad_norm)
            self.optim.step()

            total_loss += loss.item()
            total_policy += torch.stack(policy_losses).mean().item()
            total_value += torch.stack(value_losses).mean().item()
            total_entropy += torch.stack(entropies).mean().item()
            n_updates += 1

        if n_updates == 0:
            return {"loss": 0.0, "policy_loss": 0.0,
                    "value_loss": 0.0, "entropy": 0.0}

        return {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy / n_updates,
            "value_loss": total_value / n_updates,
            "entropy": total_entropy / n_updates,
        }

    # ------------------------------------------------------------------
    def update(self) -> dict[str, float]:
        """Run PPO updates over all trajectories in the buffer."""
        trajectories = self.buffer.get_trajectories()
        if not trajectories:
            return {"loss": 0.0, "policy_loss": 0.0,
                    "value_loss": 0.0, "entropy": 0.0}

        aggregated = {"loss": 0.0, "policy_loss": 0.0,
                      "value_loss": 0.0, "entropy": 0.0}
        count = 0

        for traj in trajectories:
            metrics = self._process_trajectory(traj)
            for k in aggregated:
                aggregated[k] += metrics[k]
            count += 1

        self.buffer.reset()
        n = max(1, count)
        return {k: v / n for k, v in aggregated.items()}

    # ------------------------------------------------------------------
    def save(self, tag: str = ""):
        p = self.cfg.ckpt_dir / f"ppo_step{self.total_steps:06d}{tag}.pt"
        model = self.model._orig_mod if hasattr(
            self.model, "_orig_mod") else self.model
        torch.save({
            "step": self.total_steps,
            "episode": self.episode,
            "model": model.state_dict(),
            "optim_name": "muon",
            "optim": self.optim.state_dict(),
        }, p)

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"], strict=False)
        if ckpt.get("optim_name") == "muon":
            self.optim.load_state_dict(ckpt["optim"])
        self.total_steps = ckpt.get("step", 0)
        self.episode = ckpt.get("episode", 0)
