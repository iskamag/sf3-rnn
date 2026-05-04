"""PPO fine-tuning on top of behaviour-cloned policy.

The model runs once per *temporal_stride* (N) frames and produces N action
sets per recurrence step.  The rollout buffer stores per-frame data; during
updates we reshape to stride-N groups.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from config import Config
from model.rnn import SF3RNN
from train.muon import make_muon_optimizer
from train.reward import compute_returns


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
        self.N = cfg.temporal_stride
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.model = SF3RNN(
            feature_dim=cfg.feature_dim,
            use_audio=cfg.use_audio,
            audio_dim=cfg.audio_dim,
            context_dim=cfg.context_dim,
            action_embed_dim=cfg.action_embed_dim,
            temporal_stride=cfg.temporal_stride,
            mingru_hidden=cfg.mingru_hidden,
            mingru_layers=cfg.mingru_layers,
            mingru_dropout=cfg.mingru_dropout,
            num_directions=cfg.num_directions,
            num_buttons=cfg.num_buttons,
        ).to(self.device)

        if bc_checkpoint is not None:
            ckpt = torch.load(bc_checkpoint, map_location=self.device)
            self.model.load_state_dict(ckpt["model"], strict=False)
            print(f"[PPO] Loaded BC weights from {bc_checkpoint}")

        self.optim = make_muon_optimizer(
            self.model,
            lr=cfg.muon_lr * 0.1,
            adamw_lr=cfg.lr * 0.1,
            weight_decay=cfg.weight_decay,
            momentum=cfg.muon_momentum,
            ns_steps=cfg.muon_ns_steps,
        )

        self.buffer = RolloutBuffer(cfg)
        self.writer = SummaryWriter(cfg.log_dir / "ppo")
        self.step = 0

    # ------------------------------------------------------------------
    def _reshape_batch(self, batch: dict[str, torch.Tensor], start: int, end: int,
                       ) -> dict[str, torch.Tensor]:
        """Slice a frame-range and reshape to (1, T, N, ...) recurrence-step format."""
        N = self.N
        chunk_frames = end - start
        T_recur = chunk_frames // N
        assert chunk_frames % N == 0, f"chunk {chunk_frames} not divisible by {N}"

        out = {
            "flows": batch["flows"][start:end].reshape(1, T_recur, N, *batch["flows"].shape[1:]),
            "movement": batch["movement"][start:end].reshape(1, T_recur, N),
            "buttons": batch["buttons"][start:end].reshape(1, T_recur, N, -1),
        }
        if self.cfg.use_audio:
            out["audio"] = batch["audio"][start:end].reshape(1, T_recur, N, -1)
        return out

    def _forward_chunk(self, batch: dict[str, torch.Tensor], start: int, end: int,
                        hidden: Optional[torch.Tensor] = None,
                        context: Optional[torch.Tensor] = None,
                        prev_actions: Optional[torch.Tensor] = None,
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                    torch.Tensor, torch.Tensor]:
        N = self.N
        T_recur = (end - start) // N

        b = self._reshape_batch(batch, start, end)

        kw = {}
        if self.cfg.use_audio:
            kw["audio"] = b["audio"]
        if prev_actions is not None:
            kw["prev_actions"] = prev_actions[:, start // N : end // N]

        out = self.model(
            b["flows"],
            torch.zeros(1, 1, device=self.device),
            hidden=hidden,
            context=context,
            **kw,
        )

        chunk_frames = end - start
        return (out["movement"].reshape(1, chunk_frames, -1).squeeze(0),
                out["buttons"].reshape(1, chunk_frames, -1).squeeze(0),
                out["value"].reshape(1, chunk_frames).squeeze(0),
                out["hidden"],
                out["context"])

    @torch.no_grad()
    def _chunk_initial_states(self, batch: dict[str, torch.Tensor],
                              starts: list[int], seq_len: int,
                              prev_actions: Optional[torch.Tensor] = None,
                              ) -> dict[int, tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        hiddens: dict[int, Optional[torch.Tensor]] = {}
        contexts: dict[int, Optional[torch.Tensor]] = {}
        hidden = None
        context = None
        T = len(batch["rewards"])
        for start in starts:
            hiddens[start] = None if hidden is None else hidden.detach()
            contexts[start] = None if context is None else context.detach()
            _, _, _, hidden, context = self._forward_chunk(
                batch, start, min(start + seq_len, T), hidden, context,
                prev_actions=prev_actions)
        return {start: (hiddens[start], contexts[start]) for start in starts}

    def _sample_starts(self, starts: list[int], advantages: torch.Tensor,
                        seq_len: int, chunks_per_update: int) -> list[int]:
        if not self.cfg.ppo_prioritized_segments:
            order = torch.randperm(len(starts), device=self.device)[:chunks_per_update]
            return [starts[int(i)] for i in order]

        weights = torch.tensor([
            advantages[start:start + seq_len].abs().mean().item() + 1e-6
            for start in starts
        ], device=self.device)
        idx = torch.multinomial(
            weights / weights.sum(), chunks_per_update,
            replacement=chunks_per_update > len(starts),
        )
        return [starts[int(i)] for i in idx]

    def _make_prev_actions(self, batch: dict[str, torch.Tensor], T: int,
                           ) -> torch.Tensor:
        """Build (1, T_recur, N, 15) teacher-forced previous actions from batch."""
        N = self.N
        num_dir = self.cfg.num_directions
        num_btn = self.cfg.num_buttons
        device = self.device

        T_recur = T // N

        # movement: (T,) → (1, T_recur, N)
        movement = batch["movement"][:T_recur * N].reshape(1, T_recur, N)
        mov_oh = F.one_hot(movement, num_classes=num_dir).float()               # (1, T_recur, N, 9)

        # buttons: (T, 6) → (1, T_recur, N, 6)
        buttons = batch["buttons"][:T_recur * N].reshape(1, T_recur, N, num_btn)

        shifted_mov = torch.cat([
            torch.zeros(1, 1, N, num_dir, device=device), mov_oh[:, :-1],
        ], dim=1)                                                                 # (1, T_recur, N, 9)
        shifted_btn = torch.cat([
            torch.zeros(1, 1, N, num_btn, device=device), buttons[:, :-1],
        ], dim=1)                                                                 # (1, T_recur, N, 6)

        return torch.cat([shifted_mov, shifted_btn], dim=-1)                     # (1, T_recur, N, 15)

    def update(self) -> dict[str, float]:
        """Run PPO updates over recurrent sequence chunks from the buffer."""
        batch = self.buffer.get_batch(self.device)
        cfg = self.cfg
        N = self.N
        T = len(batch["rewards"])

        # Truncate to stride boundary
        T_recur = T // N
        T = T_recur * N
        chunk_frames = (cfg.ppo_seq_len // N) * N
        if chunk_frames < N:
            chunk_frames = N

        starts = list(range(0, T, chunk_frames))
        chunks_per_update = max(1, min(len(starts), cfg.ppo_minibatch // max(1, chunk_frames)))

        # ---- Advantages & returns from rollout-time values ----
        advantages, returns = compute_returns(
            batch["rewards"], batch["dones"],
            gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
            values=batch["values"].detach(),
        )
        advantages = advantages.detach()
        returns = returns.detach()

        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0

        prev_actions = self._make_prev_actions(batch, T)

        for _ in range(cfg.ppo_updates):
            chunk_states = self._chunk_initial_states(
                batch, starts, chunk_frames, prev_actions=prev_actions)
            selected = self._sample_starts(starts, advantages, chunk_frames, chunks_per_update)

            losses = []
            policy_losses = []
            value_losses = []
            entropies = []

            for start in selected:
                end = min(start + chunk_frames, T)
                hidden, context = chunk_states[start]
                new_mov, new_btn, new_val, _, _ = self._forward_chunk(
                    batch, start, end, hidden, context, prev_actions=prev_actions)

                mov_dist = torch.distributions.Categorical(logits=new_mov)
                mov_logp = mov_dist.log_prob(batch["movement"][start:end])

                btn_dist = torch.distributions.Bernoulli(logits=new_btn)
                btn_logp = btn_dist.log_prob(
                    batch["buttons"][start:end].clamp(0, 1)
                ).sum(-1)

                logp = mov_logp + btn_logp
                entropy = mov_dist.entropy().mean() + btn_dist.entropy().sum(-1).mean()

                ratio = torch.exp(logp - batch["old_log_probs"][start:end])
                clipped = torch.clamp(ratio, 1 - cfg.ppo_clip_eps, 1 + cfg.ppo_clip_eps)
                policy_loss = -torch.min(
                    ratio * advantages[start:end],
                    clipped * advantages[start:end],
                ).mean()

                value_loss = F.mse_loss(new_val, returns[start:end])
                losses.append(policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy)
                policy_losses.append(policy_loss.detach())
                value_losses.append(value_loss.detach())
                entropies.append(entropy.detach())

            loss = torch.stack(losses).mean()

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            self.optim.step()

            total_loss += loss.item()
            total_policy += torch.stack(policy_losses).mean().item()
            total_value += torch.stack(value_losses).mean().item()
            total_entropy += torch.stack(entropies).mean().item()

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
        model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        torch.save({
            "step": self.step,
            "model": model.state_dict(),
            "optim_name": "muon",
            "optim": self.optim.state_dict(),
        }, p)
