"""PPO fine-tuning on top of behaviour-cloned policy.

Self-play or environment returns frames.  The agent processes one delayed
optical-flow frame per minGRU step.  PPO updates run over the full episode
sequence so the minGRU hidden state is respected during the backward pass.
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
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.model = SF3RNN(
            feature_dim=cfg.feature_dim,
            use_audio=cfg.use_audio,
            audio_dim=cfg.audio_dim,
            context_dim=cfg.context_dim,
            action_embed_dim=cfg.action_embed_dim,
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
    def _forward_chunk(self, batch: dict[str, torch.Tensor], start: int, end: int,
                        hidden: Optional[torch.Tensor] = None,
                        context: Optional[torch.Tensor] = None,
                        prev_actions: Optional[torch.Tensor] = None,
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                    torch.Tensor, torch.Tensor]:
        kw = {}
        if self.cfg.use_audio:
            kw["audio"] = batch["audio"][start:end].unsqueeze(0)
        if prev_actions is not None:
            kw["prev_actions"] = prev_actions[:, start:end]
        out = self.model(
            batch["flows"][start:end].unsqueeze(0),
            torch.zeros(1, 1, device=self.device),
            hidden=hidden,
            context=context,
            **kw,
        )
        return (out["movement"].squeeze(0),
                out["buttons"].squeeze(0),
                out["value"].squeeze(0),
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

    def update(self) -> dict[str, float]:
        """Run PPO updates over recurrent sequence chunks from the buffer."""
        batch = self.buffer.get_batch(self.device)
        cfg = self.cfg
        T = len(batch["rewards"])
        seq_len = min(cfg.ppo_seq_len, T)
        starts = list(range(0, T, seq_len))
        chunks_per_update = max(1, min(len(starts), cfg.ppo_minibatch // seq_len))

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

        # ---- Previous-action inputs (teacher forcing with rollout actions) ----
        T_len = len(batch["rewards"])
        mov_oh = F.one_hot(batch["movement"], num_classes=cfg.num_directions).float()
        shifted_mov = torch.cat([
            torch.zeros(1, cfg.num_directions, device=self.device), mov_oh[:-1],
        ])
        shifted_btn = torch.cat([
            torch.zeros(1, cfg.num_buttons, device=self.device), batch["buttons"][:-1],
        ])
        prev_actions = torch.cat([shifted_mov, shifted_btn], dim=-1).unsqueeze(0)

        for _ in range(cfg.ppo_updates):
            chunk_states = self._chunk_initial_states(
                batch, starts, seq_len, prev_actions=prev_actions)
            selected = self._sample_starts(starts, advantages, seq_len, chunks_per_update)

            losses = []
            policy_losses = []
            value_losses = []
            entropies = []

            for start in selected:
                end = min(start + seq_len, T)
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
        torch.save({
            "step": self.step,
            "model": self.model.state_dict(),
            "optim_name": "muon",
            "optim": self.optim.state_dict(),
        }, p)
