"""PPO trainer — CleanRL-style with trajectory-aware recurrent updates.

Collects rollouts through the environment with proper motor-lag buffering
and audio pipeline.  The PPO update replays every trajectory in sequence
so that the minGRU hidden state and MatchContext propagate correctly,
allowing gradients to flow through time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
from torch.utils.tensorboard import SummaryWriter

from config import Config
from model.agent import SF3Agent, HiddenState
from model.audio_buffer import AudioBuffer
from model.output_buffer import OutputDelayBuffer
from train.muon import make_muon_optimizer


# ---------------------------------------------------------------------------
@dataclass
class Transition:
    flow: torch.Tensor
    audio: Optional[torch.Tensor]
    movement: int
    buttons: torch.Tensor
    log_prob: float
    value: float
    reward: float
    done: bool


# ---------------------------------------------------------------------------
def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    T = len(rewards)
    device = rewards.device
    adv = torch.zeros(T, device=device)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * next_val * (1 - float(dones[t])) - values[t]
        gae = delta + gamma * gae_lambda * (1 - float(dones[t])) * gae
        adv[t] = gae
    returns = adv + values
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, returns


# ---------------------------------------------------------------------------
class PPOTrainer:
    def __init__(self, env, cfg: Config, bc_checkpoint: Optional[Path] = None):
        self.env = env
        self.cfg = cfg
        self.N = cfg.temporal_stride
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.agent = SF3Agent(cfg)
        if bc_checkpoint is not None:
            ckpt = torch.load(bc_checkpoint, map_location=self.device)
            self.agent.model.load_state_dict(ckpt["model"], strict=False)
            print(f"[PPO] Loaded BC weights from {bc_checkpoint}")

        self.optim = make_muon_optimizer(
            self.agent.model,
            lr=cfg.muon_lr * 0.1,
            adamw_lr=cfg.lr * 0.1,
            weight_decay=cfg.weight_decay,
            momentum=cfg.muon_momentum,
            ns_steps=cfg.muon_ns_steps,
        )

        self.writer = SummaryWriter(cfg.log_dir / "ppo")
        self.total_steps = 0
        self.episode = 0

        self.use_amp = cfg.use_amp and self.device.type == "cuda"
        self.amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16

        # Audio pipeline
        self._audio_buf: Optional[AudioBuffer] = None
        if cfg.use_audio:
            from env.audio_capture import AudioCapture
            self._audio_buf = AudioBuffer(
                audio_lag=cfg.audio_lag_frames,
                sample_rate=cfg.audio_sample_rate,
                feat_dim=cfg.audio_feat_dim,
                fps=cfg.fps,
                device=str(self.device),
            )
            self._audio_cap = AudioCapture(
                callback=self._audio_buf.add_samples,
                sample_rate=cfg.audio_sample_rate,
            )
            self._audio_cap.start()
        else:
            self._audio_cap = None

        if cfg.compile_model:
            self.agent.model = torch.compile(self.agent.model)

    # ------------------------------------------------------------------
    def close(self):
        if self._audio_cap is not None:
            self._audio_cap.stop()

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    @torch.no_grad()
    def collect_rollouts(self, num_steps: int) -> list[list[Transition]]:
        trajectories: list[list[Transition]] = []
        cur_traj: list[Transition] = []

        obs, _ = self.env.reset()
        self.agent.reset()
        motor_buf = OutputDelayBuffer(motor_lag=self.cfg.motor_lag_frames)

        stored = 0
        collected = 0

        while collected < num_steps:
            flow_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)

            audio_feats = None
            if self._audio_buf is not None:
                af = self._audio_buf.get_features()
                audio_feats = af.squeeze(0) if af is not None else None
            elif self.cfg.use_audio:
                audio_feats = torch.zeros(self.cfg.audio_feat_dim, device=self.device)

            mov_int, btn_mask, logp, val = self.agent.act(flow_t, audio=audio_feats)

            # Motor delay buffer
            delayed = motor_buf.push(int(mov_int), btn_mask, logp, val)
            if delayed is not None:
                action = np.concatenate(
                    [[delayed.movement], delayed.buttons.numpy().astype(np.int64)])
            else:
                action = np.zeros(7, dtype=np.int64)

            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            collected += 1

            # Only store when motor buffer is primed (action was actually executed)
            if delayed is not None:
                cur_traj.append(Transition(
                    flow=flow_t.squeeze(0).cpu(),
                    audio=audio_feats.cpu() if audio_feats is not None else None,
                    movement=delayed.movement,
                    buttons=delayed.buttons.cpu(),
                    log_prob=delayed.log_prob.item(),
                    value=delayed.value.item(),
                    reward=reward,
                    done=done,
                ))
                stored += 1

            if done:
                trajectories.append(cur_traj)
                cur_traj = []
                self.episode += 1
                obs, _ = self.env.reset()
                self.agent.reset()
                motor_buf = OutputDelayBuffer(motor_lag=self.cfg.motor_lag_frames)

        if cur_traj:
            trajectories.append(cur_traj)

        self.total_steps += stored
        return trajectories

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, trajectories: list[list[Transition]]) -> dict[str, float]:
        if not trajectories:
            return {"loss": 0.0, "policy_loss": 0.0,
                    "value_loss": 0.0, "entropy": 0.0}

        cfg = self.cfg
        N = self.N
        device = self.device

        aggregated = {"loss": 0.0, "policy_loss": 0.0,
                      "value_loss": 0.0, "entropy": 0.0}
        count = 0

        for epoch in range(cfg.ppo_updates):
            epoch_loss = 0.0
            epoch_policy = 0.0
            epoch_value = 0.0
            epoch_entropy = 0.0
            n_traj = 0

            random.shuffle(trajectories)

            for traj in trajectories:
                result = self._process_trajectory(traj)
                if result is None:
                    continue
                epoch_loss += result["loss"]
                epoch_policy += result["policy_loss"]
                epoch_value += result["value_loss"]
                epoch_entropy += result["entropy"]
                n_traj += 1

            if n_traj == 0:
                continue

            aggregated["loss"] += epoch_loss / n_traj
            aggregated["policy_loss"] += epoch_policy / n_traj
            aggregated["value_loss"] += epoch_value / n_traj
            aggregated["entropy"] += epoch_entropy / n_traj
            count += 1

        n = max(1, count)
        return {k: v / n for k, v in aggregated.items()}

    # ------------------------------------------------------------------
    def _process_trajectory(self, traj: list[Transition]) -> Optional[dict[str, float]]:
        T = len(traj)
        N = self.N
        if T < N:
            return None

        cfg = self.cfg
        device = self.device

        flows = torch.stack([t.flow for t in traj]).to(device)          # (T, 2, H, W)
        movements = torch.tensor([t.movement for t in traj], device=device)
        buttons = torch.stack([t.buttons for t in traj]).to(device)     # (T, 6)
        old_log_probs = torch.tensor([t.log_prob for t in traj], device=device)
        old_values = torch.tensor([t.value for t in traj], device=device)
        rewards = torch.tensor([t.reward for t in traj],
                               dtype=torch.float32, device=device)
        dones = torch.tensor([t.done for t in traj], dtype=torch.bool, device=device)

        if cfg.use_audio:
            feat_dim = self.agent.model.audio_encoder.net[0].in_features
            audio_ft = torch.stack([
                t.audio if t.audio is not None else torch.zeros(feat_dim)
                for t in traj
            ]).to(device)

        advantages, returns = compute_gae(
            rewards, dones, old_values, cfg.gamma, cfg.gae_lambda,
        )

        T_recur = T // N
        eff_T = T_recur * N

        flows_chunked = flows[:eff_T].reshape(1, T_recur, N, 2, *flows.shape[2:])
        movements_chunked = movements[:eff_T].reshape(1, T_recur, N)
        buttons_chunked = buttons[:eff_T].reshape(1, T_recur, N, 6)

        # Teacher-forced prev_actions (shifted by one stride)
        mov_oh = F.one_hot(movements_chunked, num_classes=cfg.num_directions).float()
        shifted_mov = torch.cat([
            torch.zeros(1, 1, N, cfg.num_directions, device=device),
            mov_oh[:, :-1],
        ], dim=1)
        shifted_btn = torch.cat([
            torch.zeros(1, 1, N, cfg.num_buttons, device=device),
            buttons_chunked[:, :-1],
        ], dim=1)
        prev_actions = torch.cat([shifted_mov, shifted_btn], dim=-1)

        if cfg.use_audio:
            audio_chunked = audio_ft[:eff_T].reshape(1, T_recur, N, -1)

        losses = []
        p_losses = []
        v_losses = []
        entropies = []

        hidden: HiddenState = None
        context: Optional[torch.Tensor] = None

        for t_step in range(T_recur):
            f_in = flows_chunked[:, t_step:t_step + 1].contiguous()
            ego = torch.zeros(1, 1, device=device)

            kw = {}
            if cfg.use_audio:
                kw["audio"] = audio_chunked[:, t_step:t_step + 1]

            with torch.amp.autocast(
                device_type=device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                out = self.agent.evaluate_stride(
                    f_in, ego, hidden=hidden, context=context,
                    prev_actions=prev_actions[:, t_step:t_step + 1],
                    **kw,
                )
            hidden = out["hidden"]
            context = out["context"]

            mov_log = out["movement"][0, 0]      # (N, 9)
            btn_log = out["buttons"][0, 0]       # (N, 6)
            vals = out["value"][0, 0]             # (N,)

            for i in range(N):
                idx = t_step * N + i
                if idx >= eff_T:
                    break

                mov_dist = Categorical(logits=mov_log[i])
                btn_dist = Bernoulli(logits=btn_log[i])

                new_logp = (
                    mov_dist.log_prob(movements_chunked[0, t_step, i])
                    + btn_dist.log_prob(buttons_chunked[0, t_step, i]).sum()
                )
                entropy = mov_dist.entropy() + btn_dist.entropy().sum()

                ratio = torch.exp(new_logp - old_log_probs[idx])
                adv = advantages[idx]

                # Policy loss (clipped)
                pg_loss = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1 - cfg.ppo_clip_eps, 1 + cfg.ppo_clip_eps) * adv,
                )

                # Value loss (clipped per SB3/CleanRL)
                v_unclipped = F.mse_loss(vals[i].view(-1), returns[idx].view(-1))
                v_clipped = old_values[idx] + torch.clamp(
                    vals[i] - old_values[idx],
                    -cfg.ppo_clip_eps, cfg.ppo_clip_eps,
                )
                v_clipped_loss = F.mse_loss(v_clipped.view(-1), returns[idx].view(-1))
                v_loss = torch.max(v_unclipped, v_clipped_loss)

                total = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
                losses.append(total)
                p_losses.append(pg_loss.detach())
                v_losses.append(v_loss.detach())
                entropies.append(entropy.detach())

        if not losses:
            return None

        loss = torch.stack(losses).mean()
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.max_grad_norm)
        self.optim.step()

        return {
            "loss": torch.stack(losses).mean().item(),
            "policy_loss": torch.stack(p_losses).mean().item() if p_losses else 0.0,
            "value_loss": torch.stack(v_losses).mean().item() if v_losses else 0.0,
            "entropy": torch.stack(entropies).mean().item() if entropies else 0.0,
        }

    # ------------------------------------------------------------------
    def save(self, tag: str = ""):
        p = self.cfg.ckpt_dir / f"ppo_step{self.total_steps:06d}{tag}.pt"
        model = self.agent.model._orig_mod if hasattr(
            self.agent.model, "_orig_mod") else self.agent.model
        torch.save({
            "step": self.total_steps,
            "episode": self.episode,
            "model": model.state_dict(),
            "optim_name": "muon",
            "optim": self.optim.state_dict(),
        }, p)

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device)
        self.agent.model.load_state_dict(ckpt["model"], strict=False)
        if ckpt.get("optim_name") == "muon":
            self.optim.load_state_dict(ckpt["optim"])
        self.total_steps = ckpt.get("step", 0)
        self.episode = ckpt.get("episode", 0)
