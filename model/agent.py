"""CleanRL-style agent wrapping SF3RNN with stride-N accumulation.

Rollout (``act``) accumulates N flow+audio frames, runs the model once,
and caches N sub-actions, returning one per env step.  The model's
MatchContext and prev_action are managed internally.

Training (``evaluate_stride``) is teacher-forced — used during PPO
trajectory replay where hidden/context are passed explicitly.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

from config import Config
from model.rnn import SF3RNN, HiddenState


class SF3Agent(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.temporal_stride
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

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

        self._flow_buf: list[torch.Tensor] = []
        self._audio_buf: list[torch.Tensor] = []
        self._action_buf: list[dict] = []
        self._prev_N: deque[dict] = deque(maxlen=self.N)

        # minGRU hidden — managed here because model.step() reads it as a param
        self.hidden: HiddenState = None

    # ------------------------------------------------------------------
    def reset(self):
        self._flow_buf.clear()
        self._audio_buf.clear()
        self._action_buf.clear()
        self._prev_N.clear()
        self.hidden = None
        self.model.reset_hidden()

    # ------------------------------------------------------------------
    # Rollout — per-frame accumulation, cached actions returned one at a time
    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self,
            flow: torch.Tensor,
            audio: Optional[torch.Tensor] = None,
            ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._flow_buf.append(flow.squeeze().cpu())
        if audio is not None:
            self._audio_buf.append(audio.squeeze().cpu())

        if len(self._flow_buf) >= self.N and not self._action_buf:
            self._forward()

        if self._action_buf:
            a = self._action_buf.pop(0)
            self._prev_N.append(a)
            return (int(a["mov_int"].item()), a["btn_mask"],
                    a["log_prob"], a["value"])

        return (0, torch.zeros(6, dtype=torch.float32),
                torch.tensor(0.0), torch.tensor(0.0))

    def _forward(self):
        N = self.N
        device = next(self.model.parameters()).device

        flows_in = torch.stack(self._flow_buf[-N:], dim=0)
        flows_in = flows_in.unsqueeze(0).unsqueeze(0).to(device)

        audio_in = None
        if self.model.use_audio and len(self._audio_buf) >= N:
            audio_in = torch.stack(self._audio_buf[-N:], dim=0)
            audio_in = audio_in.unsqueeze(0).to(device)

        if self._prev_N:
            mov_oh = F.one_hot(
                torch.tensor([a["mov_int"] for a in self._prev_N], device=device),
                num_classes=self.model.num_directions,
            ).float().unsqueeze(0)
            btn_stack = torch.stack(
                [a["btn_mask"].to(device) for a in list(self._prev_N)]
            ).unsqueeze(0)
            self.model._prev_action = torch.cat([mov_oh, btn_stack], dim=-1)

        actions, new_hidden = self.model.step(
            flows_in, self.hidden, audio_feats=audio_in,
        )
        self.hidden = new_hidden

        self._flow_buf = self._flow_buf[-1:]
        self._audio_buf = self._audio_buf[-1:]

        for i in range(N):
            mov_log = actions["movement"][i]
            btn_log = actions["buttons"][i]
            val = actions["value"][i]

            mov_dist = Categorical(logits=mov_log)
            mov_int = mov_dist.sample()
            btn_dist = Bernoulli(logits=btn_log)
            btn_mask = btn_dist.sample()
            logp = mov_dist.log_prob(mov_int) + btn_dist.log_prob(btn_mask).sum()

            self._action_buf.append({
                "mov_int":  mov_int.detach().cpu(),
                "btn_mask": btn_mask.detach().cpu(),
                "log_prob": logp.detach().cpu(),
                "value":    val.detach().cpu(),
            })

    # ------------------------------------------------------------------
    # Training — teacher-forced stride evaluation (no stride buffer)
    # ------------------------------------------------------------------
    def evaluate_stride(self,
                        flows: torch.Tensor,
                        ego_player: torch.Tensor,
                        hidden: HiddenState,
                        context: Optional[torch.Tensor],
                        prev_actions: Optional[torch.Tensor],
                        audio: Optional[torch.Tensor] = None,
                        ) -> dict:
        return self.model(flows, ego_player,
                          hidden=hidden, context=context,
                          prev_actions=prev_actions, audio=audio)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_value(self,
                  flows: torch.Tensor,
                  hidden: HiddenState,
                  context: Optional[torch.Tensor] = None,
                  audio: Optional[torch.Tensor] = None,
                  ego_player: float = 0.0,
                  ) -> torch.Tensor:
        ego = torch.full((1, 1), ego_player, device=flows.device, dtype=torch.float32)
        out = self.model(flows, ego,
                         hidden=hidden, context=context,
                         audio=audio)
        return out["value"][:, -1, -1].squeeze(-1)
