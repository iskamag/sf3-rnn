"""CNN + (optional) Audio + MatchContext -> minGRU policy model for SF3.

Visual:    optical flow (2-ch)  → FlowCNN      → feature_dim
Audio:     MFCC + energy        → AudioMLP      → audio_dim    (if use_audio)
Context:   minGRU output        → MatchContext  → context_dim  (persistent opponent profile)
Fused:     concat(vis, aud, ctx)→ minGRU         → movement / buttons / value

The match-context vector is a learned opponent profile that persists
across BC training segments (detached at boundaries) and across the
full episode during PPO.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from minGRU_pytorch import minGRU

from model.context import MatchContext


HiddenState = Optional[torch.Tensor]


# ---------------------------------------------------------------------------
# Visual encoder — 2-channel optical flow → feature vector
# ---------------------------------------------------------------------------
class _FlowEncoder(nn.Module):
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 7, stride=2, padding=3), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, feature_dim),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:       # (..., 2, H, W) → (..., F)
        lead = x.shape[:-3]
        flat = x.reshape(-1, 2, x.shape[-2], x.shape[-1])
        h = self.conv(flat)
        h = self.pool(h).flatten(1)
        feats = self.fc(h)
        return feats.reshape(*lead, -1)


# ---------------------------------------------------------------------------
# Audio encoder — 15-d acoustic features → compact embedding
# ---------------------------------------------------------------------------
class _AudioEncoder(nn.Module):
    def __init__(self, in_dim: int = 15, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:       # (..., 15) → (..., A)
        lead = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        feats = self.net(flat)
        return feats.reshape(*lead, -1)


class _MinGRUStack(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.layers = nn.ModuleList([minGRU(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout) if num_layers > 1 and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, hidden: HiddenState = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        next_hidden = []
        for layer_idx, layer in enumerate(self.layers):
            prev = None if hidden is None else hidden[layer_idx]
            x, prev = layer(x, prev, return_next_prev_hidden=True)
            next_hidden.append(prev)
            if layer_idx < self.num_layers - 1:
                x = self.dropout(x)
        return x, torch.stack(next_hidden, dim=0)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class SF3RNN(nn.Module):
    def __init__(self,
                 feature_dim: int = 256,
                 use_audio: bool = True,
                 audio_dim: int = 64,
                 audio_feat_dim: int = 15,
                 context_dim: int = 64,
                 action_embed_dim: int = 64,
                 mingru_hidden: int = 512,
                 mingru_layers: int = 2,
                 mingru_dropout: float = 0.1,
                 num_directions: int = 9,
                 num_buttons: int = 6,
                 ):
        super().__init__()
        self.mingru_hidden = mingru_hidden
        self.mingru_layers = mingru_layers
        self.use_audio = use_audio
        self.context_dim = context_dim
        self.num_directions = num_directions
        self.num_buttons = num_buttons
        self.action_dim = num_directions + num_buttons  # one-hot movement + binary buttons

        self.encoder = _FlowEncoder(feature_dim)

        self._action_embed = nn.Sequential(
            nn.Linear(self.action_dim, action_embed_dim),
            nn.ReLU(),
            nn.Linear(action_embed_dim, action_embed_dim),
        )

        # +1 for the ego_player indicator bit (0=P1, 1=P2)
        recurrent_input = feature_dim + context_dim + action_embed_dim + 1
        if use_audio:
            self.audio_encoder = _AudioEncoder(audio_feat_dim, audio_dim)
            recurrent_input += audio_dim
        else:
            self.audio_encoder = None

        self.match_context = MatchContext(mingru_hidden, context_dim)

        self.recurrent = _MinGRUStack(recurrent_input, mingru_hidden,
                                      mingru_layers, mingru_dropout)

        head = [mingru_hidden, mingru_hidden // 2]
        self.movement_head = nn.Sequential(
            nn.Linear(*head), nn.ReLU(), nn.Linear(head[1], num_directions),
        )
        self.button_head = nn.Sequential(
            nn.Linear(*head), nn.ReLU(), nn.Linear(head[1], num_buttons),
        )
        self.value_head = nn.Sequential(
            nn.Linear(*head), nn.ReLU(), nn.Linear(head[1], 1),
        )

        self.hidden: HiddenState = None
        self._context: Optional[torch.Tensor] = None          # (B, context_dim)

    # ------------------------------------------------------------------
    def reset_hidden(self):
        self.hidden = None
        self._context = None
        self._prev_action = None

    def detach_context(self):
        """Detach the persistent context so gradients don't flow across
        BC segment boundaries (the value itself survives)."""
        if self._context is not None:
            self._context = self._context.detach()

    # ------------------------------------------------------------------
    def forward(self,
                flows: torch.Tensor,                          # (B, T, 2, H, W)
                ego_player: torch.Tensor,                      # (B, 1) float 0=P1, 1=P2
                hidden: HiddenState = None,
                audio: Optional[torch.Tensor] = None,          # (B, T, 15) or None
                context: Optional[torch.Tensor] = None,         # (B, C) initial context
                prev_actions: Optional[torch.Tensor] = None,    # (B, T, D) or None  D=num_dir+num_btn
                ) -> Dict[str, torch.Tensor]:
        vis = self.encoder(flows)                             # (B, T, F)

        if self.use_audio:
            if audio is None:
                raise ValueError(
                    "use_audio=True but no audio tensor passed to forward()")
            aud = self.audio_encoder(audio)                   # (B, T, A)

        B, T = vis.shape[:2]
        device = vis.device

        if context is None:
            ctx = self.match_context.initial_context(B, device)
        else:
            ctx = context

        ego = ego_player.unsqueeze(1).expand(-1, T, -1)
        ctx_seq = ctx.unsqueeze(1).expand(-1, T, -1)

        if prev_actions is not None:
            act_emb = self._action_embed(prev_actions)            # (B, T, E)
        else:
            act_emb = torch.zeros(B, T, self._action_embed[-1].out_features,
                                  device=device)

        parts = [vis, ctx_seq, ego, act_emb]
        if self.use_audio:
            parts.insert(1, aud)
        recurrent_in = torch.cat(parts, dim=-1)
        recurrent_out, hidden = self.recurrent(recurrent_in, hidden)

        # Update once per chunk so training stays sequence-parallel while
        # preserving a persistent opponent profile across chunks/matches.
        ctx = self.match_context(recurrent_out[:, -1], ctx)

        return {
            "movement": self.movement_head(recurrent_out),
            "buttons":  self.button_head(recurrent_out),
            "value":    self.value_head(recurrent_out).squeeze(-1),
            "hidden":   hidden,
            "context":  ctx,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self,
             flow: torch.Tensor,                              # (1, 1, 2, H, W)
             hidden: HiddenState = None,
             audio_feats: Optional[torch.Tensor] = None,       # (1, 15)
             ego_player: float = 0.0,                           # 0=P1, 1=P2
             ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Single-timestep inference.  *flow* and *audio_feats* are the
        (separately delayed) sensory inputs.  The match context persists
        across calls via self._context."""
        if self.use_audio and audio_feats is None:
            audio_dim = self.audio_encoder.net[0].in_features
            audio_feats = torch.zeros((1, audio_dim), device=flow.device)
        audio = audio_feats.unsqueeze(1) if audio_feats is not None else None
        ego = torch.full((flow.shape[0], 1), ego_player,
                         device=flow.device, dtype=torch.float32)

        prev = self._prev_action                                          # (B, 15) or None
        if prev is not None:
            if prev.device != flow.device:
                prev = prev.to(flow.device)
            prev = prev.unsqueeze(1)                                      # (B, 1, 15)

        out = self.forward(flow, hidden=hidden, audio=audio,
                           context=self._context, ego_player=ego,
                           prev_actions=prev)
        self._context = out["context"]

        mov = out["movement"][:, -1]                                      # (B, 9)
        btn = out["buttons"][:, -1]                                       # (B, 6)
        self._prev_action = torch.cat([
            torch.softmax(mov, dim=-1),
            torch.sigmoid(btn),
        ], dim=-1)                                                        # (B, 15)

        actions = {
            "movement": mov.squeeze(0),
            "buttons":  btn.squeeze(0),
            "value":    out["value"][:, -1].squeeze(0),
        }
        return actions, out["hidden"]
