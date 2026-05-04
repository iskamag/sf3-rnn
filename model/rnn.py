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
                 mingru_hidden: int = 512,
                 mingru_layers: int = 2,
                 mingru_dropout: float = 0.1,
                 num_directions: int = 9,
                 num_buttons: int = 6,
                 use_match_context: bool = False,
                 ):
        super().__init__()
        self.mingru_hidden = mingru_hidden
        self.mingru_layers = mingru_layers
        self.use_audio = use_audio
        self.context_dim = context_dim
        self.use_match_context = use_match_context

        self.encoder = _FlowEncoder(feature_dim)

        # +1 for the ego_player indicator bit (0=P1, 1=P2)
        recurrent_input = feature_dim + context_dim + 1
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

        if not self.use_match_context:
            ego = ego_player.unsqueeze(1).expand(-1, T, -1)
            ctx_seq = ctx.unsqueeze(1).expand(-1, T, -1)
            parts = [vis, ctx_seq, ego]
            if self.use_audio:
                parts.insert(1, aud)
            recurrent_in = torch.cat(parts, dim=-1)
            recurrent_out, hidden = self.recurrent(recurrent_in, hidden)
        else:
            # Context profiling is opt-in because it prevents vectorized recurrent execution.
            recurrent_outs = []
            for t in range(T):
                parts = [vis[:, t], ctx, ego_player]
                if self.use_audio:
                    parts.insert(1, aud[:, t])
                step = torch.cat(parts, dim=-1).unsqueeze(1)

                out, hidden = self.recurrent(step, hidden)    # (B, 1, H)
                ctx = self.match_context(out.squeeze(1), ctx) # update profile
                recurrent_outs.append(out)

            recurrent_out = torch.cat(recurrent_outs, dim=1)   # (B, T, H)

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
        ego = torch.tensor([[ego_player]], device=flow.device, dtype=torch.float32)
        out = self.forward(flow, hidden=hidden, audio=audio,
                           context=self._context, ego_player=ego)
        self._context = out["context"] if self.use_match_context else None
        actions = {k: out[k][:, -1].squeeze(0) for k in ("movement", "buttons", "value")}
        return actions, out["hidden"]
