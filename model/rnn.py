"""CNN + (optional) Audio + MatchContext -> minGRU policy model for SF3.

Visual:    N optical-flow frames  → FlowCNN      → N × feature_dim
Audio:     N MFCC+energy frames   → pool → AudioMLP → audio_dim
Context:   minGRU output          → MatchContext  → context_dim
Fused:     [pool(vis), pool(aud), ctx, ego, pool(prev_actions)]  → minGRU
                                                                  → [flow_feat | hidden] → N action heads

The model runs once per *temporal_stride* (N) frames and produces N consecutive
action sets in one recurrence step — matching the human ~67 ms decision cadence.
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


# ---------------------------------------------------------------------------
# minGRU stack wrapper
# ---------------------------------------------------------------------------
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
                 temporal_stride: int = 4,
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
        self.temporal_stride = temporal_stride
        self.N = temporal_stride                                        # shorthand
        self.action_dim = num_directions + num_buttons                  # 15
        self.head_out_dim = num_directions + num_buttons + 1            # 16  (mov+btn+val)

        self.encoder = _FlowEncoder(feature_dim)

        # Single-action embedder — applied per-frame, then pooled across N
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

        # Project [hidden | flow_feat] → hidden for the action heads
        self._action_proj = nn.Linear(mingru_hidden + feature_dim, mingru_hidden)

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
        self._context: Optional[torch.Tensor] = None                   # (B, context_dim)
        self._prev_action: Optional[torch.Tensor] = None                # (B, N, 15)

    # ------------------------------------------------------------------
    def reset_hidden(self):
        self.hidden = None
        self._context = None
        self._prev_action = None

    def detach_context(self):
        if self._context is not None:
            self._context = self._context.detach()

    # ------------------------------------------------------------------
    def forward(self,
                flows: torch.Tensor,                          # (B, T, N, 2, H, W)
                ego_player: torch.Tensor,                      # (B, 1) float 0=P1, 1=P2
                hidden: HiddenState = None,
                audio: Optional[torch.Tensor] = None,          # (B, T, N, 15) or (B, T, 15)
                context: Optional[torch.Tensor] = None,         # (B, C) initial context
                prev_actions: Optional[torch.Tensor] = None,    # (B, T, N, D)  D=num_dir+num_btn
                ) -> Dict[str, torch.Tensor]:
        # --- visual ---
        # flows: (B, T, N, 2, H, W)  →  flat: (B*T*N, 2, H, W)  →  feats: (B*T*N, F)
        # reshaped back to (B, T, N, F)
        B, T, N = flows.shape[:3]
        device = flows.device
        flow_feats = self._encoder_flat(flows, (B, T, N))             # (B, T, N, F)

        # --- audio ---
        aud_pooled = None
        if self.use_audio:
            if audio is None:
                raise ValueError("use_audio=True but no audio tensor passed to forward()")
            if audio.dim() == 4:                                       # (B, T, N, 15) → pool
                aud_pooled = audio.mean(dim=2)                         # (B, T, 15)
            else:
                aud_pooled = audio                                     # (B, T, 15) already pooled
            aud_pooled = self.audio_encoder(aud_pooled)                # (B, T, A)

        # --- context ---
        if context is None:
            ctx = self.match_context.initial_context(B, device)
        else:
            ctx = context

        # --- ego player ---
        ego = ego_player.unsqueeze(1).expand(-1, T, -1)               # (B, T, 1)

        # --- previous actions ---
        if prev_actions is not None:
            # Embed each of the N actions, then pool across N
            act_emb = self._action_embed(prev_actions)                 # (B, T, N, E)
            act_emb = act_emb.mean(dim=2)                              # (B, T, E)
        else:
            act_emb = torch.zeros(B, T, self._action_embed[-1].out_features,
                                  device=device)

        # --- pooled flow for recurrence ---
        flow_pooled = flow_feats.mean(dim=2)                           # (B, T, F)

        # --- recurrent core ---
        ctx_seq = ctx.unsqueeze(1).expand(-1, T, -1)                  # (B, T, C)
        parts = [flow_pooled, ctx_seq, ego, act_emb]
        if aud_pooled is not None:
            parts.insert(1, aud_pooled)
        recurrent_in = torch.cat(parts, dim=-1)
        recurrent_out, hidden = self.recurrent(recurrent_in, hidden)   # (B, T, H)

        # --- action decoder: [hidden | flow_feat] per frame ---
        h_exp = recurrent_out.unsqueeze(2).expand(-1, -1, N, -1)      # (B, T, N, H)
        slot_in = self._action_proj(torch.cat([h_exp, flow_feats], dim=-1))  # (B, T, N, H)

        # --- update context ---
        ctx = self.match_context(recurrent_out[:, -1], ctx)

        return {
            "movement": self.movement_head(slot_in),                   # (B, T, N, 9)
            "buttons":  self.button_head(slot_in),                     # (B, T, N, 6)
            "value":    self.value_head(slot_in).squeeze(-1),          # (B, T, N)
            "hidden":   hidden,
            "context":  ctx,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self,
             flows: torch.Tensor,                             # (B, N, 2, H, W)  or  (B, 1, N, 2, H, W)
             hidden: HiddenState = None,
             audio_feats: Optional[torch.Tensor] = None,       # (B, N, 15)
             ego_player: float = 0.0,
             ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Single multi-frame inference step.  Produces N consecutive actions."""
        if flows.dim() == 5:                                           # (B, N, 2, H, W)
            flows = flows.unsqueeze(1)                                  # (B, 1, N, 2, H, W)

        B = flows.shape[0]
        device = flows.device

        if self.use_audio and audio_feats is None:
            audio_feats = torch.zeros((B, self.N, self.audio_encoder.net[0].in_features),
                                      device=device)
        audio = audio_feats.unsqueeze(1) if audio_feats is not None else None   # (B, 1, N, 15)

        ego = torch.full((B, 1), ego_player, device=device, dtype=torch.float32)

        prev = self._prev_action                                          # (B, N, 15) or None
        if prev is not None:
            if prev.device != device:
                prev = prev.to(device)
            prev = prev.unsqueeze(1)                                      # (B, 1, N, 15)

        out = self.forward(flows, hidden=hidden, audio=audio,
                           context=self._context, ego_player=ego,
                           prev_actions=prev)
        self._context = out["context"]

        # Store this step's actions as prev_action for next step
        mov = out["movement"][:, -1]                                       # (B, N, 9)
        btn = out["buttons"][:, -1]                                        # (B, N, 6)
        self._prev_action = torch.cat([
            torch.softmax(mov, dim=-1),
            torch.sigmoid(btn),
        ], dim=-1)                                                         # (B, N, 15)

        # Return squeezed (B=1 → scalar) actions for caller compatibility
        actions = {
            "movement": mov.squeeze(0),                                    # (N, 9)
            "buttons":  btn.squeeze(0),                                    # (N, 6)
            "value":    out["value"][:, -1].squeeze(0),                    # (N,)
        }
        return actions, out["hidden"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _encoder_flat(self, flows: torch.Tensor, lead_shape: tuple) -> torch.Tensor:
        """Encode flow frames with arbitrary leading batch dims."""
        lead = flows.shape[:-3]                            # e.g. (B, T, N)
        flat = flows.reshape(-1, 2, flows.shape[-2], flows.shape[-1])
        h = self.encoder.conv(flat)
        h = self.encoder.pool(h).flatten(1)
        feats = self.encoder.fc(h)
        return feats.reshape(*lead, -1)                    # (B, T, N, F)
