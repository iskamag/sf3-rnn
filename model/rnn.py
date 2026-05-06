"""CNN + (optional) Audio + MatchContext -> minGRU policy model for SF3.

Visual:    N optical-flow frames  →  pool  →  FlowCNN  →  feature_dim
Audio:     N MFCC+energy frames   → encode →  max-pool →  audio_dim
Context:   minGRU output          →  MatchContext (multi-layer GRU)  →  context_dim
Fused:     [vis, aud, ctx, ego, prev_actions]  →  minGRU
                                                     →  [hidden + pos_embedₙ] → N action heads

The model runs once per *temporal_stride* (N) frames and produces N consecutive
action sets in one recurrence step — matching the human ~67 ms decision cadence.

Per-frame variation comes from a learned sub-frame position embedding, not
per-frame CNN features — this cuts CNN FLOPs by N× (4× at default stride).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from minGRU_pytorch import minGRU

from model.context import MatchContext


HiddenState = Optional[torch.Tensor]


# ---------------------------------------------------------------------------
# Visual encoder — 2-channel optical flow → feature vector
# ---------------------------------------------------------------------------
class _FlowEncoder(nn.Module):
    """Lightweight convolutional flow encoder.

    Uses InstanceNorm (no running stats overhead on CPU) and smaller channel
    counts to keep the encoder efficient while preserving receptive field.
    Base channel count ``ch`` controls the capacity/speed trade-off:
      ch=16  → ~36 MFLOPs/frame
      ch=24  → ~78 MFLOPs/frame
      ch=32  → ~126 MFLOPs/frame
    """
    def __init__(self, feature_dim: int = 256, ch: int = 16):
        super().__init__()
        c = ch
        self.conv = nn.Sequential(
            nn.Conv2d(2, c, 5, stride=2, padding=2), nn.InstanceNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.InstanceNorm2d(c * 2), nn.ReLU(),
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1), nn.InstanceNorm2d(c * 4), nn.ReLU(),
            nn.Conv2d(c * 4, c * 4, 3, stride=2, padding=1), nn.InstanceNorm2d(c * 4), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        fc_in = c * 4 * 4 * 4
        self.fc = nn.Sequential(
            nn.Linear(fc_in, feature_dim),
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
                 context_dim: int = 128,
                 context_layers: int = 2,
                 action_embed_dim: int = 64,
                 temporal_stride: int = 4,
                 mingru_hidden: int = 512,
                 mingru_layers: int = 2,
                 mingru_dropout: float = 0.1,
                 num_directions: int = 9,
                 num_buttons: int = 6,
                 cnn_channels: int = 16,
                 ):
        super().__init__()
        self.mingru_hidden = mingru_hidden
        self.mingru_layers = mingru_layers
        self.use_audio = use_audio
        self.context_dim = context_dim
        self.context_layers = context_layers
        self.num_directions = num_directions
        self.num_buttons = num_buttons
        self.temporal_stride = temporal_stride
        self.N = temporal_stride                                        # shorthand
        self.action_dim = num_directions + num_buttons                  # 15
        self.head_out_dim = num_directions + num_buttons + 1            # 16  (mov+btn+val)

        self.encoder = _FlowEncoder(feature_dim, cnn_channels)

        # Per-frame action embedder
        self.action_embed_dim = action_embed_dim
        self._action_embed = nn.Sequential(
            nn.Linear(self.action_dim, action_embed_dim),
            nn.ReLU(),
            nn.Linear(action_embed_dim, action_embed_dim),
        )

        # Temporal encoder over the N prev-action embeddings. The depthwise 1D
        # causal conv preserves intra-stride button-press ordering.
        self._action_temporal = nn.Sequential(
            nn.Conv1d(action_embed_dim, action_embed_dim,
                      kernel_size=3, padding=0, groups=action_embed_dim),
            nn.ReLU(),
            nn.Conv1d(action_embed_dim, action_embed_dim, kernel_size=1),
        )

        # +1 for the ego_player indicator bit (0=P1, 1=P2)
        recurrent_input = feature_dim + context_dim + action_embed_dim + 1
        if use_audio:
            self.audio_encoder = _AudioEncoder(audio_feat_dim, audio_dim)
            recurrent_input += audio_dim
        else:
            self.audio_encoder = None

        self.match_context = MatchContext(mingru_hidden, context_dim, context_layers)

        self.recurrent = _MinGRUStack(recurrent_input, mingru_hidden,
                                       mingru_layers, mingru_dropout)

        # Per-subframe position embedding — replaces per-frame CNN features
        # so the action heads can vary outputs within the stride window.
        self._subframe_pos = nn.Parameter(torch.zeros(temporal_stride, mingru_hidden))

        # Action head projection: hidden → hidden (no flow concat needed)
        self._action_proj = nn.Linear(mingru_hidden, mingru_hidden)

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
        self._context: Optional[torch.Tensor] = None                   # (L, B, context_dim)
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
    def _encode_prev_actions(self, prev_actions: torch.Tensor
                             ) -> torch.Tensor:
        """Encode N prev-action distributions → (B, T, E) vector.

        Uses a per-frame MLP followed by a small causal 1D conv that
        preserves intra-stride ordering.
        """
        B, T, N = prev_actions.shape[:3]

        # Per-frame embedding
        emb = self._action_embed(prev_actions)                       # (B, T, N, E)

        # Causal temporal conv over the N frames
        BT = B * T
        x = emb.reshape(BT, N, -1).permute(0, 2, 1)                  # (BT, E, N)
        x = F.pad(x, (2, 0))                                          # causal left-pad
        x = self._action_temporal(x)                                  # (BT, E, N)
        return x[:, :, -1].reshape(B, T, -1)                          # (B, T, E)

    def _encode_prev_actions_fallback(self, B: int, T: int, device: torch.device
                                      ) -> torch.Tensor:
        """Zero-initialised prev-action encoding when no prior is available."""
        return torch.zeros(B, T, self.action_embed_dim,
                           device=device)

    # ------------------------------------------------------------------
    def forward(self,
                flows: torch.Tensor,                          # (B, T, N, 2, H, W)
                ego_player: torch.Tensor,                      # (B, 1) float 0=P1, 1=P2
                hidden: HiddenState = None,
                audio: Optional[torch.Tensor] = None,          # (B, T, N, 15) or (B, T, 15)
                context: Optional[torch.Tensor] = None,         # (L, B, C) multi-layer state
                prev_actions: Optional[torch.Tensor] = None,    # (B, T, N, D)  D=num_dir+num_btn
                ) -> Dict[str, torch.Tensor]:
        B, T, N = flows.shape[:3]
        device = flows.device

        # --- visual: pool N flow frames → 1 representative frame → CNN once ---
        flow_pooled_frames = flows.mean(dim=2)                     # (B, T, 2, H, W)
        flow_feat = self.encoder(flow_pooled_frames)               # (B, T, F)

        # --- audio: encode per-frame, then max-pool (captures onset events) ---
        aud_pooled = None
        if self.use_audio:
            if audio is None:
                raise ValueError("use_audio=True but no audio tensor passed to forward()")
            if audio.dim() == 4:                                       # (B, T, N, 15)
                aud_encoded = self.audio_encoder(audio)                # (B, T, N, A)
                aud_pooled = aud_encoded.max(dim=2).values             # (B, T, A) — max for onsets
            else:
                aud_pooled = self.audio_encoder(audio)                 # (B, T, 15) already pooled

        # --- context: multi-layer GRU state, last layer feeds minGRU ---
        if context is None:
            full_ctx = self.match_context.initial_context(B, device)   # (L, B, C)
        else:
            full_ctx = context                                         # (L, B, C)
        ctx = full_ctx[-1]                                             # (B, C) for minGRU input

        # --- ego player ---
        ego = ego_player.unsqueeze(1).expand(-1, T, -1)               # (B, T, 1)

        # --- prev actions: temporal conv preserves intra-stride ordering ---
        if prev_actions is not None:
            act_emb = self._encode_prev_actions(prev_actions)          # (B, T, E)
        else:
            act_emb = self._encode_prev_actions_fallback(B, T, device)

        # --- recurrent core ---
        ctx_seq = ctx.unsqueeze(1).expand(-1, T, -1)                  # (B, T, C)
        parts = [flow_feat, ctx_seq, ego, act_emb]
        if aud_pooled is not None:
            parts.insert(1, aud_pooled)
        recurrent_in = torch.cat(parts, dim=-1)
        recurrent_out, hidden = self.recurrent(recurrent_in, hidden)   # (B, T, H)

        # --- action decoder: hidden + subframe position embedding ---
        pos = self._subframe_pos.unsqueeze(0).unsqueeze(0)             # (1, 1, N, H)
        h_exp = recurrent_out.unsqueeze(2) + pos                       # (B, T, N, H)
        slot_in = self._action_proj(h_exp)                             # (B, T, N, H)

        # --- update context ---
        full_ctx = self.match_context(recurrent_out, full_ctx)

        return {
            "movement": self.movement_head(slot_in),                   # (B, T, N, 9)
            "buttons":  self.button_head(slot_in),                     # (B, T, N, 6)
            "value":    self.value_head(slot_in).squeeze(-1),          # (B, T, N)
            "hidden":   hidden,
            "context":  full_ctx,                                       # (L, B, C)
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
