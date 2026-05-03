"""CNN + Audio → LSTM policy model for SF3.

Visual:  optical flow (2-ch) → FlowCNN → feature_dim
Audio:   MFCC + energy features  → AudioMLP → audio_dim
Fused:   concat(visual, audio)   → LSTM → movement / buttons / value

Audio features are always expected.  During BC training (no audio capture)
they are filled with zeros; the model learns to rely on visual features
primarily, and audio becomes additive during PPO / inference.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
# Full model
# ---------------------------------------------------------------------------
class SF3RNN(nn.Module):
    def __init__(self,
                 feature_dim: int = 256,
                 audio_dim: int = 64,
                 lstm_hidden: int = 512,
                 lstm_layers: int = 2,
                 lstm_dropout: float = 0.1,
                 num_directions: int = 9,
                 num_buttons: int = 6,
                 audio_feat_dim: int = 15,
                 ):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        self.encoder = _FlowEncoder(feature_dim)
        self.audio_encoder = _AudioEncoder(audio_feat_dim, audio_dim)

        self.lstm = nn.LSTM(feature_dim + audio_dim, lstm_hidden,
                            num_layers=lstm_layers,
                            dropout=lstm_dropout,
                            batch_first=True)

        head = [lstm_hidden, lstm_hidden // 2]
        self.movement_head = nn.Sequential(
            nn.Linear(*head), nn.ReLU(), nn.Linear(head[1], num_directions),
        )
        self.button_head = nn.Sequential(
            nn.Linear(*head), nn.ReLU(), nn.Linear(head[1], num_buttons),
        )
        self.value_head = nn.Sequential(
            nn.Linear(*head), nn.ReLU(), nn.Linear(head[1], 1),
        )

        self.hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # ------------------------------------------------------------------
    def reset_hidden(self):
        self.hidden = None

    # ------------------------------------------------------------------
    def forward(self,
                flows: torch.Tensor,                          # (B, T, 2, H, W)
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                audio: Optional[torch.Tensor] = None,          # (B, T, 15) or None
                ) -> Dict[str, torch.Tensor]:
        vis = self.encoder(flows)                             # (B, T, F)

        if audio is not None:
            aud = self.audio_encoder(audio)                   # (B, T, A)
            feats = torch.cat([vis, aud], dim=-1)              # (B, T, F+A)
        else:
            B, T = vis.shape[:2]
            zeros = torch.zeros(B, T, self.audio_encoder.net[-1].out_features,
                                device=vis.device, dtype=vis.dtype)
            feats = torch.cat([vis, zeros], dim=-1)

        lstm_out, h = self.lstm(feats, hidden)

        return {
            "movement": self.movement_head(lstm_out),
            "buttons":  self.button_head(lstm_out),
            "value":    self.value_head(lstm_out).squeeze(-1),
            "hidden":   h,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self,
             flow: torch.Tensor,                              # (1, 1, 2, H, W)
             hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
             audio_feats: Optional[torch.Tensor] = None,       # (1, 15)
             ) -> Tuple[Dict[str, torch.Tensor],
                        Tuple[torch.Tensor, torch.Tensor]]:
        """Single-timestep inference.  *flow* and *audio_feats* are the
        (separately delayed) sensory inputs."""
        audio = audio_feats.unsqueeze(1) if audio_feats is not None else None
        out = self.forward(flow, hidden=hidden, audio=audio)
        actions = {k: out[k][:, -1] for k in ("movement", "buttons", "value")}
        return actions, (out["hidden"][0], out["hidden"][1])
