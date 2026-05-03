"""CNN → LSTM policy model for SF3.

One optical-flow frame per LSTM timestep.  The LSTM's hidden state carries
over across the full episode during RL / inference, and across a truncated
sequence during behaviour-cloning.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Small 2-channel CNN (optical-flow input → feature vector)
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
# Full model
# ---------------------------------------------------------------------------
class SF3RNN(nn.Module):
    def __init__(self,
                 feature_dim: int = 256,
                 lstm_hidden: int = 512,
                 lstm_layers: int = 2,
                 lstm_dropout: float = 0.1,
                 num_directions: int = 9,
                 num_buttons: int = 6):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        self.encoder = _FlowEncoder(feature_dim)
        self.lstm = nn.LSTM(feature_dim, lstm_hidden,
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

        # Carried across episode steps during online play
        self.hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # ------------------------------------------------------------------
    def reset_hidden(self):
        self.hidden = None

    # ------------------------------------------------------------------
    def forward(self,
                flows: torch.Tensor,                          # (B, T, 2, H, W)
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                ) -> Dict[str, torch.Tensor]:
        """One LSTM step per optical-flow frame.

        Returns
        -------
        "movement"  (B, T, 9)  raw logits
        "buttons"   (B, T, 6)  raw logits (sigmoid → probability)
        "value"     (B, T)     scalar value estimate
        "hidden"    tuple of final (h, c)
        """
        feats = self.encoder(flows)                           # (B, T, F)
        lstm_out, h = self.lstm(feats, hidden)                # (B, T, H)

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
             ) -> Tuple[Dict[str, torch.Tensor],
                        Tuple[torch.Tensor, torch.Tensor]]:
        """Single-timestep inference.  *flow* is the (delayed) OF frame."""
        out = self.forward(flow, hidden=hidden)
        actions = {k: out[k][:, -1] for k in ("movement", "buttons", "value")}
        return actions, (out["hidden"][0], out["hidden"][1])
