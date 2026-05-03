"""Persistent match-context vector — accumulates opponent profile.

A small GRU that the main LSTM reads and writes at every timestep.
The context survives across BC training segments (detached at boundaries)
so the model can learn to profile an opponent over a full match.

Analogous to the "working memory" a human player maintains:
  "she's been waking up with DP" → adjust oki
  "he never techs throws"       → throw more
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MatchContext(nn.Module):
    def __init__(self, input_dim: int = 512, context_dim: int = 64):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, context_dim)

    def forward(self,
                lstm_out: torch.Tensor,                        # (B, H)
                prev_context: torch.Tensor,                    # (B, C)
                ) -> torch.Tensor:                             # (B, C)
        """Update the match profile with one timestep of LSTM output."""
        return self.gru(lstm_out, prev_context)

    def initial_context(self, batch_size: int,
                        device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)
