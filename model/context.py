"""Persistent match-context vector — accumulates opponent profile.

A small multi-layer GRU that the main minGRU reads and writes at every timestep.
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
    def __init__(self, input_dim: int = 512, context_dim: int = 128,
                 num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.context_dim = context_dim
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else context_dim
            self.cells.append(nn.GRUCell(in_dim, context_dim))

    def forward(self,
                recurrent_out: torch.Tensor,                   # (B, H)
                prev_context: torch.Tensor,                    # (L, B, C)
                ) -> torch.Tensor:                             # (L, B, C)
        """Update the match profile with one timestep of recurrent output."""
        new_ctx = []
        x = recurrent_out
        for i, cell in enumerate(self.cells):
            x = cell(x, prev_context[i])
            new_ctx.append(x)
        return torch.stack(new_ctx, dim=0)

    def initial_context(self, batch_size: int,
                        device: torch.device) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size,
                           self.context_dim, device=device)
