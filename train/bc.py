"""Behavior-cloning training loop.

Imitation learning from human replays.  The LSTM receives one optical-flow
frame per timestep; the reaction-time delay is baked into the dataset's
target offset.  Loss: CE (movement) + BCE (buttons) summed over timesteps.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from data.dataset import ReplayDataset
from model.rnn import SF3RNN


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def bc_loss(outputs: dict[str, torch.Tensor],
            targets: dict[str, torch.Tensor]) -> torch.Tensor:
    """Categorical CE for movement + binary CE for buttons."""
    movement_loss = nn.CrossEntropyLoss()(
        outputs["movement"].reshape(-1, outputs["movement"].size(-1)),
        targets["movement"].reshape(-1),
    )
    button_loss = nn.BCEWithLogitsLoss()(
        outputs["buttons"].reshape(-1, outputs["buttons"].size(-1)),
        targets["buttons"].reshape(-1, targets["buttons"].size(-1)),
    )
    return movement_loss + button_loss


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class BCTrainer:
    def __init__(self, cfg: Config, resume: Optional[Path] = None):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.model = SF3RNN(
            feature_dim=cfg.feature_dim,
            use_audio=cfg.use_audio,
            audio_dim=cfg.audio_dim,
            lstm_hidden=cfg.lstm_hidden,
            lstm_layers=cfg.lstm_layers,
            lstm_dropout=cfg.lstm_dropout,
            num_directions=cfg.num_directions,
            num_buttons=cfg.num_buttons,
        ).to(self.device)

        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        self.writer = SummaryWriter(cfg.log_dir / "bc")
        self.epoch = 0

        if resume is not None:
            self._load(resume)

        self.dataset = ReplayDataset(
            cfg.processed_dir, cfg.seq_len, cfg.total_delay_frames,
            use_audio=cfg.use_audio)
        self.loader = DataLoader(
            self.dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    # ------------------------------------------------------------------
    def run(self):
        cfg = self.cfg
        for epoch in range(self.epoch, cfg.bc_epochs):
            self.epoch = epoch
            self.model.train()
            total_loss = 0.0
            t0 = time.time()

            pbar = tqdm(self.loader, desc=f"BC epoch {epoch+1}/{cfg.bc_epochs}")
            for batch in pbar:
                flows = batch["flows"].to(self.device, non_blocking=True)
                movement = batch["movement"].to(self.device, non_blocking=True)
                buttons = batch["buttons"].to(self.device, non_blocking=True)

                kwargs = {}
                if cfg.use_audio:
                    kwargs["audio"] = batch["audio"].to(self.device, non_blocking=True)

                out = self.model(flows, **kwargs)
                loss = bc_loss(out, {"movement": movement, "buttons": buttons})

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg = total_loss / len(self.loader)
            self.writer.add_scalar("loss/train", avg, epoch)
            print(f"[BC] epoch {epoch+1:3d}  loss={avg:.4f}  "
                  f"time={time.time()-t0:.1f}s")
            self._save()

    # ------------------------------------------------------------------
    def _save(self):
        p = self.cfg.ckpt_dir / f"bc_epoch{self.epoch:03d}.pt"
        torch.save({
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
        }, p)

    def _load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optim.load_state_dict(ckpt["optim"])
        self.epoch = ckpt["epoch"] + 1
