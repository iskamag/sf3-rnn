"""Behavior-cloning training loop.

Imitation learning from human replays.  The minGRU receives one optical-flow
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
from train.muon import make_muon_optimizer


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
            context_dim=cfg.context_dim,
            mingru_hidden=cfg.mingru_hidden,
            mingru_layers=cfg.mingru_layers,
            mingru_dropout=cfg.mingru_dropout,
            num_directions=cfg.num_directions,
            num_buttons=cfg.num_buttons,
            use_match_context=cfg.use_match_context,
        ).to(self.device)

        self.optim = make_muon_optimizer(
            self.model,
            lr=cfg.muon_lr,
            adamw_lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=cfg.muon_momentum,
            ns_steps=cfg.muon_ns_steps,
        )

        self.writer = SummaryWriter(cfg.log_dir / "bc")
        self.epoch = 0

        if resume is not None:
            self._load(resume)

        self.dataset = ReplayDataset(
            cfg.processed_dir, cfg.seq_len, cfg.total_delay_frames,
            use_audio=cfg.use_audio, segment_stride=cfg.segment_stride)
        loader_kwargs = {
            "batch_size": cfg.batch_size,
            "shuffle": False,
            "num_workers": cfg.num_workers,
            "pin_memory": self.device.type == "cuda",
            "drop_last": True,
        }
        if cfg.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 1
        self.loader = DataLoader(self.dataset, **loader_kwargs)

    # ------------------------------------------------------------------
    def run(self):
        cfg = self.cfg
        for epoch in range(self.epoch, cfg.bc_epochs):
            self.epoch = epoch
            self.model.train()
            self.model.reset_hidden()
            total_loss = 0.0
            t0 = time.time()
            last_file = None

            pbar = tqdm(self.loader, desc=f"BC epoch {epoch+1}/{cfg.bc_epochs}")
            for batch_idx, batch in enumerate(pbar):
                flows = batch["flows"].to(self.device, non_blocking=True)
                movement = batch["movement"].to(self.device, non_blocking=True)
                buttons = batch["buttons"].to(self.device, non_blocking=True)

                kwargs = {}
                ego = batch["ego_player"]
                if isinstance(ego, torch.Tensor):
                    ego = ego.to(self.device, dtype=torch.float32).unsqueeze(1)
                else:
                    ego = torch.tensor(ego, device=self.device,
                                       dtype=torch.float32).view(-1, 1)
                if cfg.use_audio:
                    kwargs["audio"] = batch["audio"].to(self.device, non_blocking=True)

                # Detect match boundary — reset context for new matches
                idx = batch_idx * cfg.batch_size
                if idx < len(self.dataset):
                    cur_file = self.dataset.file_of(idx)
                    if last_file is not None and cur_file != last_file:
                        self.model.reset_hidden()
                    last_file = cur_file

                out = self.model(flows, ego, **kwargs)
                loss = bc_loss(out, {"movement": movement, "buttons": buttons})

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()

                # Persist context for next segment, detach to block BPTT across segments
                self.model.detach_context()

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
            "optim_name": "muon",
            "optim": self.optim.state_dict(),
        }, p)

    def _load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if ckpt.get("optim_name") == "muon":
            self.optim.load_state_dict(ckpt["optim"])
        else:
            print("[BC] Checkpoint optimizer is not Muon; resetting optimizer state")
        self.epoch = ckpt["epoch"] + 1
