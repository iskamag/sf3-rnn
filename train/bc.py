"""Behavior-cloning training loop.

The model runs once per *temporal_stride* (N) frames and produces N action
sets in one recurrence step.  Targets are reshaped accordingly and the loss
sums over all N outputs per step.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
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
    """Categorical CE for movement + binary CE for buttons — over all N slots."""
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
        self.N = cfg.temporal_stride
        assert cfg.seq_len % self.N == 0, \
            f"seq_len ({cfg.seq_len}) must be divisible by temporal_stride ({self.N})"
        self.recur_steps = cfg.seq_len // self.N

        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self._is_cpu = self.device.type == "cpu"
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        if self._is_cpu:
            torch.set_num_threads(torch.get_num_threads())

        self.model = SF3RNN(
            feature_dim=cfg.feature_dim,
            use_audio=cfg.use_audio,
            audio_dim=cfg.audio_dim,
            context_dim=cfg.context_dim,
            context_layers=cfg.context_layers,
            action_embed_dim=cfg.action_embed_dim,
            temporal_stride=cfg.temporal_stride,
            mingru_hidden=cfg.mingru_hidden,
            mingru_layers=cfg.mingru_layers,
            mingru_dropout=cfg.mingru_dropout,
            num_directions=cfg.num_directions,
            num_buttons=cfg.num_buttons,
            cnn_channels=cfg.cnn_channels,
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
        self.use_amp = cfg.use_amp and self.device.type == "cuda"
        self.amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)

        if resume is not None:
            self._load(resume)

        self.dataset = ReplayDataset(
            cfg.processed_dir, cfg.seq_len, cfg.total_delay_frames,
            audio_offset=cfg.total_delay_frames - (cfg.audio_lag_frames + cfg.motor_lag_frames),
            use_audio=cfg.use_audio, segment_stride=cfg.segment_stride,
            cache_size=max(2, cfg.batch_size))

        if cfg.compile_model:
            self.model = torch.compile(self.model)

    # ------------------------------------------------------------------
    def run(self):
        cfg = self.cfg
        for epoch in range(self.epoch, cfg.bc_epochs):
            self.epoch = epoch
            self.model.train()
            total_loss = 0.0
            total_chunks = 0
            t0 = time.time()

            sources = [
                fi for fi in range(self.dataset.num_sources())
                if self.dataset.sequences_for_file(fi)
            ]
            random.shuffle(sources)
            progress_total = sum(
                len(self.dataset.sequences_for_file(fi)) for fi in sources)
            pbar = tqdm(total=progress_total,
                        desc=f"BC epoch {epoch+1}/{cfg.bc_epochs}")

            for group_start in range(0, len(sources), cfg.batch_size):
                group = sources[group_start:group_start + cfg.batch_size]
                seqs = [self.dataset.sequences_for_file(fi) for fi in group]
                egos = [random.randint(0, 1) for _ in group]
                hiddens: list[Optional[torch.Tensor]] = [None] * len(group)
                contexts: list[Optional[torch.Tensor]] = [None] * len(group)

                for step_idx in range(max(len(s) for s in seqs)):
                    active = [i for i, s in enumerate(seqs) if step_idx < len(s)]
                    samples = []
                    for i in active:
                        start = seqs[i][step_idx]
                        samples.append(self.dataset.get_sequence(group[i], start, egos[i]))
                    batch = default_collate(samples)

                    hidden = None
                    if any(hiddens[i] is not None for i in active):
                        hidden = torch.cat([hiddens[i] for i in active], dim=1)

                    context = None
                    if any(contexts[i] is not None for i in active):
                        context = torch.cat([contexts[i] for i in active], dim=1)

                    loss, next_hidden, next_context = self._train_batch(
                        batch, hidden, context)

                    for col, i in enumerate(active):
                        hiddens[i] = next_hidden[:, col:col + 1].detach()
                        contexts[i] = next_context[:, col:col + 1].detach()

                    batch_chunks = len(active)
                    total_loss += loss.item() * batch_chunks
                    total_chunks += batch_chunks
                    pbar.update(batch_chunks)
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

            pbar.close()

            avg = total_loss / max(1, total_chunks)
            self.writer.add_scalar("loss/train", avg, epoch)
            print(f"[BC] epoch {epoch+1:3d}  loss={avg:.4f}  "
                  f"time={time.time()-t0:.1f}s")
            self._save()

    def _train_batch(self,
                     batch: dict[str, torch.Tensor],
                     hidden: Optional[torch.Tensor],
                     context: Optional[torch.Tensor],
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L = batch["movement"].shape                                    # L = seq_len (frames)
        T, N = self.recur_steps, self.N

        # Reshape flat-frame data → (B, T, N, ...)
        flows       = batch["flows"].to(self.device, non_blocking=True)    # (B, L, 2, H, W)
        flows       = flows.reshape(B, T, N, *flows.shape[2:])
        movement    = batch["movement"].to(self.device, non_blocking=True)  # (B, L)
        movement    = movement.reshape(B, T, N)
        buttons     = batch["buttons"].to(self.device, non_blocking=True)  # (B, L, 6)
        buttons     = buttons.reshape(B, T, N, -1)
        ego = batch["ego_player"].to(self.device, dtype=torch.float32).unsqueeze(1)

        audio = None
        if self.cfg.use_audio:
            audio = batch["audio"].to(self.device, non_blocking=True)       # (B, L, 15)
            audio = audio.reshape(B, T, N, -1)

        if hidden is not None:
            hidden = hidden.to(self.device, non_blocking=True)
        if context is not None:
            context = context.to(self.device, non_blocking=True)

        # --- previous-action inputs (teacher forcing, shifted by one step) ---
        mov_oh = F.one_hot(movement, num_classes=self.cfg.num_directions).float()   # (B, T, N, 9)
        shifted_mov = torch.cat([
            torch.zeros(B, 1, N, self.cfg.num_directions, device=self.device),
            mov_oh[:, :-1],
        ], dim=1)                                                                    # (B, T, N, 9)
        shifted_btn = torch.cat([
            torch.zeros(B, 1, N, self.cfg.num_buttons, device=self.device),
            buttons[:, :-1],
        ], dim=1)                                                                    # (B, T, N, 6)
        prev_actions = torch.cat([shifted_mov, shifted_btn], dim=-1)                # (B, T, N, 15)

        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            out = self.model(flows, ego, hidden=hidden,
                             audio=audio, context=context,
                             prev_actions=prev_actions)
            loss = bc_loss(out, {"movement": movement, "buttons": buttons})

        self.optim.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optim.step()
        return loss.detach(), out["hidden"].detach(), out["context"].detach()

    # ------------------------------------------------------------------
    def _save(self):
        p = self.cfg.ckpt_dir / f"bc_epoch{self.epoch:03d}.pt"
        model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        torch.save({
            "epoch": self.epoch,
            "model": model.state_dict(),
            "optim_name": "muon",
            "optim": self.optim.state_dict(),
        }, p)

    def _load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"], strict=False)
        if ckpt.get("optim_name") == "muon":
            self.optim.load_state_dict(ckpt["optim"])
        else:
            print("[BC] Checkpoint optimizer is not Muon; resetting optimizer state")
        self.epoch = ckpt["epoch"] + 1
