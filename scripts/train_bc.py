#!/usr/bin/env python3
"""Behavior-cloning entry point.

    python scripts/train_bc.py [--resume checkpoints/bc_epoch042.pt]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import Config
from train.bc import BCTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=Path, default=None)
    args = p.parse_args()

    cfg = Config()
    trainer = BCTrainer(cfg, resume=args.resume)
    trainer.run()


if __name__ == "__main__":
    main()
