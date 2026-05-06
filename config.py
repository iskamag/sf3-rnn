import random
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Config:
    # --- Game ---
    fps: int = 60
    native_width: int = 384
    native_height: int = 224
    frame_width: int = 160
    frame_height: int = 120
    flow_width: int = 160
    flow_height: int = 120
    flow_norm: float = 20.0

    # Native-resolution HUD ROIs: (x, y, width, height). These are defaults
    # for SFIII 3S/Fightcade captures and may need calibration per crop.
    p1_health_roi: tuple[int, int, int, int] = (31, 17, 127, 8)
    p2_health_roi: tuple[int, int, int, int] = (226, 17, 127, 8)
    p1_stun_roi: tuple[int, int, int, int] = (31, 31, 127, 5)
    p2_stun_roi: tuple[int, int, int, int] = (226, 31, 127, 5)

    # --- Reaction-time constraint ---
    # Human pipelines have different latencies:
    #
    #   Visual:  photoreceptors → retina → V1 → higher cortex  ≈ 100–140 ms  ≈ 8 f
    #   Audio:   cochlea → brainstem → auditory cortex          ≈  40–60 ms  ≈ 3 f
    #   Motor:   cortex → spinal cord → muscle → USB poll       ≈  50–70 ms  ≈ 4 f
    #
    # The model sees visual flow delayed by sensory_lag, audio features
    # delayed by audio_lag (shorter), and actions are held by motor_lag.
    # Total end-to-end visual reaction = sensory_lag + motor_lag ≈ 200 ms.
    sensory_lag_frames: int = 8       # visual input delay
    audio_lag_frames: int = 3         # audio input delay  (faster pathway)
    motor_lag_frames: int = 4         # output delay

    # --- Match boundary detection ---
    match_start_offset_frames: int = 20  # cue match starts slightly before gameplay; skip ahead
    match_min_len: int = 120          # ignore very short audio/log match segments

    @property
    def total_delay_frames(self) -> int:
        return self.sensory_lag_frames + self.motor_lag_frames

    # --- Audio ---
    # Audio mode. Enabled runs with audio features and requires audio.wav
    # during collection. Disabled omits the audio encoder.
    use_audio: bool = True
    audio_sample_rate: int = 16000
    audio_feat_dim: int = 15         # MFCC (13) + RMS (1) + centroid (1)
    audio_dim: int = 64              # internal embedding size

    # --- Model ---
    feature_dim: int = 256
    action_embed_dim: int = 64     # prev-action embedding fed into minGRU
    temporal_stride: int = 4       # recurrence step span (frames); produces N actions/step
    mingru_hidden: int = 512
    mingru_layers: int = 2
    mingru_dropout: float = 0.1
    num_directions: int = 9      # neutral + 8-way
    num_buttons: int = 6         # LP, MP, HP, LK, MK, HK
    cnn_channels: int = 16       # base channels for FlowCNN (16=fast/CPU, 32=balanced, 24=default)

    # --- Training ---
    batch_size: int = 16
    seq_len: int = 96            # frames per training segment  (~1.6 s)
    segment_stride: int = 128    # ordered chunks; context carries across them
    lr: float = 1e-4
    muon_lr: float = 2e-3
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    weight_decay: float = 1e-5
    bc_epochs: int = 100
    use_amp: bool = True
    amp_dtype: str = "float16"
    compile_model: bool = True

    # --- Match-context profiling ---
    # A multi-layer GRU that the minGRU reads every timestep and updates at
    # chunk boundaries.  Persists across ordered BC chunks and full PPO
    # episodes, accumulating an opponent profile: tendencies, habits,
    # adaptation.
    context_dim: int = 128
    context_layers: int = 2

    # --- PPO fine-tuning ---
    ppo_clip_eps: float = 0.2
    ppo_updates: int = 10
    ppo_minibatch: int = 256
    ppo_seq_len: int = 128
    ppo_prioritized_segments: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # --- Reward weights ---
    reward_health_delta: float = 1.0     # damage dealt – damage taken per step
    reward_round_win: float = 10.0       # bonus for winning a round
    reward_match_win: float = 50.0       # bonus for winning the match
    reward_remaining_health: float = 2.0 # per-point of remaining health when winning a round
    reward_idle_penalty: float = -0.01   # small penalty for no inputs

    # --- Fightcade FBNeo Lua bridge ---
    # emulator/bot_bridge.lua reads this file once per frame and applies P1 input.
    fbneo_input_path: Path = field(
        default_factory=lambda: Path.home() / "fightcade-emu" / "bot_input.txt")
    # bot_bridge.lua writes game state here after each frame.
    fbneo_state_path: Path = field(
        default_factory=lambda: Path.home() / "fightcade-emu" / "bot_state.txt")

    # --- Paths ---
    data_dir: Path = field(default_factory=lambda: Path("data"))
    replay_dir: Path = field(default_factory=lambda: Path("data/replays"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    ckpt_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    # --- Reproducibility ---
    seed: int = 0

    # --- Device ---
    device: str = "cuda"

    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        for d in [self.data_dir, self.replay_dir, self.processed_dir,
                  self.ckpt_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)
