from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Game ---
    fps: int = 60
    native_width: int = 384
    native_height: int = 224

    # --- Reaction-time constraint ---
    # Total human visual reaction ≈ 200–250 ms → 12–15 frames @ 60 fps.
    # The model fuses sensory + motor latency into one total delay:
    #   sensory lag (display + retina + V1)  ~100 ms  ~ 6 frames
    #   motor   lag (nerves + muscle + USB)  ~100 ms  ~ 6 frames
    #   ─────────────────────────────────────────────
    #   total_delay ≈ 200 ms ≈ 12 frames
    #
    # You can split these explicitly with sensory_lag_frames +
    # motor_lag_frames = total_delay.  The ReactionBuffer implements
    # sensory lag; OutputDelayBuffer (optional) implements motor lag.
    # For training data the target offset equals total_delay.
    sensory_lag_frames: int = 6
    motor_lag_frames: int = 6

    @property
    def total_delay_frames(self) -> int:
        return self.sensory_lag_frames + self.motor_lag_frames

    # --- Vision ---
    frame_size: int = 128       # resize greyscale frames to square
    flow_norm: float = 20.0     # max expected displacement for normalisation

    # Health-bar bounding boxes in NATIVE resolution (384 x 224)
    # Values below are approximations — calibrate with your setup.
    p1_health_roi: tuple = (96, 32, 151, 5)
    p2_health_roi: tuple = (138, 32, 151, 5)
    p1_stun_roi: tuple = (96, 40, 151, 3)
    p2_stun_roi: tuple = (138, 40, 151, 3)
    super_bar_y: int = 212       # y coordinate of super-meter bar
    timer_roi: tuple = (180, 18, 25, 12)

    # --- Model ---
    feature_dim: int = 256
    lstm_hidden: int = 512
    lstm_layers: int = 2
    lstm_dropout: float = 0.1
    num_directions: int = 9      # neutral + 8-way
    num_buttons: int = 6         # LP, MP, HP, LK, MK, HK

    # --- Training ---
    batch_size: int = 32
    seq_len: int = 128           # frames per training segment
    lr: float = 1e-4
    weight_decay: float = 1e-5
    bc_epochs: int = 100
    num_workers: int = 4

    # --- PPO fine-tuning ---
    ppo_clip_eps: float = 0.2
    ppo_updates: int = 10
    ppo_minibatch: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # --- Reward weights ---
    reward_health_delta: float = 1.0     # damage dealt – damage taken per step
    reward_round_win: float = 10.0       # bonus for winning a round
    reward_match_win: float = 50.0       # bonus for winning the match
    reward_idle_penalty: float = -0.01   # small penalty for no inputs

    # --- Character select (inference only) ---
    # Frames without health bars that trigger menu mode.
    menu_detect_frames: int = 150        # 2.5 s
    # After a match ends, stall this many frames before navigating menus.
    post_match_stall: int = 30
    # Sequence of (direction_name, button_names, hold_frames).
    # Direction / button names must match the enums in model/action_space.py.
    # "NEUTRAL" + empty buttons + hold_frames = wait.
    char_select_seq: tuple = (
        ("NEUTRAL",  (),        60),    # wait for screen to settle
        ("NEUTRAL",  ("LP",),   10),    # confirm default character (Ryu)
    )

    # --- Paths ---
    data_dir: Path = field(default_factory=lambda: Path("data"))
    replay_dir: Path = field(default_factory=lambda: Path("data/replays"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    ckpt_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    # --- Device ---
    device: str = "cuda"

    def __post_init__(self):
        for d in [self.data_dir, self.replay_dir, self.processed_dir,
                  self.ckpt_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)
