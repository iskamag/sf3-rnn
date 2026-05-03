# st-rnn — LSTM for Street Fighter III: 3rd Strike

CNN + LSTM policy that plays SF3 through computer vision (optical flow),
constrained to human reaction time (~200 ms).  Trained via behaviour
cloning on Fightcade replays, then fine-tuned with PPO.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  screen capture                         audio capture       │
│      ↓                                       ↓              │
│  Greyscale resize (128 × 128)         16 kHz mono           │
│      ↓                                       ↓              │
│  ReactionBuffer — visual lag 8 f       AudioBuffer — 3 f    │
│      ↓                                       ↓              │
│  Farneback OF  (2-ch, 128×128)        MFCC+energy  (15-d)  │
│      ↓                                       ↓              │
│  FlowCNN → 256-d                       AudioMLP → 64-d      │
│      ↓                                       ↓              │
│      └───────── concat ──── 320-d ──────────┘              │
│                        ↓                                    │
│  2-layer LSTM (hidden=512)  ←  hidden state persistent      │
│                        ↓                                    │
│  ┌──────────┬──────────┬──────────┐                         │
│  │ movement │ buttons  │  value   │                         │
│  │ (9-way)  │ (6-bit)  │  scalar  │                         │
│  └──────────┴──────────┴──────────┘                         │
│                        ↓                                    │
│  OutputDelayBuffer — motor lag 4 f                          │
│                        ↓                                    │
│  game input                                                 │
└─────────────────────────────────────────────────────────────┘
```

**Why optical flow instead of raw pixels?**
- Invariant to static backgrounds and character costumes
- Motion patterns encode hit-stun, block-stun, knockdown, and spacing
- 2-channel input is cheaper than RGB and more robust

**Why two latency buffers?**
Real human reaction time splits into:

| Pipeline stage              | Latency   | Frames  | Buffer              |
|-----------------------------|-----------|---------|----------------------|
| Visual (retina + V1)        | ≈ 130 ms  | 8 frm   | `ReactionBuffer`    |
| Audio  (cochlea + cortex)   | ≈ 50 ms   | 3 frm   | `AudioBuffer`       |
| Motor  (nerves + muscle)    | ≈ 67 ms   | 4 frm   | `OutputDelayBuffer` |
| **Total visual reaction**   | 200 ms    | 12 frm  | —                   |

Audio reaches consciousness ~3× faster than vision, so it gets its own
shorter-delay buffer.  The LSTM fuses both streams — visual flow (256-d)
and acoustic features (64-d) — at each timestep.

### Model detail

- **CNN**: 4 conv layers (stride-2) → adaptive pool 4×4 → FC 256-d
- **LSTM**: 2 layers, hidden=512, dropout=0.1
- **Audio**: 15-d per-frame features (13 MFCC + RMS + centroid) → MLP → 64-d, fused with visual by concatenation
- **Action space**: multi-head — categorical (9 directions) + multi-label (6 buttons)

## Project layout

```
st-rnn/
├── config.py                  # All hyperparameters
├── model/
│   ├── action_space.py        # Direction / Button enums, CPS3 bitfield codec
│   ├── rnn.py                 # FlowCNN + AudioMLP → LSTM + value head
│   ├── reaction_buffer.py     # Visual sensory-lag frame buffer
│   ├── audio_buffer.py        # Audio sensory-lag buffer (shorter delay)
│   └── output_buffer.py       # Motor-lag action buffer
├── vision/
│   ├── optical_flow.py        # Farneback flow computation
│   ├── audio_features.py      # MFCC + RMS + spectral centroid per frame
│   └── game_state.py          # Health / stun / meter extraction + match splitting
├── data/
│   └── dataset.py             # PyTorch Dataset (replay .npz files)
├── train/
│   ├── bc.py                  # Behaviour cloning trainer
│   ├── ppo.py                 # PPO trainer + rollout buffer
│   └── reward.py              # Health-delta reward + GAE
├── replay/
│   └── parser.py              # Fightcade .fc2 replay parser
├── env/
│   ├── sf3_env.py             # Screen-capture / RetroArch environment
│   └── audio_capture.py       # Background audio capture via sounddevice
│   └── parser.py              # Fightcade .fc2 replay parser
├── env/
│   └── sf3_env.py             # Screen-capture / RetroArch environment
├── scripts/
│   ├── collect_data.py        # .fc2 → .npz pipeline
│   ├── train_bc.py            # Run BC training
│   ├── train_rl.py            # Run PPO fine-tuning
│   └── run_bot.py             # Real-time inference
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

GPU recommended.  Optical flow runs on CPU via OpenCV (Farneback); model
runs on CUDA.

## 1. Collect training data

### Option A — Fightcade replay files

1. Obtain `.fc2` replays from Fightcade's replay library or your own matches.
   Place them in `data/replays/`.
2. Optionally capture frames during replay playback (e.g. with OBS or `mss`).
3. Run the parser:

   ```bash
   python scripts/collect_data.py --replay-dir data/replays --output-dir data/processed
   ```

   This produces one `.npz` **per match** (character select, intros, and
   victory screens are trimmed):
   - `flows` — (T, 2, 128, 128) optical-flow frames
   - `movement` — (T,) direction indices 0–8
   - `buttons` — (T, 6) button states {0, 1}
   - `health_p1`, `health_p2` — (T,) 0–1

   **Match splitting**: If pre-captured frames are available (`--frames-dir`),
   the script detects health bars via HSV thresholding and trims replays
   at character-select / menu boundaries.  Without frames it falls back to
   a button-activity heuristic (menus have sparse inputs; gameplay has
   frequent presses).

### Option B — Live capture with inputs

Use `env/sf3_env.py` to capture frames while logging inputs from
Fightcade's input display overlay (or via the emulator's Lua API).

## 2. Behaviour cloning

Train the model to imitate human players:

```bash
python scripts/train_bc.py
```

The dataset applies `total_delay_frames` (default 12) offset between
input flows and target actions, baking the reaction-time constraint
into the supervision signal.

Resume from a checkpoint:

```bash
python scripts/train_bc.py --resume checkpoints/bc_epoch042.pt
```

Loss: `CrossEntropy(movement) + BCEWithLogits(buttons)` summed over the
full LSTM sequence.  Checkpoints saved to `checkpoints/bc_epochXXX.pt`.

## 3. PPO fine-tuning

Fine-tune with self-play or environment interaction:

```bash
python scripts/train_rl.py --bc-ckpt checkpoints/bc_epoch099.pt
```

The rollout uses the dual-buffer latency pipeline.  Each logged transition
stores the flow from `total_delay_frames` ago, so the PPO update sees
correctly aligned `(state, action, reward)` tuples.

### Reward function

```
reward = health_delta * (damage_dealt - damage_taken)
       + round_win_bonus
       + match_win_bonus
       - idle_penalty
```

All components are computable from screen pixels (health bars, round
indicators).  No game-internal memory access needed.

## 4. Run the bot

```bash
python scripts/run_bot.py --ckpt checkpoints/ppo_step001000.pt
```

Requires a running SF3 instance (Fightcade or RetroArch).  Configure the
screen-capture region in `env/sf3_env.py`.

Override latency parameters:

```bash
python scripts/run_bot.py --ckpt ... --sensory-lag 8 --motor-lag 4
```

### Character select & menu navigation

Between matches the bot watches for health-bar absence (`menu_detect_frames`
config parameter).  When health bars disappear for long enough, it runs the
`char_select_seq` from `config.py` — a hardcoded sequence of direction +
button + hold-time tuples.  The default selects Ryu (cursor start position
+ LP to confirm).

```python
# config.py
char_select_seq: tuple = (
    ("NEUTRAL",  (),        60),    # wait for screen to settle
    ("NEUTRAL",  ("LP",),   10),    # confirm default character (Ryu)
)
```

To pick a different character, extend the sequence with directional moves
before the confirm step.  Direction/button names must match the enums in
`model/action_space.py`.  Disable auto-navigation with `--no-menu-nav`.

After navigation the LSTM hidden state and output buffer are reset, and
the reaction buffer re-primes with gameplay frames.

## Configuration

All tunables live in `config.py`.  Key ones:

| Parameter              | Default | Meaning                              |
|------------------------|---------|--------------------------------------|
| `sensory_lag_frames`   | 8       | Visual sensory delay (frames)        |
| `audio_lag_frames`     | 3       | Audio sensory delay (faster pathway) |
| `motor_lag_frames`     | 4       | Motor/output delay (frames)          |
| `total_delay_frames`   | 12      | Visual total ≈ 200 ms @ 60 fps       |
| `use_audio`            | True    | Enable audio capture & features      |
| `audio_dim`            | 64      | Internal audio embedding size        |
| `feature_dim`          | 256     | Visual feature dimension             |
| `lstm_hidden`          | 512     | LSTM hidden size                     |
| `lstm_layers`          | 2       | LSTM depth                           |
| `seq_len`              | 128     | BC training segment length           |
| `lr`                   | 1e-4    | Learning rate                        |
| `ppo_clip_eps`         | 0.2     | PPO clip epsilon                     |

Health-bar ROIs are native-resolution (384 × 224) bounding boxes.
Calibrate with your specific capture setup.

## Fightcade replay format

Fightcade 2 `.fc2` files are zlib-compressed JSON + CPS3 input bitfields.
The parser (`replay/parser.py`) extracts:

- Game metadata (players, characters)
- Per-frame input bitfields (CPS3 format: bits 0–3 = directions, 4–9 = buttons)

If parsing fails, convert to JSON first with
[fc2json](https://github.com/eezstreet/fc2json) and pass the `.json` path.

### CPS3 input bitfield

```
bit 0 — Up          bit 4 — LP (jab)
bit 1 — Down        bit 5 — MP (strong)
bit 2 — Left        bit 6 — HP (fierce)
bit 3 — Right       bit 7 — LK (short)
                    bit 8 — MK (forward)
                    bit 9 — HK (roundhouse)
bit 10 — Start
```

## Notes

- The optical-flow CNN is a small custom architecture (not a pretrained
  backbone) because flow fields are 2-channel and cheap to process.
- Behaviour cloning trains on agnostic player data — the model learns
  general SF3 competency, not to mimic a single player.
- PPO reward is fully screen-observable.  No emulator memory hooks.
- The ReactionBuffer forces the model to act on stale information;
  the OutputDelayBuffer forces actions to wait before execution.
  Together they impose the full ~200 ms human reaction pipeline.
