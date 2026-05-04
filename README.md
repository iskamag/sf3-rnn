# st-rnn — LSTM for Street Fighter III: 3rd Strike

CNN + LSTM policy that plays SF3 through computer vision (optical flow),
constrained to human reaction time (~200 ms).  Trained via behaviour
cloning on Fightcade replays, then fine-tuned with PPO.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  screen capture                         audio capture       │
│      ↓                                       ↓              │
│  Greyscale resize (160 × 120)         16 kHz mono           │
│      ↓                                       ↓              │
│  ReactionBuffer — visual lag 8 f       AudioBuffer — 3 f    │
│      ↓                                       ↓              │
│  Farneback OF  (2-ch, 160×120)        MFCC+energy  (15-d)  │
│      ↓                                       ↓              │
│  FlowCNN → 256-d                       AudioMLP → 64-d      │
│      ↓                                       ↓              │
│      └───── concat ──── 385-d ──────────┘                  │
│      └───── + MatchContext (64-d GRU) ───┘                  │
│      └───── + ego_player (1-d flag)  ───┘                  │
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

**Latency model** — split to match biological pipelines:

| Pipeline stage              | Latency   | Frames  | Buffer              |
|-----------------------------|-----------|---------|----------------------|
| Visual (retina + V1)        | ≈ 130 ms  | 8 frm   | `ReactionBuffer`    |
| Audio  (cochlea + cortex)   | ≈ 50 ms   | 3 frm   | `AudioBuffer`       |
| Motor  (nerves + muscle)    | ≈ 67 ms   | 4 frm   | `OutputDelayBuffer` |
| **Total visual reaction**   | 200 ms    | 12 frm  | —                   |

Audio reaches consciousness ~3× faster than vision; it gets its own shorter
buffer.  The `MatchContext` (64-d GRU) accumulates an opponent profile across
the full match from optical flow alone.

### Model detail

- **CNN**: 4 conv layers (stride-2) → adaptive pool 4×4 → FC 256-d
- **Audio**: 15-d per-frame features (13 MFCC + RMS + centroid) → MLP → 64-d
- **LSTM**: 2 layers, hidden=512, dropout=0.1, input = 256 + 64 + 64 + 1 = 385-d
- **MatchContext**: GRUCell(LSTM_out → 64-d), persisted across BC segments
- **Action space**: multi-head — categorical (9 directions) + multi-label (6 buttons)
- **Value head**: scalar, for PPO
- **ego_player**: single scalar (0=P1, 1=P2) concatenated into LSTM input so the
  model can play either side; dataset randomly picks a player per sample

## Project layout

```
st-rnn/
├── config.py                  # All hyperparameters
├── model/
│   ├── action_space.py        # Direction / Button enums, CPS3 bitfield codec
│   ├── rnn.py                 # FlowCNN + AudioMLP + MatchContext → LSTM
│   ├── context.py             # MatchContext GRU (opponent profiling)
│   ├── reaction_buffer.py     # Visual sensory-lag frame buffer
│   ├── audio_buffer.py        # Audio sensory-lag buffer
│   └── output_buffer.py       # Motor-lag action buffer
├── vision/
│   ├── optical_flow.py        # Farneback flow computation
│   ├── audio_features.py      # MFCC + RMS + spectral centroid (256-pt FFT)
│   └── game_state.py          # Match splitting from health signal
├── emulator/
│   ├── input_log.lua          # FBNeo Lua script — logs P1/P2 inputs, health, stun
│   ├── capture_replay.sh      # Launch + capture video/audio/inputs
│   └── start_screencast.py    # XDG Desktop Portal screencast helper
├── data/
│   └── dataset.py             # PyTorch Dataset (.npz files)
├── train/
│   ├── bc.py                  # Behaviour cloning trainer
│   ├── ppo.py                 # PPO trainer + rollout buffer
│   └── reward.py              # Health-delta reward + GAE
└── scripts/
    ├── collect_data.py        # Capture → .npz pipeline
    ├── train_bc.py            # BC training entry point
    ├── train_rl.py            # PPO fine-tuning entry point
    └── run_bot.py             # Real-time inference
```

## Setup

```bash
pip install -r requirements.txt
```

GPU recommended.  Optical flow runs on CPU via OpenCV (Farneback); model
runs on CUDA.

## Guide: from replay to trained model

### 1. Prerequisites

You need the Fightcade emulator and the SF3 ROM.  The Fightcade Linux
tarball (`Fightcade-linux-latest.tar.gz`) contains `emulator/fbneo/`.

```bash
tar xf Fightcade-linux-latest.tar.gz
mkdir -p ~/fightcade-emu
cp -r Fightcade/emulator/fbneo/* ~/fightcade-emu/
cp sfiii3nr1.zip ~/fightcade-emu/ROMs/
```

Also required: `wine`, `gamescope`, `ffmpeg` (with VAAPI), `pipewire`,
`pw-v4l2`.

### 2. Capture a replay

Pick a replay from [fightcade.com](https://www.fightcade.com/game/sfiii3nr1).
Copy its URL — e.g. `https://replay.fightcade.com/fbneo/sfiii3nr1/1777806845400-9194`.
The last segment (`1777806845400-9194`) is the challenge ID.

```bash
./emulator/capture_replay.sh 1777806845400-9194
```

What happens:
1. `gamescope` launches the emulator with `input_log.lua`
2. A screencast dialog appears — **accept it**
3. GStreamer captures the gamescope window as native-resolution lossless H.264
   video + 8-bit 16 kHz mono WAV audio
4. The Lua script writes per-frame P1/P2 CPS3 bitfields, health, hitstun,
   timer, and wins to `input_log.txt`

Output in `data/captures/1777806845400-9194/`:
```
video.mkv       — 640×480 lossless H.264, 60 fps
audio.wav       — 8-bit mono 16 kHz PCM
input_log.txt   — "frame p1bits p2bits p1hp p2hp p1stun p2stun timer p1wins p2wins" per line
```

Repeat for many replays — different players, characters, skill levels.
Aim for at least 50–100 matches for BC training to be useful.

### 3. Process into training data

Converts the captures into mmap-friendly shard directories the dataset can
read. Each shard contains one match (character select / intros trimmed
automatically via health-bar transitions).

```bash
python scripts/collect_data.py \
    --capture-dir data/captures \
    --output-dir data/processed
```

Output: `data/processed/<id>_match00/`, `..._match01/`, etc.

Contents of each shard:

| Array          | Shape              | Description                          |
|----------------|--------------------|--------------------------------------|
| `flow_u8.npy`       | (T-1, 2, 120, 160) | Farneback optical flow, clipped to [-1, 1] and uint8-quantized |
| `audio_f16.npy`     | (T, 15)             | MFCC + energy features, float16    |
| `movement_u8.npy`   | (T,)                | P1 direction index 0–8             |
| `buttons_u8.npy`    | (T, 6)              | P1 button states {0, 1}            |
| `p2_movement_u8.npy`| (T,)                | P2 direction index (raw, no mirror)|
| `p2_buttons_u8.npy` | (T, 6)              | P2 button states                   |
| `health_p1_u8.npy`  | (T,)                | P1 health 0–255                    |
| `health_p2_u8.npy`  | (T,)                | P2 health 0–255                    |
| `stun_p1_u8.npy`    | (T,)                | P1 hitstun {0, 1}                  |
| `stun_p2_u8.npy`    | (T,)                | P2 hitstun {0, 1}                  |
| `meta.json`         | —                   | Quantization and source metadata   |

### 4. Behaviour cloning

Trains the model to imitate the captured players.  The reaction-time
delay is baked into the dataset: the model learns to map optical flow
at time *t* to the action the human executed at time *t + total_delay*.

```bash
python scripts/train_bc.py
```

Resume from a checkpoint:

```bash
python scripts/train_bc.py --resume checkpoints/bc_epoch042.pt
```

Loss: `CrossEntropy(movement) + BCEWithLogits(buttons)` summed over the
full LSTM sequence.  The MatchContext vector persists across segments
from the same match (detached at boundaries) so the model learns to
accumulate an opponent profile.

Checkpoints saved to `checkpoints/bc_epochXXX.pt`.
Logs at `logs/bc/` (view with `tensorboard --logdir logs`).

### 5. PPO fine-tuning

Fine-tune the BC policy with self-play or environment interaction.
The reward is fully screen-observable — no emulator memory hooks needed.

```bash
python scripts/train_rl.py --bc-ckpt checkpoints/bc_epoch099.pt
```

**Reward function:**

```
reward = health_delta * (damage_dealt − damage_taken)
       + round_win_bonus
       + match_win_bonus
       − idle_penalty
```

The dual-buffer latency pipeline is active during rollout: visual lag
8 frames, audio lag 3 frames, motor lag 4 frames.  Logged transitions
use the total-delay flow so the PPO update sees correctly aligned
`(state, action, reward)` tuples.

Checkpoints saved to `checkpoints/ppo_stepNNNNNN.pt`.
Logs at `logs/ppo/`.

### 6. Run the bot

```bash
python scripts/run_bot.py --ckpt checkpoints/ppo_step001000.pt
```

Requires a running SF3 instance.  The bot captures frames and audio,
delays them through the reaction buffers, runs the LSTM, delays the
output through the motor buffer, and injects inputs.

Override latency:

```bash
python scripts/run_bot.py --ckpt ... --sensory-lag 8 --motor-lag 4
```

Between matches the bot detects character-select by watching for
health-bar absence and navigates via `char_select_seq` in `config.py`.
The LSTM hidden state and match context are reset at each new match.

## Configuration

All tunables in `config.py`:

| Parameter              | Default | Meaning                              |
|------------------------|---------|--------------------------------------|
| `sensory_lag_frames`   | 8       | Visual sensory delay (frames)        |
| `audio_lag_frames`     | 3       | Audio sensory delay (faster pathway) |
| `motor_lag_frames`     | 4       | Motor/output delay (frames)          |
| `total_delay_frames`   | 12      | Visual total ≈ 200 ms @ 60 fps       |
| `use_audio`            | True    | Enable audio pipeline                |
| `frame_width`          | 160     | Model optical-flow input width       |
| `frame_height`         | 120     | Model optical-flow input height      |
| `context_dim`          | 64      | MatchContext opponent-profile size   |
| `feature_dim`          | 256     | Visual feature dimension             |
| `audio_dim`            | 64      | Audio embedding dimension            |
| `lstm_hidden`          | 512     | LSTM hidden size                     |
| `lstm_layers`          | 2       | LSTM depth                           |
| `seq_len`              | 128     | BC training segment length           |
| `lr`                   | 1e-4    | Learning rate                        |
| `ppo_clip_eps`         | 0.2     | PPO clip epsilon                     |

## CPS3 input bitfield

The Lua script logs P1 and P2 inputs in CPS3 format (same encoding for both):

```
bit 0 — Up          bit 4 — LP (jab)
bit 1 — Down        bit 5 — MP (strong)
bit 2 — Left        bit 6 — HP (fierce)
bit 3 — Right       bit 7 — LK (short)
                    bit 8 — MK (forward)
                    bit 9 — HK (roundhouse)
bit 10 — Start
```

Hitstun is tracked via CPS3 combo-counter memory addresses: P1 is in stun
when P2's combo counter is non-zero (P2 is hitting P1), and vice versa.

## Notes

- The optical-flow CNN is a small custom architecture (not a pretrained
  backbone) — flow fields are 2-channel and cheap to process.
- BC trains on diverse player data — the model learns general SF3
  competency, not to mimic a single player.
- PPO reward is fully screen-observable; the Lua script reads health
  directly from CPS3 memory during capture, so no HUD extraction needed.
- The MatchContext GRU gives the LSTM a persistent 64-d opponent profile
  that survives BC segment boundaries and full PPO episodes.
- Audio features use a manual 256-point DCT (no scipy dependency).
