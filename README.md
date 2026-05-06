# st-rnn — minGRU for Street Fighter III: 3rd Strike

CNN + minGRU policy that plays SF3 through computer vision (optical flow),
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
│  Farneback OF (N×2-ch, H×W)      MFCC+energy  (N×15-d) │
│      ↓                                       ↓              │
│  FlowCNN → N×256-d                     pool → AudioMLP 64-d │
│      ↓                                       ↓              │
│      └─── pool ──── 256-d ────────────────┘                │
│      └───── + MatchContext (2×128-d GRU) ──┘               │
│      └───── + ego_player (1-d flag)  ───┘                  │
│      └───── + prev_actions (temporal conv) ─┘              │
│                        ↓                                    │
│  2-layer minGRU (hidden=512) ← 1 step per N frames          │
│                        ↓                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  hidden + pos-embed₀ → movement₀ buttons₀ value₀    │    │
│  │  hidden + pos-embed₁ → movement₁ buttons₁ value₁    │    │
│  │  ...                        ...                      │    │
│  │  hidden + pos-embedₙ → movementₙ buttonsₙ valueₙ    │    │
│  └─────────────────────────────────────────────────────┘    │
│                        ↓                                    │
│  OutputDelayBuffer — motor lag 4 f                          │
│                        ↓                                    │
│  game input                                                 │
└─────────────────────────────────────────────────────────────┘
```

**Stride-N recurrence**: The minGRU runs once per `temporal_stride` (N) frames
(default N=4, ~15 Hz). Each step ingests N flow frames and produces N
consecutive action sets, matching the ~67 ms human decision cadence while
vision stays at 60 Hz.

**Autoregressive prev-actions**: The model feeds its own previous N actions
(distribution over movement + buttons) into the next recurrence step via a
learned embedding, pooled across the N frames.  During BC/PPO training this
uses teacher-forced ground-truth, shifted by one step; at inference it uses
the model's own softmax/sigmoid.

**Optical flow**
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
buffer.  The `MatchContext` (2×128-d stacked GRU) accumulates an opponent
profile across the full match from optical flow alone.

### Model detail

- **CNN**: 4 conv layers (stride-2) → adaptive pool 4×4 → FC 256-d; processes pooled flow frames
- **Audio**: 15-d per-frame features (13 MFCC + RMS + centroid) → per-frame MLP → max-pool over N frames to emphasize onset events
- **minGRU**: 2 layers, hidden=512, dropout=0.1, runs once per N=4 frames
- **minGRU input**: 256 (pooled flow) + 64 (audio) + 128 (context) + 1 (ego) + 64 (prev-actions) = 513-d (with audio)
- **Action decoder**: learned sub-frame position embedding `(N, 512)` broadcast onto hidden → shared MLP → N independent (movement, buttons, value) sets per step
- **MatchContext**: stacked GRUCell (2 layers, 128-d), updated at each stride and persisted across BC segments and full PPO episodes
- **Action space**: multi-head — categorical (9 directions) + multi-label (6 buttons)
- **Value head**: merged into the per-slot action decoder; scalar per frame
- **ego_player**: single scalar (0=P1, 1=P2) concatenated into minGRU input
- **Prev-actions**: per-frame MLP + causal depthwise 1D conv over N frames → preserves intra-stride button-press ordering
- **Per-frame variation**: comes from a learned sub-frame position embedding, which cuts CNN FLOPs by N×

## Project layout

```
st-rnn/
├── config.py                  # All hyperparameters
├── model/
│   ├── action_space.py        # Direction / Button enums, CPS3 bitfield codec
│   ├── rnn.py                 # FlowCNN + AudioMLP + MatchContext → minGRU
│   ├── agent.py               # CleanRL-style agent: stride-N rollout + teacher-forced evaluation
│   ├── context.py             # MatchContext GRU (opponent profiling)
│   ├── reaction_buffer.py     # Visual sensory-lag frame buffer
│   ├── audio_buffer.py        # Audio sensory-lag buffer
│   └── output_buffer.py       # Motor-lag action buffer
├── vision/
│   ├── optical_flow.py        # Farneback flow computation, preprocess, clip
│   ├── audio_features.py      # MFCC + RMS + spectral centroid (256-pt FFT)
│   └── game_state.py          # Audio-template match detection for BC shard splitting
├── emulator/
│   ├── input_log.lua          # FBNeo Lua script — logs P1/P2 inputs, health, stun
│   ├── recorder.py            # Capture replays — video, audio, inputs (multi-worker, timer-based)
│   └── bot_bridge.lua         # FBNeo Lua script — read bot inputs from file
├── data/
│   ├── dataset.py             # PyTorch Dataset (shard directories)
│   └── shard.py               # Flow quantisation helpers
├── train/
│   ├── bc.py                  # Behaviour cloning trainer
│   └── ppo.py                 # PPO trainer (CleanRL-style, multi-epoch, value clipping)
└── scripts/
    ├── train.sh               # Unified CLI wrapper
    ├── collect_data.py        # Capture → shard pipeline
    ├── train_bc.py            # BC training entry point
    ├── train_rl.py            # PPO fine-tuning entry point
    ├── view_capture.py        # Visualise captures with match-boundary overlay
    └── run_bot.py             # Real-time inference
```

## Setup

```bash
./scripts/setup_env.sh
source .venv/bin/activate
```

The setup script creates `.venv/` and installs CPU PyTorch.

This repo currently runs on CPU. Optical flow runs through OpenCV
(Farneback, ROI-based on changed regions).

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

Also required: `wine`, `gamescope`, `gstreamer`, `pipewire`, and `xdotool`.

### 2. Capture replays

Pick replays from [fightcade.com](https://www.fightcade.com/game/sfiii3nr1).
Copy their URLs — e.g. `https://replay.fightcade.com/fbneo/sfiii3nr1/1777806845400-9194`.
The last segment (`1777806845400-9194`) is the challenge ID.  Each replay also has
a known duration in seconds (shown on the replay page).

**Batch capture** — create a file with one `<challenge_id> <length_seconds>` per
line (lines starting with `#` are ignored), then fan out across parallel workers:

```
# replay_ids.txt
1777806845400-9194 180
1777895668498-5828 240
1778123456789-1234 195
```

```bash
./emulator/recorder.py replay_ids.txt                # 4 workers (default)
./emulator/recorder.py replay_ids.txt --workers 8    # 8 workers
```

The `length_seconds` is the known replay duration (available on the Fightcade
replay page).  Already-captured replays (with `input_log.txt` in their output
directory) are skipped automatically.

What happens per replay:
1. A per-worker Wine prefix is cloned from `~/.fightcade-wine`
2. `gamescope` launches FBNeo with the quark stream URL and `input_log.lua`
3. The video PipeWire node ID is read directly from gamescope's stderr
4. The Xwayland display number is also read from gamescope's stderr
5. After the replay starts, FBNeo is toggled to fullscreen via `xdotool`
6. GStreamer captures video (160×120 GRAY8, 60 fps) and audio (16 kHz mono U8)
7. The Lua script writes per-frame inputs and game state to `input_log.txt`
8. A countdown timer runs for `length_seconds` from the moment the replay begins
9. When time expires the emulator is killed (process-group SIGKILL + wineserver -k), log is saved

Each worker reuses its Wine prefix across replays.  Workers are isolated via
`start_new_session` and per-worker Wine prefixes (`~/.fightcade-wine-replay-{0,1,…}`).

Audio is captured from the wine process's PipeWire stream, matched by
descendant PID.  Video comes from gamescope's PipeWire node.

Output in `data/captures/<challenge_id>/`:
```
video.raw       — 160×120 GRAY8 raw frame bytes, 60 fps
audio.wav       — 8-bit mono 16 kHz PCM
input_log.txt   — "frame p1bits p2bits p1hp p2hp p1stun p2stun timer p1wins p2wins" per line
```

Repeat for many replays — different players, characters, skill levels.
Aim for at least 50–100 matches for BC training to be useful.

For a 1-second match-start cue sample:
```bash
ffmpeg -y -i hmm_16k.wav -t 1.0 -c:a copy hmm_16k_1s.wav
```

To inspect a capture (match boundaries, optical flow):
```bash
./scripts/train.sh view data/captures/1777895668498-5828 \
    --match-template hmm_16k_1s.wav
```

### 3. Process into training data

Converts the captures into mmap-friendly shard directories the dataset can
 read. Each shard contains one match (audio-template starts paired with
 logged match-end wins, with `match_start_offset_frames` skipping ahead from
 the cue to the actual gameplay start).

```bash
./scripts/train.sh collect \
    --capture-dir data/captures \
    --output-dir data/processed \
    --match-template hmm_16k_1s.wav
```

Output: `data/processed/<id>_match00/`, `..._match01/`, etc.

Contents of each shard:

| Array          | Shape                 | Description                          |
|----------------|-----------------------|--------------------------------------|
| `flow_u8.npy`       | (T-1, 2, H, W) | ROI-based Farneback flow, clipped to [-1, 1], uint8-quantized; resolution = `flow_width`×`flow_height` |
| `audio_f16.npy`     | (T, 15)        | MFCC + energy features, float16    |
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
./scripts/train.sh bc
```

Resume from a checkpoint:

```bash
./scripts/train.sh bc-resume
```

Loss: `CrossEntropy(movement) + BCEWithLogits(buttons)` summed over the
full minGRU sequence.  The MatchContext vector persists across segments
from the same match (detached at boundaries) so the model learns to
accumulate an opponent profile.

Checkpoints saved to `checkpoints/bc_epochXXX.pt`.
Logs at `logs/bc/` (view with `tensorboard --logdir logs`).

### 5. PPO fine-tuning

Fine-tune the BC policy with self-play or environment interaction.
Reward is computed from CPS3 memory values (health, wins) read by the
Lua bridge each frame.  The audio pipeline is managed inside the trainer;
motor lag (`OutputDelayBuffer`) decouples the stored (action, logprob,
value) from the executed action so reward alignment is correct.

```bash
./scripts/train.sh ppo
```

Pass a specific BC checkpoint when needed:

```bash
./scripts/train.sh ppo checkpoints/bc_epoch099.pt
```

**Reward function** (read from CPS3 memory via Lua bridge):

```
reward = health_delta * (damage_dealt − damage_taken)
       + round_win_bonus
       + match_win_bonus
       − idle_penalty
```

The update replays each trajectory in stride-N windows with full hidden-state
propagation, so gradients flow through time across the minGRU and MatchContext.
Multiple PPO epochs per batch with per-trajectory shuffling.  Value loss uses
SB3/CleanRL-style clipping.

Checkpoints saved to `checkpoints/ppo_stepNNNNNN.pt`.
Logs at `logs/ppo/`.

Start TensorBoard with:

```bash
./scripts/train.sh tensorboard
```

### 6. Run the bot

```bash
python scripts/run_bot.py --ckpt checkpoints/ppo_step001000.pt
```

Manual start/stop — the bot captures frames and audio, runs them through
the full latency pipeline (sensory buffers, stride-N accumulation, motor
buffer), and injects inputs. Start it when already in-game.

Override latency:

```bash
python scripts/run_bot.py --ckpt ... --sensory-lag 8 --motor-lag 4
```

## Configuration

All tunables in `config.py`:

| Parameter              | Default | Meaning                              |
|------------------------|---------|--------------------------------------|
| `sensory_lag_frames`   | 8       | Visual sensory delay (frames)        |
| `audio_lag_frames`     | 3       | Audio sensory delay (faster pathway) |
| `motor_lag_frames`     | 4       | Motor/output delay (frames)          |
| `total_delay_frames`   | 12      | Visual total ≈ 200 ms @ 60 fps       |
| `temporal_stride`      | 4       | Frames per recurrence step (N)       |
| `use_audio`            | True    | Enable audio pipeline                |
| `frame_width`          | 160     | Raw video frame width (capture)      |
| `frame_height`         | 120     | Raw video frame height (capture)     |
| `flow_width`           | 160     | Optical-flow output width (model input) |
| `flow_height`          | 120     | Optical-flow output height (model input) |
| `cnn_channels`         | 16      | FlowCNN base channels (8/12/16/24/32) |
| `compile_model`        | True    | torch.compile the model              |
| `context_dim`          | 128     | MatchContext opponent-profile size   |
| `action_embed_dim`     | 64      | Prev-action embedding dimension      |
| `feature_dim`          | 256     | Visual feature dimension             |
| `audio_dim`            | 64      | Audio embedding dimension            |
| `mingru_hidden`        | 512     | minGRU hidden size                   |
| `mingru_layers`        | 2       | minGRU depth                         |
| `seq_len`              | 96      | BC training segment length (frames)  |
| `lr`                   | 1e-4    | AdamW learning rate (non-Muon params)|
| `muon_lr`              | 2e-3    | Muon learning rate (matrix params)   |
| `ppo_updates`          | 10      | PPO epochs per batch of trajectories |
| `ppo_clip_eps`         | 0.2     | PPO clip epsilon                     |
| `gamma`                | 0.99    | Discount factor                      |
| `gae_lambda`           | 0.95    | GAE trace decay                      |
| `ent_coef`             | 0.01    | Entropy bonus coefficient            |
| `vf_coef`              | 0.5     | Value loss coefficient               |
| `max_grad_norm`        | 0.5     | Global gradient clipping             |

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

- The optical-flow CNN is a small custom architecture sized for 2-channel flow fields.
- BC trains on diverse player data — the model learns general SF3
  competency across players.
- During offline capture health/stun is read from CPS3 memory by the Lua
  script; training data is ground-truth accurate.
- The MatchContext GRU gives the minGRU a persistent 128-d opponent profile
  that survives BC segment boundaries and full PPO episodes.
- Audio features use a manual 256-point DCT-II (no scipy dependency).
