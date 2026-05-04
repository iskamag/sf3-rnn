# Codebase Audit

Tracks the current state of `st-rnn` after the stride-N recurrence and
prev-action feedback refactor.

## Current State

- Runtime targets Fightcade FBNeo through Lua, not RetroArch.
- `env/sf3_env.py` captures the game window with `mss`, crops to 4:3, and
  resizes to `384×224` (SF3 native resolution). Inputs are written as CPS3
  bitfields to `cfg.fbneo_input_path`.
- `emulator/bot_bridge.lua` reads `bot_input.txt`, applies P1 input through
  FBNeo Lua, and logs state in the same format as `input_log.lua`.
- Replay capture uses `emulator/input_log.lua` — all health, stun, timer,
  and win data is read directly from CPS3 emulator memory, not from HUD
  extraction. The logged format is `"frame p1bits p2bits p1hp p2hp p1stun
  p2stun timer p1wins p2wins"` per line.
- Capture output: `video.raw` (160×120 GRAY8, 60 fps) + `audio.wav` +
  `input_log.txt`. No HUD extraction during offline processing.
- `scripts/collect_data.py` converts captures into mmap-friendly shard
  directories. Health/stun/wins come from the CPS3-memory log, not from
  video frames.
- HUD ROIs in `config.py` (`p1_health_roi`, `p2_health_roi`, etc.) are at
  **384×224 native resolution**. They are used exclusively at runtime:
  - `has_health_bar()` for menu/character-select detection (`run_bot.py`)
  - `extract_game_state()` for PPO reward computation (`train_rl.py`)
- `vision/game_state.py` has dead `read_super()` code and unused `p1_super` /
  `p2_super` fields — they are never populated.
- Model flow input is 160×120, computed from the 384×224 frames via
  `ReactionBuffer`.
- Shards store flow as clipped uint8 (`flow_u8.npy`) and audio features as
  float16 (`audio_f16.npy`).
- Audio keeps a 256-sample FFT window with per-frame feature hops following
  the 60 FPS video timeline (no drift).
- **Stride-N recurrence**: minGRU runs once per `temporal_stride` (N=4)
  frames (~15 Hz), producing N action sets per step. Vision still streams
  at 60 Hz; per-frame flow features are passed through the action decoder.
- **Autoregressive prev-actions**: The previous step's N action distributions
  are embedded, pooled, and fed into the recurrent input. BC/PPO training
  uses teacher-forced ground-truth shifted by one step; inference uses the
  model's own softmax/sigmoid.
- `MatchContext` is always enabled. BC trains ordered match chunks, carrying
  minGRU hidden state and context across chunks so the model builds an
  opponent profile over the full match.
- PPO uses recurrent chunks with detached chunk-start hidden states, honors
  `ppo_minibatch`, and can prioritize chunks by mean absolute advantage.
- `seq_len` = 96 frames (24 recurrence steps at N=4).

## Remaining Risks

- HUD ROIs in `config.py` are calibration defaults at 384×224 native
  resolution. They should be verified against the actual Fightcade capture
  window crop.
- The Lua input bridge uses a command file (`bot_input.txt`). It is simple
  and robust, but latency/jitter should be measured; tighter IPC may be
  needed later.
- `run_bot.py` assumes a running Fightcade/FBNeo instance already loaded
  with `bot_bridge.lua`. There is no dedicated live-launch script.
- `requirements.txt` covers Python packages, but the real stack also depends
  on system packages: PipeWire, GStreamer, Wine, gamescope, Fightcade/FBNeo,
  and portal/DBus components.
- PPO chunking is in Python/PyTorch — no CUDA graph / static allocator /
  kernel fusion benefits. Collection and update throughput are untested
  end-to-end.
- The stride-N action decoder (shared MLP per slot) is simple and fast, but
  the N actions within a step are independent — no autoregressive dependency
  across slots within the same recurrence step.
- `ReactionBuffer` maxlen was increased to accommodate multi-frame flow
  collection; edge-case frame-timing scenarios are not stress-tested.
- `vision/game_state.py` has dead super-meter extraction code and unused
  `GameState.p1_super` / `p2_super` fields.
- Capture manifests are not implemented. Reproducibility depends on keeping
  local raw capture caches or manually tracking Fightcade challenge IDs.
- `read_super()` in `game_state.py` is unreachable dead code — it should
  either be wired in or removed.

## Next Work

1. Add a manifest format for Fightcade challenge IDs and cache status.
2. Add a live-launch script that copies `bot_bridge.lua`, clears
   `bot_input.txt`, starts Fightcade/FBNeo, and records enough metadata
   for synchronization.
3. Calibrate HUD ROIs against real 384×224 captures and document the crop
   assumptions.
4. Add dependency-backed smoke tests for shard loading, `SF3RNN.step()`,
   `AudioBuffer`, and the Lua bridge input contract.
5. Measure whether intra-step autoregression (action₀ → action₁ → …)
   improves BC validation loss compared to the current slot-independent
   decoder.
6. Remove or wire in the dead `read_super()` super-meter extraction code.
7. Profile PPO collection/update time after real end-to-end Lua/Fightcade
   testing.
