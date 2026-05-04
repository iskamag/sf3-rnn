# Codebase Audit

This tracks the current state of the active `st-rnn` code after the cleanup pass. Ignored local Fightcade/SF3 archives and captured data are workspace cache artifacts, not source bloat.

## Current State

- Runtime targets Fightcade FBNeo through Lua, not RetroArch.
- `env/sf3_env.py` captures the game window with `mss` and writes CPS3 input bitfields to `cfg.fbneo_input_path`.
- `emulator/bot_bridge.lua` reads `bot_input.txt`, applies P1 input through FBNeo Lua, and logs state in the same format as `input_log.lua`.
- Replay capture still uses `emulator/input_log.lua`, which is appropriate for passive replay logging.
- Model flow input is aspect-preserving `160x120`, not square and not power-of-two for its own sake.
- Raw replay capture is currently cached as `160x120` `GRAY8` `video.raw` from the PipeWire/GStreamer capture path.
- Processed BC data uses mmap-friendly shard directories only.
- Shards store flow as clipped uint8 (`flow_u8.npy`) and audio features as float16 (`audio_f16.npy`).
- Audio keeps the 256-sample FFT window, but offline and real-time feature hops follow the 60 FPS video timeline to avoid drift.
- **Stride-N recurrence**: minGRU runs once per `temporal_stride` (N=4) frames (~15 Hz), producing N action sets per step. Vision still streams at 60 Hz; flow per-frame features are passed through the action decoder so each slot sees different visual input.
- **Autoregressive prev-actions**: The previous step's N action distributions are embedded, pooled, and fed into the recurrent input. BC/PPO training uses teacher-forced ground-truth shifted by one step; inference uses the model's own softmax/sigmoid.
- `MatchContext` is always enabled. BC trains ordered match chunks, carrying minGRU hidden state and context across chunks so the model builds an opponent profile over the full match.
- PPO uses recurrent chunks with detached chunk-start hidden states, honors `ppo_minibatch`, and can prioritize chunks by mean absolute advantage.
- `seq_len` reduced from 128 to 96 frames (24 recurrence steps at N=4).

## Remaining Risks

- HUD ROIs in `config.py` are calibration defaults. They need verification against the actual capture crop.
- The Lua input bridge uses a command file. It is simple and robust, but latency/jitter should be measured; tighter IPC may be needed later.
- `run_bot.py` assumes a running Fightcade/FBNeo instance already loaded with `bot_bridge.lua`. There is no dedicated live-launch script yet.
- `requirements.txt` covers Python packages, but the real stack also depends on system packages: PipeWire, GStreamer, Wine, gamescope, Fightcade/FBNeo, and portal/DBus components.
- PPO chunking is better than full-episode backprop, but it is still a Python/PyTorch implementation. It does not get PufferLib-style CUDA graph/static allocator/kernel benefits.
- The stride-N action decoder (shared MLP per slot) is simple and fast, but the N actions within a step are independent — no autoregressive dependency across slots within the same recurrence step. This may be suboptimal for precise button sequences.
- `ReactionBuffer` maxlen was increased to accommodate multi-frame flow collection; this has not been stress-tested with edge-case frame timing.
- Capture manifests are not implemented. Reproducibility currently depends on keeping local raw capture caches or manually tracking Fightcade challenge IDs.

## Next Work

1. Add a manifest format for Fightcade challenge IDs and cache status.
2. Add a live-launch script that copies `bot_bridge.lua`, clears `bot_input.txt`, starts Fightcade/FBNeo, and records enough metadata for synchronization.
3. Calibrate HUD ROIs on real `640x480` captures and document the crop assumptions.
4. Add dependency-backed smoke tests for shard loading, `SF3RNN.step()`, `AudioBuffer`, and the Lua bridge input contract.
5. Measure whether intra-step autoregression (action₀ → action₁ → …) improves BC validation loss compared to the current slot-independent decoder.
6. Profile PPO collection/update time after real end-to-end Lua/Fightcade testing.
