# Codebase Audit

This tracks the current state of the active `st-rnn` code after the cleanup pass. Ignored local Fightcade/SF3 archives and captured data are workspace cache artifacts, not source bloat.

## Current State

- Runtime targets Fightcade FBNeo through Lua, not RetroArch.
- `env/sf3_env.py` captures the game window with `mss` and writes CPS3 input bitfields to `cfg.fbneo_input_path`.
- `emulator/bot_bridge.lua` reads `bot_input.txt`, applies P1 input through FBNeo Lua, and logs state in the same format as `input_log.lua`.
- Replay capture still uses `emulator/input_log.lua`, which is appropriate for passive replay logging.
- Model flow input is aspect-preserving `160x120`, not square and not power-of-two for its own sake.
- Raw replay capture is cached as `640x480` lossless H.264 to avoid preprocessing from macroblocked 128px video while avoiding FFV1 decode cost.
- Processed BC data uses mmap-friendly shard directories by default.
- Shards store flow as clipped uint8 (`flow_u8.npy`) and audio features as float16 (`audio_f16.npy`). Legacy compressed `.npz` remains readable.
- Audio keeps the 256-sample FFT window, but offline and real-time feature hops follow the 60 FPS video timeline to avoid drift.
- `MatchContext` is opt-in. The default path uses zero context features and parallel minGRU-pytorch sequence execution.
- PPO uses recurrent chunks with detached chunk-start hidden states, honors `ppo_minibatch`, and can prioritize chunks by mean absolute advantage.

## Remaining Risks

- HUD ROIs in `config.py` are calibration defaults. They need verification against the actual capture crop.
- The Lua input bridge uses a command file. It is simple and robust, but latency/jitter should be measured; tighter IPC may be needed later.
- `run_bot.py` assumes a running Fightcade/FBNeo instance already loaded with `bot_bridge.lua`. There is no dedicated live-launch script yet.
- `requirements.txt` covers Python packages, but the real stack also depends on system packages: PipeWire, GStreamer, Wine, gamescope, Fightcade/FBNeo, and portal/DBus components.
- PPO chunking is better than full-episode backprop, but it is still a Python/PyTorch implementation. It does not get PufferLib-style CUDA graph/static allocator/kernel benefits.
- minGRU-pytorch is now the recurrent core. It still needs benchmarking against the previous LSTM baseline for BC throughput and validation loss.
- Capture manifests are not implemented. Reproducibility currently depends on keeping local raw capture caches or manually tracking Fightcade challenge IDs.

## Next Work

1. Add a manifest format for Fightcade challenge IDs and cache status.
2. Add a live-launch script that copies `bot_bridge.lua`, clears `bot_input.txt`, starts Fightcade/FBNeo, and records enough metadata for synchronization.
3. Calibrate HUD ROIs on real `640x480` captures and document the crop assumptions.
4. Add dependency-backed smoke tests for shard loading, `SF3RNN.step()`, `AudioBuffer`, and the Lua bridge input contract.
5. Benchmark minGRU-pytorch vs the previous LSTM baseline on BC throughput and validation loss.
6. Profile PPO collection/update time after real end-to-end Lua/Fightcade testing.
