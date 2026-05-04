# Codebase Audit

This file tracks bloat, broken paths, and optimization issues found in the active `st-rnn` code. The ignored Fightcade and SF3 archive files are not treated as problems here.

## Addressed

- Added the missing runtime config values used by capture, inference, and PPO: frame size, flow normalization, and HUD ROIs.
- Removed the unused RetroArch backend and replaced placeholder input sending with a Fightcade FBNeo Lua command-file bridge.
- Added `emulator/bot_bridge.lua`, which reads CPS3 bitfields from `bot_input.txt`, applies P1 inputs through FBNeo Lua, and logs state.
- Fixed single-step inference to return unbatched action/value tensors, avoiding ambiguous tensor comparisons in runtime callers.
- Made audio priming safe when `use_audio=True` by substituting zero audio features until the real buffer is ready.
- Kept the 256-sample FFT window for audio, but aligned offline and real-time feature hops to the 60 FPS video timeline to avoid drift.
- Fixed PPO's return/value shape mismatch and recomputed forward passes per PPO update instead of reusing a stale autograd graph.
- Removed duplicate environment steps during PPO priming and wired reward collection to `Config` reward weights.
- Reduced BC dataset overhead with configurable segment stride and per-worker decompressed `.npz` caching.
- Made `MatchContext` opt-in so the default model path can use vectorized LSTM execution.

## Critical Runtime Breakages

- `requirements.txt` may still be incomplete for the exact desktop/audio stack used in production. `mss` is listed now, but PipeWire, DBus, GObject, GStreamer, Wine, gamescope, and FBNeo are system dependencies.
- HUD ROI defaults are now present, but they are calibration guesses and should be verified against the actual Fightcade capture crop.
- The Lua bridge uses a simple command file. This is enough to remove the placeholder, but a tighter IPC mechanism may be needed if file polling adds measurable latency or jitter.

## Fightcade FBNeo Direction

- The project interfaces with Fightcade FBNeo through Lua scripting, not RetroArch.
- `env/sf3_env.py` now writes CPS3 bitfields to the configured Lua command file.
- `emulator/bot_bridge.lua` applies live P1 input and logs game state in the same format as `input_log.lua`.
- Remaining work: launch/capture scripts should be updated to copy/use `bot_bridge.lua` for live bot runs, while replay capture can continue using `input_log.lua`.

## Training And PPO Issues

- PPO no longer passes `T + 1` values into `compute_returns()` or reuses the same autograd graph for multiple updates.
- PPO still updates over the full episode as one sequence. This preserves LSTM ordering but does not use `ppo_minibatch`; long episodes may need truncated sequence minibatches.
- PPO still depends on the live Fightcade/Lua environment being calibrated and tested end-to-end.

## Dataset And IO Bottlenecks

- `ReplayDataset` now supports coarser segment sampling and caches decompressed `.npz` arrays per worker.
- Processed data is still saved with `np.savez_compressed()`. This saves disk but remains suboptimal for large random-access optical-flow tensors.
- Remaining better path: uncompressed `.npy`/memmap shards or a consolidated dataset format with explicit sequence indexes.

## Model Bloat And Mismatched Intent

- `MatchContext` is now opt-in. The default path keeps the context input shape but fills it with zeros and runs the LSTM over the full sequence in one call.
- If match profiling is re-enabled, BC segment ordering must preserve chronology and the context persistence behavior should be tested explicitly.
- Keeping the `MatchContext` module preserves checkpoint compatibility for now, but it can be removed later if no trained checkpoints need it.

## Audio Alignment

- Audio still uses a 256-sample FFT window, which keeps the FFT efficient.
- Offline and real-time feature extraction now place those windows on the 60 FPS video timeline instead of advancing by fixed 256-sample hops.
- Remaining work: verify the effective audio lag against captured Fightcade audio/video timestamps.

## Repository Hygiene

- `.git` was much larger than tracked source because of unreachable loose objects, including large blobs from prior churn. It was pruned and packed.
- Ignored local archives and captured data are acceptable workspace artifacts and should remain untracked.

## Suggested Cleanup Order

1. Update launch/capture scripts to use `bot_bridge.lua` for live bot runs.
2. Calibrate and test HUD ROIs against actual Fightcade captures.
3. Replace compressed `.npz` training data with memory-mapped shards if caching/stride are not enough.
4. Add dependency-backed smoke tests for model step, dataset loading, audio buffering, and the Lua bridge contract.
5. Decide whether `MatchContext` is worth keeping before training checkpoints that depend on it.
