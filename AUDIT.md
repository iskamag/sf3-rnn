# Codebase Audit

This file tracks the current bloat, broken paths, and optimization issues found in the active `st-rnn` code. The ignored Fightcade and SF3 archive files are not treated as problems here.

## Critical Runtime Breakages

- `scripts/run_bot.py`, `scripts/train_rl.py`, and `scripts/collect_data.py` reference missing `Config` fields: `frame_size`, `flow_norm`, `p1_health_roi`, `p2_health_roi`, `p1_stun_roi`, and `p2_stun_roi`. These paths crash before meaningful work.
- Runtime action decoding uses batched tensors incorrectly. `model.step()` returns movement logits as `(1, 9)` and button logits as `(1, 6)`, but callers iterate or compare them as if they were unbatched vectors.
- With `use_audio=True`, startup can call `SF3RNN.forward()` before `AudioBuffer` is ready. That passes `None` for audio and raises immediately.
- `requirements.txt` omits runtime dependencies used by the code, including `mss` and the intended input-injection backend.

## Fightcade FBNeo Direction

- The project should interface with Fightcade FBNeo through Lua scripting, not RetroArch.
- `env/sf3_env.py` currently contains a large unused `RetroArchEnv` stub with black-frame output and no input implementation. This should be removed unless there is a concrete reason to keep it.
- `MssScreenEnv._send_input()` is also a placeholder. Input should be implemented through the Fightcade/FBNeo Lua bridge instead of desktop keyboard stubs where possible.
- The replay/capture pipeline should center around `emulator/input_log.lua` and a proper Lua command/control path for live input, state reads, and synchronization.

## Training And PPO Issues

- `train/ppo.py` passes `T + 1` values into `compute_returns()`, which then adds them to `T` advantages. This produces a shape mismatch.
- PPO reuses the same forward graph across multiple optimizer updates. After the first backward pass, later updates are invalid unless the graph is recomputed for each update or the update loop is redesigned.
- PPO collection double-steps the environment with neutral input during priming, which skips frames and doubles capture/input work in those branches.
- `RewardConfig()` is recreated inside collection instead of using `Config` reward weights, so configured reward values are ignored.

## Dataset And IO Bottlenecks

- `ReplayDataset` builds every stride-1 sliding window and opens a compressed `.npz` in every `__getitem__`. Adjacent samples overlap heavily, so training spends excessive CPU on decompression and memory copies.
- Processed data is saved with `np.savez_compressed()`. This saves disk but is a poor fit for random-access training over large optical-flow tensors.
- Loading each sequence copies flow, action, health, stun, and audio arrays even when most adjacent samples reuse the same frames.
- A better path is uncompressed `.npy`/memmap shards, per-file caching, or coarser segment sampling with explicit epoch reshuffling.

## Model Bloat And Mismatched Intent

- `SF3RNN.forward()` loops over timesteps in Python so `MatchContext` can be injected every frame. This prevents efficient sequence-level LSTM execution and adds autograd overhead.
- The advertised persistent BC match context is not actually persisted. `forward()` returns the final context but does not assign `self._context`; only `step()` does that. `BCTrainer.detach_context()` therefore does nothing useful during BC.
- `ReplayDataset` shuffles segments globally, so even if context were persisted, batch order would not preserve match chronology.
- The extra `MatchContext` machinery should either be made real and measured or removed for a simpler vectorized LSTM path.

## Audio Alignment

- Audio features use 256 samples per frame at 16 kHz, which is 62.5 Hz, not 60 Hz. The drift is about 2.4 seconds per minute against video/input frames.
- The README says the mismatch is negligible, but it is not negligible for behavior cloning alignment.
- Audio extraction should use frame timestamps or accumulate fractional samples per video frame instead of fixed 256-sample chunks.

## Repository Hygiene

- `.git` was much larger than tracked source because of unreachable loose objects, including large blobs from prior churn. Clean with object pruning/GC when those objects are no longer needed for recovery.
- The untracked file `erver shutdown).` appears to be accidental `less` help output and should not be committed.
- Ignored local archives and captured data are acceptable workspace artifacts and should remain untracked.

## Suggested Cleanup Order

1. Add the missing config fields or remove the broken runtime/RL entry points until the values are defined.
2. Delete `RetroArchEnv` and design the live backend around Fightcade FBNeo Lua scripting.
3. Fix tensor shape handling in action decoding and sampling.
4. Fix PPO returns and recompute model forward passes per PPO update.
5. Replace compressed per-sample `.npz` loading with cacheable or memory-mapped training data.
6. Either remove `MatchContext` or make persistence real and benchmark the cost.
7. Fix audio/video alignment using timestamps or fractional sample windows.
