# Codebase audit: st-rnn

## Bugs

1. **`train/ppo.py:_process_trajectory` — per-trajectory backward instead of minibatch accumulation**
   Each trajectory calls `loss.backward()` + `optim.step()` independently (line 358-362). Most PPO implementations accumulate gradients over a minibatch. With short trajectories (common in fighting games), this gives very high variance updates. The `update()` loop also averages metrics over trajectory count, not frame count, weighting short and long trajectories equally.

2. **`scripts/collect_data.py:296-305` — thread pool race on `flows` array**
   `_compute_flow_pair` writes into `flows[i]` by index via `as_completed`. Each future knows its index, so the final array is correct — but the code mixes a side-effect pattern with a thread pool, which is fragile if anyone changes the indexing. The pool also has no error handling; a single failed flow computation silently drops that frame.

## Dead code

1. **`model/reaction_buffer.py:get_flow_at(offset)`** — defined but never called anywhere.

2. **`vision/optical_flow.py:frame_to_flow_stack()`** — defined but never called anywhere.

3. **`data/dataset.py:71` — `jitter_end`** stored in every index tuple but destructured as `_` in `__getitem__`. Originally intended for random segment jittering, never implemented.

## Inconsistencies

1. **Gradient norm**: BC trainer clips at **1.0** (hardcoded in `train/bc.py:224,229`) but the config says `max_grad_norm: float = 0.5` and PPO uses `cfg.max_grad_norm`. BC ignores its own config.

2. **`torch.compile` config is BC-only**: `config.py:compile_model=True`, but only `BCTrainer` compiles. PPO trainer doesn't compile, so the config flag is misleading for RL training.

3. **`model/agent.py:get_value()` hardcodes P1 ego**: `ego = torch.zeros(1, 1, ...)` — if this were ever called for P2 evaluation, value estimates would be wrong.

## Design / performance

1. **`model/rnn.py:_encode_prev_actions_fallback` reaches into model internals**: `self._action_embed[-1].out_features` assumes the last layer is `nn.Linear`. Architecture change = silent break.

2. **`train/ppo.py:253` also reaches into internals**: `audio_encoder.net[0].in_features` — if the audio encoder architecture changes, PPO silently breaks.

3. **`model/audio_buffer.py:68` — no Hann window in the buffer**. `extract_features` applies a Hann window, but the raw `chunk` extracted in `add_samples` is zero-padded without windowing before being sent to `extract_features`. This means the per-frame features from online (`AudioBuffer`) and offline (`collect_data.py`) pipelines apply the window at different stages. The spectrogram will differ slightly.

4. **`model/audio_buffer.py` — no reset method**. If used across episodes (e.g., match resets in PPO), residual samples from the previous round bleed into the next episode's features. The same issue applies to `ReactionBuffer`.

5. **No seed management**. No `torch.manual_seed`, no `np.random.seed` anywhere. Zero reproducibility.

6. **Zero tests**. No unit tests, no integration tests, no smoke tests. The only validation path is running the full pipeline.
