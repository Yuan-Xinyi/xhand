# FlashSAC integration

The algorithm implementation is pinned as the `third_party/FlashSAC` git submodule at
commit `87edc9061150ae9e962dd84e6544e27a1554b3ab` (Holiday Robotics, MIT license).

Do not run the upstream `uv sync --extra isaaclab` inside the existing Isaac Lab
environment.  This workspace currently uses a newer local Isaac Lab checkout and a
different PyTorch build.  The scripts in this directory import only the upstream
`flash_rl` algorithm modules and keep environment stepping in the existing conda
environment.

The integration deliberately differs from the upstream Isaac Lab wrapper in four ways:

1. Only the 115-dimensional `policy` observation is stored.  The task currently exposes
   an identical `critic` observation, so concatenating the two wastes memory and breaks
   the declared observation shape.
2. Observations, actions, and replay transitions stay as CUDA tensors.
3. The adapter captures the true post-action observation immediately before Isaac Lab
   auto-resets a completed environment.  Genuine terminations do not bootstrap; time-limit
   truncations bootstrap from this captured terminal observation, never from the next
   episode's reset state.
4. Training and strict evaluation never interleave through a stale transition.  A reset
   invalidates the previous `(observation, action)` boundary.

These contracts must remain covered by the local tests before any long training run.

## Additional deliberate departures (full-fidelity audit, 2026-07-24)

A line-by-line audit against the pinned upstream confirmed the update mathematics,
networks, buffer, reward normalizer and schedulers are the unmodified upstream code, and
the four deviations above are implemented exactly as stated.  The audit also catalogued
these further local departures, previously documented only in docstrings/CLI help:

1. **Grouped exploration noise (active in every run).**  `_sample_grouped_actions`
   replaces upstream `_sample_flashsac_actions`: independent arm/token/residual zeta
   clocks (mu/max 1.0/64, 1.25/32, 1.5/16; scale 1 / 0.5 / 0.35), per-env noise reset at
   episode end, and a latch-gated hand-noise multiplier (default 0.2 once the grasp latch
   is up).  The deterministic evaluation path is identical to upstream.
2. **Demo machinery (opt-in via `--demo`).**  Fixed 25%-demo mixed replay
   (`FixedFractionDemoReplay`) replacing the agent's buffer, a `demo_bc_rehearsal` actor
   step after each SAC actor update, and priming the reward normalizer's `G_r_max` from
   the max |n-step demo reward|.
3. **Checkpoint warm-start semantics (opt-in via `--checkpoint`).**  Deterministic
   BC-actor collection during warmup, and a critic-only burn-in mode
   (`--critic_burnin_updates`) with the actor update gated off.
4. **Default hyperparameters that differ from upstream:** `n_step=3`,
   `updates_per_interaction=2.0`, LR warmup = 5% of the decay budget (upstream 1 / 1 /
   ~0).  All are CLI-visible.
5. **Close-phase-only training (opt-in via `--close_option`).**  Env-side episode
   contract override; the algorithm is untouched.

A from-scratch run without `--demo`/`--checkpoint` therefore differs from upstream only
by item 1 and the item-4 defaults.
