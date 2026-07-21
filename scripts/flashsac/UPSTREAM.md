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
