# pick_hammer_token — YCB 048 claw hammer object swap of the pick_tool chain

`Pick-Hammer-Token-Direct-v0` runs the full multi-stage pick_tool_token machinery (nudge →
close/latch → carry → in-hand reorientation, FlashSAC-trained) against a new object: the
google_16k scan of the YCB `048_hammer` (a full-size 33 cm claw hammer), added under
`source/xhand_inhand/xhand_inhand/assets/048_hammer_google_16k/`.

No env-class fork: the task registers `PickToolTokenEnv` with `PickHammerTokenEnvCfg`
(a thin object-swap subclass of `PickToolTokenEnvCfg`).  To make that possible the env's
true-clearance hull / handle-polygon source mesh became configurable
(`object_mesh_obj` / `object_mesh_scale` cfg fields, defaulting to the old tool mesh), and
`scripts/flashsac/train.py`, `carry_demo.py`, `capture_selfplay_boundaries.py`,
`diag_pregrasp_score.py` grew a `--task` flag (default unchanged).

## Object geometry (measured from the mesh, object local frame)

| quantity | value |
|---|---|
| extents | 0.182 × 0.333 × 0.033 m, lying flat, mesh bottom at z ≈ 0 |
| handle (grip) centroid | (0.001, −0.090, 0.016) |
| handle axis (PCA) | (−0.373, 0.928, 0.009), butt at t=−0.10, head junction at t=+0.13 |
| grip cross-section radius | 1.31–1.75 cm (rubber grip; the old tool's "handle" was a 2 mm web) |
| strike face center | (−0.125, 0.053, 0.016) |
| claw center | (−0.025, 0.130, 0.014) |
| graspable band | t ∈ [−0.05, +0.05] (central 10 cm of grip) |
| grasp keypoints | 4 surface points at ~90° spacing on the t=0 grip cross-section |
| in-hand functional point | (−0.032, −0.010, 0.027) — top of the handle neck (choke-up band) |
| in-hand head axis | claw → strike face = (−0.794, −0.608, 0.015); pointing it at −z = striking attitude |

## Rest pose (measured in-sim, 2026-07-29)

Gentle place at identity, flat drop from 0.35 m, and two tilted drops from 0.50 m all
settle to **identity orientation** (residual quat < 0.023) at **z = −0.0004** with the
converted USD.  Unlike the first tool there is no metastable perch: the authored scan pose
is the single true rest, so `HAMMER_REST_Z=-0.0004`, `HAMMER_REST_QUAT=identity`.

## Deliberate recipe carry-overs

* **Mass authored at 0.15 kg** (real YCB 048 weighs 0.665 kg): every force gate
  (`contact_force_thr` 0.2 N, `safe_contact_force` 20 N, `grasp_bonus_max_force` 30 N,
  tactile shields 25/30/60 N) and the lift dynamics recipe were calibrated on the 0.15 kg
  tool.  Raise mass only together with a recalibration pass.
* Same convex decomposition (64 hulls, vertex limit 64, shrink-wrap), same solver
  iteration counts, same 21-D action / 115-D observation layout — checkpoints are
  architecture-compatible for warm starts, though the object geometry shift means behavior
  must be re-trained/fine-tuned per stage.
* `carry_goal_z_margin` raised 0.16 → 0.22 (half-diagonal 0.19 m vs the tool's 0.13).
* `reset_min_hand_object_dist` 0.08 → 0.10 (longer object at reset).

## Training entry points

```bash
# FlashSAC (per-stage, same flags as the tool chain):
python scripts/flashsac/train.py --headless --task Pick-Hammer-Token-Direct-v0 ...

# boundary capture / demos / diagnostics:
python scripts/flashsac/capture_selfplay_boundaries.py --task Pick-Hammer-Token-Direct-v0 ...
python scripts/flashsac/carry_demo.py --task Pick-Hammer-Token-Direct-v0 ...
python scripts/flashsac/nudge_grasp_chain.py --task Pick-Hammer-Token-Direct-v0 ...
```

Recommended stage order mirrors the tool journey: nudge_grasp (home → latch) →
carry / arm-locked gaiting → in-hand functional point, reusing the curriculum
boundary-capture + anneal recipe.  The functional point / head axis defaults were
auto-derived from the mesh; refine them with `tools/functional_point/build_annotator.py`
and paste into both `pick_hammer_token_env_cfg.py` and `functional_point.json`.
