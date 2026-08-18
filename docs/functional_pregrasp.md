# Functional pregrasp adjustment

This branch studies how the existing xArm7 + XHand stack can move a tool resting on a table
into a state from which an intent-specific functional grasp can succeed.  The body, 21-D
action (`arm7 + CrossDex9 + distal5`), 115-D observation and FlashSAC trainer stay unchanged.

## Phase-1 benchmark

The versioned manifest is `configs/functional_pregrasp/tools_v1.json`.  It fixes ten OakInk
training instances and one or more UniDexFPM `use` targets per instance:

1. hammer (`hammers_2`)
2. flashlight (`flashlight_1`)
3. marker pen (`marker_pen8`)
4. screwdriver (`screwdriver_1`)
5. toothbrush (`toothbrush_maya3`)
6. knife (`knife_s138`)
7. frying pan (`frying_pan_0`)
8. lotion pump (`lotion_pump_s104`)
9. power drill (`035_power_drill`)
10. trigger sprayer (`trigger_sprayer_s101`)

The first version trains one fixed-intent specialist per object.  The current environment has
one global mesh, mass, support model and grasp region, and the observation has no object ID;
mixing all ten in one vector environment would therefore be a hidden-task bug, not a valid
generalist experiment.  A later generalist must add a multi-asset spawner and an object/intent
descriptor before joint training is enabled.

## Success contract

A preparation state is successful only when all of these conditions hold for the confirmation
window:

- the object is in an intent target set, using its discrete or continuous symmetry;
- the real mesh is table-supported and adjacent-pose linear/angular speeds are low;
- the wrist, palm, proximal links and pads remain safely above the table;
- the XHand reaches the object-specific handoff region;
- a frozen downstream close/lift probe succeeds.

The last gate is the certification criterion.  A run that uses only pose and geometric handoff
gates is an infrastructure canary and must not be reported as functional-grasp success.
UniDexFPM targets use a ShadowHand palm frame and joints, so each target still needs XHand frame
calibration, retargeting, collision rejection and a physics close/lift/load probe.

The code fixes five issues inherited from the hammer nudge option: reward and observation now
share the same goal buffer at observation columns 63:70; heading error is symmetry-aware;
settling checks angular as well as linear velocity; and tipping can be an intermediate action
rather than an immediate failure.  Settling uses adjacent control-frame pose deltas because
Isaac Lab disables rigid-body sleep for contact reporters and Fabric then exposes noisy non-zero
instantaneous velocities even after the pose stops changing.  Table height is independent scene
truth, not inferred from the object's configured rest pose.

## Rigid Phase 1

The pump, drill and sprayer are locked and merged into one rigid body.  Phase 1 measures only
whether the hand reaches a functional-grasp preparation state; it does not claim trigger or
pump actuation.  The knife requires a thin-collision and blade-keepout audit.  Cylindrical tools
declare axial symmetry explicitly.

## Data setup

Raw OakInk/UniDexFPM data is external and must not be committed.  Set its extracted root and
validate every path and target before simulation:

```bash
export XHAND_UNIDEXFPM_ROOT=/disk2/xhand_datasets/unidexfpm
python scripts/functional_pregrasp/validate_manifest.py
```

The manifest records the downloaded archive checksums and data redistribution restriction.
UniDexFPM code licensing does not override the licenses of its upstream meshes and poses.
The trainer binds every `Functional-Pregrasp-*` task to its declared object/intent and writes
the task, object, intent and manifest hash into `checkpoint_final/experiment.json`.  Restoring
optimizer/reward-normalizer state requires an exact identity match; intentional cross-object
network transfer must use `--weights_only`.

## Training ladder

Run each object through `8-env smoke -> 128-env physics check -> 512-env pilot -> 1024-env
formal`.  The first new-object task is `Functional-Pregrasp-Flashlight-Direct-v0`; its mesh,
metric scale, grip band, directed functional axis and continuous axial-roll symmetry are
calibrated, and its stable side-rest was frozen by a PhysX drop-settle probe.
The existing hammer task is used first as an infrastructure regression because its downstream
close/lift chain already exists.

The flashlight seed is now PhysX-certified: all 8 deterministic attitudes were table-supported
and pose-stable after the 300-step (6 s) probe.  This certifies reset physics, not the downstream
functional grasp; XHand retargeting plus the frozen close/lift probe remains the final gate.

The first 128-environment canary completed 400 interactions (51,200 environment steps and 383
gradient updates) with finite metrics and a saved checkpoint.  Its zero success rate is not a
benchmark result: the short run verifies collection, replay, updates, reset/safety contracts and
checkpoint provenance before the intent-specific XHand handoff gate is enabled.
