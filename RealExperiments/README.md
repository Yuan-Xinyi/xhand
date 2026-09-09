# Pick Cube FoundationPose To Isaac

This folder keeps the current sim2real test entry point for the trained
`Pick-Cube-Direct-v0` policy.

The intended flow is:

1. Capture one D405 RGB-D frame.
2. Select the cube ROI like `/home/lqin/disk2/FoundationPose/cube/live_demo.py`.
3. Run FoundationPose registration and short tracking.
4. Save the detected `camera_T_cube` pose.
5. Launch `scripts/rl_games/play.py` and inject the cube pose into Isaac.

Run:

```bash
/home/lqin/miniconda3/envs/env_isaaclab/bin/python RealExperiments/foundationpose_then_play.py
```

The script saves:

- init image: `/tmp/foundationpose_init_frame.png`
- cube pose: `/tmp/foundationpose_cube_pose.npy`

If OpenCV has no GUI support, the script opens a local browser ROI page. If the
browser does not open automatically, copy the printed `http://127.0.0.1:...`
URL into a browser.

To only test FoundationPose pose extraction without launching Isaac:

```bash
/home/lqin/miniconda3/envs/env_isaaclab/bin/python RealExperiments/foundationpose_then_play.py \
  --no_play \
  --preview
```

To pass a known ROI directly:

```bash
/home/lqin/miniconda3/envs/env_isaaclab/bin/python RealExperiments/foundationpose_then_play.py \
  --roi 220 140 110 95
```

## Real Hardware

`foundationpose_then_real.py` uses the same FoundationPose ROI flow, then runs the
trained RL actor and streams joint targets to xArm7 + XHand through `one`.

Always test dry-run first. This loads the checkpoint, builds the 89-D policy
observation, and prints commands without connecting to hardware:

```bash
/home/lqin/miniconda3/envs/env_isaaclab/bin/python RealExperiments/foundationpose_then_real.py \
  --no_pose_capture \
  --pose_npy /tmp/foundationpose_cube_pose.npy \
  --steps 10
```

To capture pose and dry-run policy without hardware:

```bash
/home/lqin/miniconda3/envs/env_isaaclab/bin/python RealExperiments/foundationpose_then_real.py \
  --steps 50
```

To connect hardware but not move it:

```bash
/home/lqin/miniconda3/envs/env_isaaclab/bin/python RealExperiments/foundationpose_then_real.py \
  --real \
  --xarm-ip 192.168.1.205 \
  --xhand-port /dev/ttyUSB0 \
  --steps 10
```

Real execution requires both `--real` and `--execute` and asks for confirmation
before moving to the RL start pose and before streaming policy commands:

```bash
/home/lqin/miniconda3/envs/env_isaaclab/bin/python RealExperiments/foundationpose_then_real.py \
  --real \
  --execute \
  --xarm-ip 192.168.1.205 \
  --xhand-port /dev/ttyUSB0
```

When `--real` is used from `env_isaaclab`, the script first runs FoundationPose
there, saves the cube pose, then automatically re-executes itself with
`/home/lqin/miniconda3/envs/one/bin/python` for xArm/XHand hardware IO. The `one`
environment already carries the required `xarm` SDK and `pyserial` packages.

---

# Repose Cube FoundationPose Pipeline (sim2real)

Entry point: `foundationpose_repose_real.py` — connects the trained
`Xhand-Repose-Cube-OpenAI-LSTM-Direct-v0` policy (34-D obs LSTM) to live
FoundationPose cube tracking and the real XHand.

Two processes over UDP (127.0.0.1:9877), teleop-style:

- `foundationpose_repose_tracker.py` (env_isaaclab): D435 -> FoundationPose
  register (popup ROI) + continuous tracking -> camera_T_cube stream.
  Spawned automatically; can also run standalone.
- `foundationpose_repose_real.py` (one env, auto re-exec on --real): 20 Hz
  control loop. XHand joint targets -> URDF FK (self-contained, verified 0.00 mm
  vs Isaac) -> fingertip positions; cube pose -> T_base_cam calib -> env frame
  (real palm aligned to the sim palm pose (0,0,0.5)/(0.7071,-0.7071,0,0));
  obs(34) -> LSTM -> absolute joint targets (moving average 0.3, sim contract).

The xArm7 is never commanded — it only HOLDS the wrist palm-up. The script
prints a gravity-tilt warning if the held palm orientation deviates from sim.

Progressive bring-up:

```bash
# 1. offline smoke test (no camera, no robot; any env with torch)
python RealExperiments/foundationpose_repose_real.py --pose-source synthetic --steps 200

# 2. camera-in-the-loop dry-run (tracker + policy, prints commands, no robot)
python RealExperiments/foundationpose_repose_real.py --steps 400

# 3. hardware connected, still not moving
python RealExperiments/foundationpose_repose_real.py --real --steps 200

# 4. full run (confirms before each motion; cube placed in palm by hand)
python RealExperiments/foundationpose_repose_real.py --real --execute
```

Safety: per-cycle joint step clamp (`--max-hand-step`), joint-limit saturation,
pose-staleness hold/abort (`--max-pose-age` / `--abort-pose-age`), cube-fall
stop (0.24 m), success tolerance `--success-tol` (default 0.4 rad = trained).

Auto-center (default ON, disable with `--no-auto-center`): at startup the cube
rests in the open palm at a known sim position (0, 0.101, 0.551), so the mean
measured position defines a constant offset that cancels the calibration
TRANSLATION error entirely — no camera re-calibration needed after small camera
moves. The extrinsic ROTATION error is NOT cancelled (it biases the perceived
cube orientation); the July-2026 `camera_extrinsics.yaml` rotation is still
used, and a >15 cm offset triggers a warning that rotation is probably off too.
Real-run order: hand home -> place cube -> tracker ROI -> auto-center -> go.

Verification tools (already run once, all green):

- `repose_probe_dump.py` (env_isaaclab): dumps sim contract constants + FK
  reference to `repose_probe.npz`.
- `verify_repose_obs.py` (one env): obs layout (6e-8) + URDF FK (0.00 mm).
- `verify_repose_policy_sim.py` (env_isaaclab): drives the sim env with the
  DEPLOYMENT policy stack — 13 goal successes / 30 s, 0 falls.
