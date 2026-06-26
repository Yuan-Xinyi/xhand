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
  --xarm-ip 192.168.1.232 \
  --xhand-port /dev/ttyUSB0 \
  --steps 10
```

Real execution requires both `--real` and `--execute` and asks for confirmation
before moving to the RL start pose and before streaming policy commands:

```bash
/home/lqin/miniconda3/envs/env_isaaclab/bin/python RealExperiments/foundationpose_then_real.py \
  --real \
  --execute \
  --xarm-ip 192.168.1.232 \
  --xhand-port /dev/ttyUSB0
```

When `--real` is used from `env_isaaclab`, the script first runs FoundationPose
there, saves the cube pose, then automatically re-executes itself with
`/home/lqin/miniconda3/envs/one/bin/python` for xArm/XHand hardware IO. The `one`
environment already carries the required `xarm` SDK and `pyserial` packages.
