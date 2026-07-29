# Camera → MANO → xhand teleoperation

Real-time hand teleop: a webcam watches your hand, **WiLoR** reconstructs it as a
MANO hand, **DexPilot** retargets the fingertips onto the xhand, and the 12 joint
targets drive the XHand in Isaac Sim.

```
webcam ─▶ WiLoR (image → 21 MANO 3D keypoints) ─▶ DexPilot retarget (→ 12 xhand joints)
                                                          │  UDP :51234
                                          Isaac Sim  ◀────┘  set_joint_position_target
```

The MANO→xhand half reuses the exact DexPilot config the RL token policy was
distilled from (`tools/crossdex_retarget/configs/xhand_right_dexpilot.yml`), so a
teleoperated grasp and a policy grasp share the same hand kinematics.

## Two-process design (why)

WiLoR needs `torch<=2.5`; Isaac Lab runs `torch 2.7`. Installing WiLoR into
`env_isaaclab` would downgrade torch and break the sim. So perception and control
are **separate processes in separate conda envs**, talking over localhost UDP
(`protocol.py`, pure stdlib). This is also how real teleop deploys: a perception
node and a control node, decoupled.

| Process | Env | Deps |
|---|---|---|
| `perception_node.py` | `wilor` (py3.10, torch2.5) | WiLoR-mini, dex_retargeting, cv2 |
| `teleop_sim.py` | `env_isaaclab` (py3.11, torch2.7) | Isaac Lab |

## Run (one command)

`teleop.py` is the single entry point: it launches both halves in their own conda
envs and tears both down on Ctrl+C (or when either side exits, e.g. `q` in the
overlay window). Run it from **any** env — it only uses the stdlib:

```bash
cd tools/teleop
python teleop.py                          # sim GUI + RealSense perception with overlay
python teleop.py --no-show                # perception headless
python teleop.py --source webcam --camera 10
python teleop.py --hand left --mirror --flip
# anything not covered by the flags above:
python teleop.py --sim-args "--debug" --perc-args "--beta 2.0 --redetect-interval 6"
```

The Isaac side is never SIGKILLed on shutdown (that can corrupt the Isaac Sim
install) — worst case the launcher waits a few extra seconds for its clean exit.

## Real hardware (`real_node.py`)

Drives the physical XHand over RS-485 (`/dev/ttyUSB0` @ 3 Mbaud; protocol ported
from `one` repo branch `dex-hand`, `Yuan/dexterous_hand/xhand_con/`). Runs in the
`wilor` env (pyserial installed there). Bring-up order:

```bash
cd tools/teleop            # (conda activate wilor)
python real_node.py --ping      # 1. read-only firmware query — checks power+cable
python real_node.py --check     # 2. slow per-joint sweep — verify id<->joint mapping
python real_node.py --open      # 3. ramp to flat-open pose
python teleop.py --real         # 4. full teleop: camera -> REAL hand (one command)
```

Safety layers (always on): URDF joint-limit clamp (`--limit-margin` 0.02 rad),
per-joint speed limit (`--max-speed` 2.5 rad/s — also soft-starts), EMA smoothing
(`--smooth` 0.5), first target initialized from the hand's measured position (no
enable jump), stream watchdog (stale > `--timeout` 2 s -> hold pose, never snap).
Firmware-side: `--kp 100 --kd 10 --tor-max 300`. Start gentle on a new setup:
`python teleop.py --real --real-args "--max-speed 1.5 --tor-max 200"`.

If the hardware finger-id order ever mismatches (a `--check` sweep moves the
wrong joint), fix `HW_JOINT_NAMES` in real_node.py — one place only.

## Run (two terminals, manual)

**Terminal 1 — sim driver** (`conda activate env_isaaclab`):
```bash
cd tools/teleop
python teleop_sim.py            # live GUI; add --headless --debug to run without a window
```

**Terminal 2 — perception** (`conda activate wilor`):
```bash
cd tools/teleop
# preferred: Intel RealSense (60 fps color, cleaner frames -> fewer tracking drops).
# Auto-selects the plain D435 over the D435IF (IR-filter variant, worse color).
python perception_node.py --source realsense --show
#   --list-cameras          list connected RealSense devices + serials
#   --rs-serial 938422071322 or --rs-serial D435   pick a specific camera

# or a plain webcam (this machine's is index 10, not 0)
python perception_node.py --source webcam --camera 10 --show
#   --hand left --mirror   teleop with your left hand
#   --flip                 selfie/mirror view
```

Put your **right hand** in front of the camera; the XHand fingers follow. Measured
**60–80 FPS** end-to-end on RealSense (RTX 4090). Brief tracking dropouts hold the
last grasp; only after `--timeout` (2 s) does the hand relax to rest.

### Tuning (all on the perception node)

| Symptom | Knob | Direction |
|---|---|---|
| fingers jitter when still | `--min-cutoff` | lower (e.g. 1.0) |
| response feels laggy | `--beta` ↑, `--retarget-alpha` ↑ | higher (snappier) |
| tracking drops out a lot | `--redetect-interval` | lower (e.g. 6 — more YOLO, more robust) |
| want raw feed, no smoothing | `--no-filter` | — |

**How the speed/robustness comes from ROI tracking:** WiLoR's YOLO detector is the
slow, drop-prone stage. `hand_estimator.py` runs it only every `--redetect-interval`
frames; between detections it crops the ViT to the previous frame's keypoint box
(`predict_with_bboxes`). Fewer detector calls → faster, and bridging its misses →
steadier tracking.

## Test without a camera

`fake_source.py` stands in for the perception node, publishing a scripted
open↔close trajectory — use it to check the sim side alone:
```bash
# terminal 1: python teleop_sim.py --debug
# terminal 2 (any env):
python fake_source.py --rate 30 --period 3.0
```

## Files

- `teleop.py` — single-command launcher: spawns both nodes in their conda envs, manages shutdown (stdlib only).
- `real_node.py` — REAL XHand driver: UDP subscribe -> safety layers -> RS-485 position commands (see "Real hardware").
- `viz_panel.py` — `--show` dashboard: camera+skeleton | MANO 3D pose (mapping input, front/side views) | 12 xhand joint bars with URDF limits (mapping output = the exact wire values the sim/real hand executes).
- `protocol.py` — UDP wire format + the canonical 12-joint order (stdlib only; imported by both envs).
- `camera.py` — `CVCamera` (webcam, MJPG + 1-frame buffer) and `RealSenseCamera` (color stream, auto reset-on-busy).
- `hand_estimator.py` — `HandEstimator` interface + `WiLoREstimator` with detection-skip / ROI tracking. Swap the backend here to try a newer model.
- `filters.py` — One-Euro adaptive filter for the MANO keypoints (kills jitter without adding lag).
- `retarget.py` — `HandRetargeter`: 21 MANO keypoints → 12 xhand joints via DexPilot.
- `perception_node.py` — camera → WiLoR → filter → retarget → UDP publish (env `wilor`).
- `teleop_sim.py` — UDP subscribe → drive fixed-base XHand in Isaac Sim (env `env_isaaclab`).
- `fake_source.py` — scripted trajectory publisher for camera-less testing.

## Scope / next

- **Hand only.** The XHand is spawned fixed-base (`fix_root_link=True` +
  `disable_gravity`, same as the `xhand_repose` task) so the wrist never drifts and
  only the fingers move. DexPilot drives the 7 proximal joints; the 5 distal joints
  sit at their regularized midpoint (~0.96 rad) — fine for power grasps.
- **Next:** wrist 6-DoF → xarm7 IK for full-arm teleop, then real-hardware output
  (xhand SDK) with safety limits.

## Setup notes (already done on this machine)

`wilor` env built with: `torch==2.5.1 torchvision==0.20.1` (cu121), then chumpy
(`--no-build-isolation`), WiLoR-mini deps, `--no-deps` WiLoR-mini,
`dex_retargeting==0.4.6`, and `pyrealsense2`. WiLoR weights auto-download to the
package's `pretrained_models/` on first run. `numpy<2` is pinned (chumpy/smplx);
the cv2 5.0 "needs numpy>=2" warning is cosmetic — interop verified. The RealSense
D435 sometimes needs a hardware reset if a prior run crashed holding it —
`RealSenseCamera` does this automatically on the first-frame timeout.
