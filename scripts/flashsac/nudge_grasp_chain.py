# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Full tabletop chain: NUDGE (FlashSAC) -> RETRACT -> GRASP (PPO) -> IK LIFT.

From a random tool pose, the learned nudge policy pushes the tool back into the graspable
pose family; the arm then retracts to the home pose (joint-space, scripted) so the PPO grasp
policy takes over exactly inside its training distribution -- with the tool sitting at the
oracle pose family it was trained on; on a confirmed latch the DLS IK raises the palm +22cm
with a distal grip servo (the验证过的 90%-recovery lift).

All four stages run asynchronously per env on one physics timeline (no snapshot restore).
Strict success = true mesh clearance >= 20cm with latch/quality/force/speed held 15 frames.
Single attempt per env: an env is consumed at its first success or stage-deadline failure.

``--attempts N`` (with ``--num_envs 1``) records an mp4 per successful episode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="NUDGE -> RETRACT -> GRASP -> IK-LIFT chain.")
parser.add_argument("--task", type=str, default="Pick-Tool-Token-Direct-v0")
parser.add_argument("--nudge_checkpoint", type=Path, required=True, help="FlashSAC checkpoint dir")
parser.add_argument("--grasp_checkpoint", type=Path, default=None, help="rl_games 115/21 .pth")
parser.add_argument(
    "--grasp_flashsac_checkpoint",
    type=Path,
    default=None,
    help="FlashSAC checkpoint dir for the GRASP stage (the stage-2 close policy trained from "
    "post-nudge self-play states).  Replaces the PPO grasp actor; pair with --no_retract "
    "--retract_steps 0 for the direct nudge->close handoff the policy was trained on.",
)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--yaw_range", type=float, default=1.57, help="reset yaw sampled from [-r, r]")
# stage deadlines/gates (control steps)
parser.add_argument("--nudge_deadline", type=int, default=400)
parser.add_argument("--nudge_hold", type=int, default=10, help="in-target settled frames to advance")
parser.add_argument(
    "--nudge_pregrasp_min",
    type=float,
    default=0.0,
    help="advance gate addition (v7): the hand must also hold pregrasp readiness >= this, so "
    "the grasp stage starts inside the close policy's oracle-trained distribution.",
)
parser.add_argument("--retract_steps", type=int, default=100)
parser.add_argument(
    "--reposition_checkpoint",
    type=Path,
    default=None,
    help="FlashSAC checkpoint dir for a learned REPOSITION stage replacing the scripted "
    "retract: after the nudge gate, this policy parks the hand grasp-ready (pregrasp "
    "readiness >= --nudge_pregrasp_min held --reposition_hold frames) before the grasp "
    "stage takes over.  No home return, no scripted motion.",
)
parser.add_argument("--reposition_hold", type=int, default=10)
parser.add_argument("--reposition_deadline", type=int, default=300)
parser.add_argument(
    "--no_retract",
    action="store_true",
    help="skip RETRACT: hand the close-specialist grasp policy the nudge end state directly "
    "(hand already low over the centered tool -- the close_start pregrasp family).",
)
parser.add_argument(
    "--grasp_only",
    action="store_true",
    help="debug bisect: start every env directly in the GRASP phase from the fresh reset "
    "(no nudge/retract), isolating harness effects from stage contamination.",
)
parser.add_argument(
    "--max_cycles",
    type=int,
    default=1,
    help="grasp retries: on a grasp-deadline miss, cycle back to NUDGE (which re-centers "
    "whatever the failed attempt pushed away) and try again, up to this many cycles.",
)
parser.add_argument("--grasp_deadline", type=int, default=450)
parser.add_argument("--grasp_confirm", type=int, default=15)
parser.add_argument("--lift_ramp", type=int, default=220)
parser.add_argument(
    "--lift_profile",
    choices=("minjerk", "linear"),
    default="minjerk",
    help="height reference profile for the IK lift: 'minjerk' (10t^3-15t^4+6t^5, zero "
    "velocity/acceleration at both ends -- smooth start and stop) or the old 'linear' ramp "
    "(constant-rate with velocity discontinuities at start and top).",
)
parser.add_argument("--hold_steps", type=int, default=60)
parser.add_argument("--stable_steps", type=int, default=15)
parser.add_argument("--lift_height", type=float, default=0.22)
# IK / grip servo (the validated ppo_grasp_ik_lift settings)
parser.add_argument("--damping", type=float, default=0.05)
parser.add_argument("--ik_gain", type=float, default=0.5, help="IK feedback gain on the pose error")
parser.add_argument(
    "--ik_pos_deadband", type=float, default=0.0015, help="ignore position errors below this (m)"
)
parser.add_argument(
    "--ik_rot_deadband", type=float, default=0.01, help="ignore rotation errors below this (rad)"
)
parser.add_argument("--max_cart_step", type=float, default=0.004)
parser.add_argument("--max_rot_step", type=float, default=0.05)
parser.add_argument("--grip_force_target", type=float, default=3.0)
parser.add_argument("--grip_force_limit", type=float, default=20.0)
parser.add_argument("--grip_servo_step", type=float, default=0.006)
parser.add_argument("--grip_servo_range", type=float, default=0.60)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=Path, default=Path("/tmp/pick_tool_nudge_grasp_chain.json"))
parser.add_argument(
    "--inhand_checkpoint",
    type=Path,
    default=None,
    help="FlashSAC checkpoint dir for the IN-HAND stage: after the strict lift success, this "
    "policy reorients the held tool until the index fingertip sits on the functional point; "
    "chain success then requires that contact instead of the bare lift.",
)
parser.add_argument("--inhand_dist", type=float, default=0.018, help="fingertip-to-point gate (m)")
parser.add_argument("--inhand_confirm", type=int, default=15)
parser.add_argument("--inhand_deadline", type=int, default=320)
parser.add_argument(
    "--capture_held",
    type=Path,
    default=None,
    help="capture the full boundary snapshot of every env at its strict-success moment "
    "(tool lifted and stably held) into this dataset -- the spawn states for the in-hand "
    "reorientation stage.  Boundary key: 'inhand_start'.",
)
# demo recording
parser.add_argument("--attempts", type=int, default=1)
parser.add_argument("--max_clips", type=int, default=3)
parser.add_argument("--fps", type=int, default=40)
parser.add_argument("--video_folder", type=Path, default=Path("/tmp/nudge_chain_video"))
parser.add_argument("--cam_eye", type=float, nargs=3, default=[1.9, 0.95, 0.9])
parser.add_argument("--cam_lookat", type=float, nargs=3, default=[0.5, 0.0, 0.32])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if (args_cli.grasp_checkpoint is None) == (args_cli.grasp_flashsac_checkpoint is None):
    parser.error("provide exactly one of --grasp_checkpoint / --grasp_flashsac_checkpoint")

demo_mode = args_cli.attempts > 1
if demo_mode and args_cli.num_envs != 1:
    parser.error("--attempts > 1 requires --num_envs 1")
if demo_mode:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.math import compute_pose_error, sample_uniform
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401

# Sibling imports: the FlashSAC bridge helpers live here; the PPO actor loader lives in
# scripts/rl_games.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "rl_games"))
import agent_bridge  # noqa: E402,F401  (import side effect installs the pinned flash_rl path)
from flash_rl.agents.flashSAC.network import FlashSACActor  # noqa: E402

from bc_pick_tool import MigratedActor, clone_state, load_torch  # noqa: E402
from pick_tool_shared import capture_boundary  # noqa: E402


_COMPILED_PREFIX = "_orig_mod."

PHASE_NUDGE, PHASE_RETRACT, PHASE_GRASP, PHASE_LIFT, PHASE_INHAND, PHASE_DONE = range(6)


def _limit_norm(value: torch.Tensor, limit: float) -> torch.Tensor:
    norm = value.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
    return value * torch.clamp(limit / norm, max=1.0)


def _deadband(value: torch.Tensor, band: float) -> torch.Tensor:
    """Shrink a batched vector toward zero by ``band`` in L2 norm (zero inside the band)."""
    norm = value.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
    return value * ((norm - band).clamp_min(0.0) / norm)


def _summary(value: torch.Tensor) -> dict[str, float] | None:
    flat = value.detach().float().flatten()
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return None
    q = torch.quantile(flat, torch.tensor((0.0, 0.1, 0.5, 0.9, 1.0), device=flat.device))
    return dict(zip(("min", "p10", "median", "p90", "max"), (float(x) for x in q), strict=True))


def load_nudge_actor(checkpoint: Path, device: torch.device) -> FlashSACActor:
    payload = torch.load(checkpoint / "actor.pt", map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "network_state_dict" not in payload:
        raise KeyError("actor.pt must contain 'network_state_dict'")
    state = payload["network_state_dict"]
    canonical: dict[str, torch.Tensor] = {}
    prefixed = [str(k).startswith(_COMPILED_PREFIX) for k in state]
    strip = all(prefixed) and len(prefixed) > 0
    for key, value in state.items():
        canonical[key.removeprefix(_COMPILED_PREFIX) if strip else key] = value
    # Production architecture (train.py non-smoke defaults).
    actor = FlashSACActor(num_blocks=2, input_dim=115, hidden_dim=128, action_dim=21)
    actor.load_state_dict(canonical)
    return actor.to(device).eval()


def _checkpoint_model(path: Path) -> dict[str, torch.Tensor]:
    raw = load_torch(path)
    if not isinstance(raw, dict):
        raise TypeError("checkpoint root is not a dictionary")
    payload = raw if isinstance(raw.get("model"), dict) else (raw[0] if 0 in raw else raw.get("0"))
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise KeyError("checkpoint must contain {'model': state_dict}, optionally below key 0")
    return clone_state(payload["model"])


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(args_cli.seed)
    n = args_cli.num_envs
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n)
    cfg.seed = args_cli.seed
    cfg.episode_length_s = 60.0
    cfg.terminate_on_drop = False
    cfg.success_hold_steps = 100000
    cfg.tactile_terminate_steps = 1_000_000_000
    cfg.tactile_hard_terminate_steps = 1_000_000_000
    cfg.reset_object_yaw_range = (-args_cli.yaw_range, args_cli.yaw_range)
    if demo_mode:
        cfg.viewer.eye = tuple(args_cli.cam_eye)
        cfg.viewer.lookat = tuple(args_cli.cam_lookat)
        cfg.viewer.origin_type = "world"
    realizable_arm_step = float(cfg.act_moving_average * cfg.action_scale)

    env = gym.make(args_cli.task, cfg=cfg, render_mode="rgb_array" if demo_mode else None)
    u = env.unwrapped
    dev = u.device

    nudge_actor = load_nudge_actor(args_cli.nudge_checkpoint, dev)
    reposition_actor = (
        load_nudge_actor(args_cli.reposition_checkpoint, dev)
        if args_cli.reposition_checkpoint is not None
        else None
    )
    inhand_actor = (
        load_nudge_actor(args_cli.inhand_checkpoint, dev)
        if args_cli.inhand_checkpoint is not None
        else None
    )
    if args_cli.grasp_flashsac_checkpoint is not None:
        close_actor = load_nudge_actor(args_cli.grasp_flashsac_checkpoint, dev)

        def grasp_policy(policy_obs: torch.Tensor) -> torch.Tensor:
            mean, _ = close_actor.get_mean_and_std(policy_obs, training=False)
            return torch.tanh(mean)

    else:
        grasp_actor = MigratedActor(_checkpoint_model(args_cli.grasp_checkpoint)).to(dev).eval()
        if grasp_actor.observation_dim != 115 or grasp_actor.action_dim != 21:
            raise RuntimeError("grasp checkpoint must be a 115/21 actor")

        def grasp_policy(policy_obs: torch.Tensor) -> torch.Tensor:
            return grasp_actor(policy_obs).clamp(-1.0, 1.0)

    env.reset()
    hand_names = [u.robot.joint_names[i] for i in u._hand_ids_t.tolist()]
    distal_by_fingertip = {
        "thumb_rota_link2": "thumb_joint2",
        "index_rota_link2": "index_joint2",
        "mid_link2": "middle_joint1",
        "ring_link2": "ring_joint1",
        "pinky_link2": "pinky_joint1",
    }
    servo_hand_ids = u._hand_ids_t[
        torch.tensor([hand_names.index(distal_by_fingertip[nm]) for nm in u.ee_names], device=dev)
    ]
    ready_pose = torch.tensor(cfg.nudge_ready_arm_joints, device=dev, dtype=torch.float32)
    home_arm = u.robot.data.default_joint_pos[:, u._arm_ids_t].clone()
    eye6 = torch.eye(6, device=dev).unsqueeze(0)

    commanded = u.dof_targets.detach().clone()
    lift_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    retract_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    home_hand = u.robot.data.default_joint_pos[:, u._hand_ids_t].clone()
    original_pre_physics = u._pre_physics_step

    def hybrid_pre_physics(actions: torch.Tensor) -> None:
        original_pre_physics(actions)
        if bool(lift_mask.any()):
            u.dof_targets[lift_mask] = commanded[lift_mask]
        if bool(retract_mask.any()) and args_cli.reposition_checkpoint is None:
            # The token manifold cannot express the fully OPEN home hand, so the scripted
            # retract opens the fingers at the dof-target level (<=0.05 rad/step) while the
            # arm returns home; otherwise the grasp policy inherits a curled token-0 fist it
            # has never seen (measured: hand joints ~1.0 rad at grasp entry -> garbage acts).
            current = u.dof_targets[:, u._hand_ids_t]
            step_open = (home_hand - current).clamp(-0.05, 0.05)
            opened = current + step_open
            u.dof_targets[:, u._hand_ids_t] = torch.where(
                retract_mask.unsqueeze(-1), opened, current
            )

    u._pre_physics_step = hybrid_pre_physics

    def apply_ready_pose() -> None:
        noise = sample_uniform(
            -cfg.reset_arm_joint_noise, cfg.reset_arm_joint_noise, (n, 7), dev
        )
        joint_pos = u.robot.data.joint_pos.clone()
        arm = (ready_pose.unsqueeze(0) + noise).clamp(
            u.dof_lower[:, u._arm_ids_t], u.dof_upper[:, u._arm_ids_t]
        )
        joint_pos[:, u._arm_ids_t] = arm
        u.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
        u.robot.set_joint_position_target(joint_pos)
        u.dof_targets.copy_(joint_pos)
        u.scene.write_data_to_sim()
        u.sim.forward()
        u.scene.update(dt=u.physics_dt)

    def run_attempt(capture: bool) -> dict:
        nonlocal lift_mask, retract_mask
        # Normal home reset: the v6 nudge policy was annealed to operate from the home pose.
        obs, _ = env.reset()
        initial_phase = PHASE_GRASP if args_cli.grasp_only else PHASE_NUDGE
        phase = torch.full((n,), initial_phase, dtype=torch.long, device=dev)
        phase_entry = torch.zeros(n, dtype=torch.long, device=dev)
        nudge_hold = torch.zeros(n, dtype=torch.long, device=dev)
        repos_hold = torch.zeros(n, dtype=torch.long, device=dev)
        cycles = torch.zeros(n, dtype=torch.long, device=dev)
        grasp_confirm = torch.zeros(n, dtype=torch.long, device=dev)
        commanded.copy_(u.dof_targets)
        palm_start = torch.zeros((n, 3), device=dev)
        palm_quat_target = torch.zeros((n, 4), device=dev)
        servo_lower = torch.zeros((n, len(u.ee_names)), device=dev)
        servo_upper = torch.zeros((n, len(u.ee_names)), device=dev)
        servo_active = torch.zeros((n, len(u.ee_names)), dtype=torch.bool, device=dev)
        lift_counter = torch.zeros(n, dtype=torch.long, device=dev)
        lift_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        retract_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        stable_count = torch.zeros(n, dtype=torch.long, device=dev)
        success = torch.zeros(n, dtype=torch.bool, device=dev)
        failed = torch.zeros(n, dtype=torch.bool, device=dev)
        nudged = torch.zeros(n, dtype=torch.bool, device=dev)
        grasped = torch.zeros(n, dtype=torch.bool, device=dev)
        nudge_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        grasp_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        success_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        max_clearance = torch.full((n,), -float("inf"), device=dev)
        lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        ih_hold = torch.zeros(n, dtype=torch.long, device=dev)
        ih_lost = torch.zeros(n, dtype=torch.long, device=dev)
        ih_min_dist = torch.full((n,), 10.0, device=dev)
        # grasp-phase diagnostics
        g_max_quality = torch.zeros(n, device=dev)
        g_max_proximity = torch.zeros(n, device=dev)
        g_ever_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        g_entry_palm_z = torch.zeros(n, device=dev)
        g_entry_home_err = torch.zeros(n, device=dev)
        g_min_ft_dist = torch.full((n,), 10.0, device=dev)
        g_action_arm_abs = torch.zeros(n, device=dev)
        g_steps = torch.zeros(n, device=dev)
        frames: list = []

        total_steps = (
            args_cli.max_cycles
            * (args_cli.nudge_deadline + args_cli.retract_steps + args_cli.grasp_deadline)
            + args_cli.lift_ramp
            + args_cli.hold_steps
            + (args_cli.inhand_deadline if inhand_actor is not None else 0)
        )
        for step in range(total_steps):
            if step == 0 and os.environ.get("CHAIN_DUMP_OBS"):
                row = obs["policy"][0]
                print(
                    "[OBS-DUMP fresh reset env0] obj_pos_b:",
                    [round(float(x), 3) for x in row[56:59]],
                    "obj_quat:", [round(float(x), 3) for x in row[59:63]],
                    "target_pos:", [round(float(x), 3) for x in row[63:66]],
                    "lift:", round(float(row[86]), 3),
                    flush=True,
                )
            u._compute_intermediate_values()
            signals = u._compute_grasp_signals()
            force_max = signals["force_magnitude"].max(dim=-1).values
            clearance = u._object_true_min_z() - u._table_surface_z
            errors = u._nudge_pose_errors()
            obj_speed = u.object.data.root_com_lin_vel_w.norm(dim=-1)

            # ---- NUDGE gate ----
            in_target = (
                (errors["pos_error"] <= cfg.nudge_pos_tolerance)
                & (errors["heading_error"] <= cfg.nudge_yaw_tolerance)
                & (errors["tip_cos"] >= cfg.nudge_tip_cos_min)
                & (clearance.abs() <= cfg.nudge_on_table_tolerance)
                & (obj_speed <= cfg.nudge_max_obj_speed)
            )
            pose_ok = in_target
            if args_cli.nudge_pregrasp_min > 0.0 and reposition_actor is None:
                # v7-style single-policy gate; with a reposition stage the pregrasp
                # requirement moves to that stage's completion instead.
                in_target = in_target & (
                    u._nudge_pregrasp_score() >= args_cli.nudge_pregrasp_min
                )
            nudging = phase == PHASE_NUDGE
            nudge_hold = torch.where(nudging & in_target, nudge_hold + 1, torch.zeros_like(nudge_hold))
            advance = nudging & (nudge_hold >= args_cli.nudge_hold)
            if bool(advance.any()):
                # Even with --no_retract the hand must pass through the opening ramp (the
                # token manifold's neutral is a fist); --no_retract only skips the arm-home
                # return, keeping the palm parked low over the tool while the fingers open.
                phase[advance] = PHASE_RETRACT
                phase_entry[advance] = step
                nudged |= advance
                nudge_step[advance] = step
            nudge_out = nudging & (step - phase_entry >= args_cli.nudge_deadline) & ~advance
            if bool(nudge_out.any()):
                phase[nudge_out] = PHASE_DONE
                failed |= nudge_out

            # ---- RETRACT / REPOSITION completion ----
            retracting = phase == PHASE_RETRACT
            if reposition_actor is not None:
                parked = (
                    retracting
                    & pose_ok
                    & (u._nudge_pregrasp_score() >= args_cli.nudge_pregrasp_min)
                )
                repos_hold = torch.where(parked, repos_hold + 1, torch.zeros_like(repos_hold))
                retract_done = retracting & (repos_hold >= args_cli.reposition_hold)
                repos_out = (
                    retracting
                    & (step - phase_entry >= args_cli.reposition_deadline)
                    & ~retract_done
                )
                if bool(repos_out.any()):
                    phase[repos_out] = PHASE_DONE
                    failed |= repos_out
            else:
                retract_done = retracting & (step - phase_entry >= args_cli.retract_steps)
            if bool(retract_done.any()) and bool(retract_done[0]) and os.environ.get("CHAIN_DUMP_OBS"):
                groups = {
                    "joint_pos[0:19]": (0, 19), "joint_vel[19:38]": (19, 38),
                    "ee_pos_b[38:53]": (38, 53), "palm_b[53:56]": (53, 56),
                    "obj_pos_b[56:59]": (56, 59), "obj_quat[59:63]": (59, 63),
                    "target_pos[63:66]": (63, 66), "target_quat[66:70]": (66, 70),
                    "old_action[70:86]": (70, 86), "lift[86:87]": (86, 87),
                    "residual[87:92]": (87, 92), "phase23[92:115]": (92, 115),
                }
                row = obs["policy"][0]
                print("[OBS-DUMP grasp entry env0]", flush=True)
                for name, (a, b) in groups.items():
                    seg = row[a:b]
                    print(f"  {name}: {[round(float(x), 3) for x in seg]}", flush=True)
            if bool(retract_done.any()):
                phase[retract_done] = PHASE_GRASP
                phase_entry[retract_done] = step
                arm_home_err = (
                    (u.robot.data.joint_pos[:, u._arm_ids_t] - home_arm).abs().max(dim=-1).values
                )
                g_entry_palm_z = torch.where(
                    retract_done, u.palm_center_w[:, 2], g_entry_palm_z
                )
                g_entry_home_err = torch.where(retract_done, arm_home_err, g_entry_home_err)

            # ---- GRASP gate (same latch contract as ppo_grasp_ik_lift) ----
            latched = (
                u._is_grasped
                & (signals["grasp_quality"] >= cfg.grasp_quality_high)
                & (signals["hold_quality"] >= cfg.close_option_min_hold_quality)
                & (force_max <= cfg.grasp_bonus_max_force)
            )
            grasping = phase == PHASE_GRASP
            grasp_confirm = torch.where(
                grasping & latched, grasp_confirm + 1, torch.zeros_like(grasp_confirm)
            )
            handoff = grasping & (grasp_confirm >= args_cli.grasp_confirm)
            if bool(handoff.any()):
                phase[handoff] = PHASE_LIFT
                phase_entry[handoff] = step
                lift_counter[handoff] = 0
                commanded[handoff] = u.dof_targets[handoff].detach().clone()
                palm_start[handoff] = u.robot.data.body_pos_w[handoff][:, u.palm_idx]
                palm_quat_target[handoff] = u.robot.data.body_quat_w[handoff][:, u.palm_idx]
                frozen = u.dof_targets[handoff][:, servo_hand_ids]
                servo_lower[handoff] = frozen
                servo_upper[handoff] = torch.minimum(
                    frozen + args_cli.grip_servo_range, u.dof_upper[handoff][:, servo_hand_ids]
                )
                # Only servo fingers that are actually holding at handoff.  A finger with no
                # contact (measured: the middle finger at 0N for entire lifts) otherwise
                # ratchets 0.006 rad EVERY step chasing its 3N target -- a full 0.6 rad sweep
                # during the lift that stirs the grasp and rocks the tool.
                handoff_force = u._finger_object_force_magnitudes()
                servo_active[handoff] = handoff_force[handoff] > 1.0
                grasped |= handoff
                grasp_step[handoff] = step
            grasp_out = grasping & (step - phase_entry >= args_cli.grasp_deadline) & ~handoff
            if bool(grasp_out.any()):
                # Retry loop: a failed grasp usually pushed the tool off-pose; NUDGE is exactly
                # the skill that recovers that, so cycle back instead of giving up.
                retry = grasp_out & (cycles < args_cli.max_cycles - 1)
                give_up = grasp_out & ~retry
                if bool(retry.any()):
                    cycles[retry] += 1
                    phase[retry] = PHASE_NUDGE
                    phase_entry[retry] = step
                    nudge_hold[retry] = 0
                phase[give_up] = PHASE_DONE
                failed |= give_up
            g_max_quality = torch.where(
                grasping, torch.maximum(g_max_quality, signals["grasp_quality"]), g_max_quality
            )
            g_max_proximity = torch.where(
                grasping, torch.maximum(g_max_proximity, signals["proximity_quality"]), g_max_proximity
            )
            g_ever_latch |= grasping & u._is_grasped
            g_min_ft_dist = torch.where(
                grasping,
                torch.minimum(g_min_ft_dist, u._curr_fingertip_distances.mean(dim=-1)),
                g_min_ft_dist,
            )

            # ---- IK lift ----
            retract_mask = phase == PHASE_RETRACT
            # Successful envs KEEP the frozen command after DONE: without this they fall
            # through to the zero action, which drives the hand back toward the token-0
            # posture and the held tool slips out on camera right after the success latch.
            # Only PHASE_LIFT envs get IK/servo UPDATES (ik_mask); DONE envs replay their
            # frozen `commanded` untouched -- after an in-hand stage the palm has moved and
            # re-running the lift IK would drag it back to the lift-end pose.
            lift_mask = (phase == PHASE_LIFT) | (success & (phase == PHASE_DONE))
            ik_mask = phase == PHASE_LIFT
            if bool(ik_mask.any()):
                s = (lift_counter.float() / float(args_cli.lift_ramp)).clamp(0.0, 1.0)
                if args_cli.lift_profile == "minjerk":
                    # Peak reference speed is 1.875x the average (~1.9mm/step at the default
                    # 22cm/220 steps), still under the 4mm max_cart_step, so the arm tracks
                    # the smooth reference instead of saturating the per-step clamp.
                    s = s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)
                height = args_cli.lift_height * s
                current_pos = u.robot.data.body_pos_w[:, u.palm_idx]
                current_quat = u.robot.data.body_quat_w[:, u.palm_idx]
                desired_pos = palm_start.clone()
                desired_pos[:, 2] += height
                pos_error, rot_error = compute_pose_error(
                    current_pos, current_quat, desired_pos, palm_quat_target,
                    rot_error_type="axis_angle",
                )
                # Anti-limit-cycle conditioning (measured: full-gain P tracking through the
                # dof-target EMA lag oscillates the palm +-4mm xy / 0.02 rad at ~0.7s period,
                # amplified into visible tool swing by the ~30cm handle lever).  A deadband
                # stops the chase below perception scale and the 0.5 gain removes the
                # overshoot; the vertical (z) reference keeps full authority via the ramp.
                pos_db = _deadband(pos_error, args_cli.ik_pos_deadband)
                rot_db = _deadband(rot_error, args_cli.ik_rot_deadband)
                delta = torch.cat(
                    (
                        _limit_norm(args_cli.ik_gain * pos_db, args_cli.max_cart_step),
                        _limit_norm(args_cli.ik_gain * rot_db, args_cli.max_rot_step),
                    ),
                    dim=-1,
                )
                jac = u.robot.root_physx_view.get_jacobians()[:, u._palm_jac_idx, :, :][:, :, u._arm_ids_t]
                jt = jac.transpose(1, 2)
                solved = torch.linalg.solve(jac @ jt + (args_cli.damping**2) * eye6, delta.unsqueeze(-1))
                delta_q = (jt @ solved).squeeze(-1).clamp(-realizable_arm_step, realizable_arm_step)
                next_arm = torch.maximum(
                    torch.minimum(commanded[:, u._arm_ids_t] + delta_q, u.dof_upper[:, u._arm_ids_t]),
                    u.dof_lower[:, u._arm_ids_t],
                )
                commanded[:, u._arm_ids_t] = torch.where(
                    ik_mask.unsqueeze(-1), next_arm, commanded[:, u._arm_ids_t]
                )
                force = u._finger_object_force_magnitudes()
                distal_cmd = commanded[:, servo_hand_ids]
                delta_grip = (
                    args_cli.grip_servo_step
                    * servo_active.float()
                    * (
                        (force < args_cli.grip_force_target).float()
                        - (force > args_cli.grip_force_limit).float()
                    )
                )
                new_distal = torch.clamp(distal_cmd + delta_grip, servo_lower, servo_upper)
                commanded[:, servo_hand_ids] = torch.where(
                    ik_mask.unsqueeze(-1), new_distal, distal_cmd
                )
                lift_counter = torch.where(ik_mask, lift_counter + 1, lift_counter)
                if os.environ.get("CHAIN_DUMP_LIFT") and bool(ik_mask[0]):
                    print(
                        "[LIFT]",
                        int(lift_counter[0]),
                        "palm",
                        [round(float(x), 4) for x in current_pos[0]],
                        "des",
                        [round(float(x), 4) for x in desired_pos[0]],
                        "err",
                        [round(float(x), 4) for x in pos_error[0]],
                        "rot_err",
                        [round(float(x), 4) for x in rot_error[0]],
                        "grip",
                        [round(float(x), 2) for x in force[0]],
                        flush=True,
                    )

            # ---- action composition per phase ----
            policy_obs = obs["policy"]
            mean, _ = nudge_actor.get_mean_and_std(policy_obs, training=False)
            nudge_action = torch.tanh(mean)
            grasp_action = grasp_policy(policy_obs)
            if reposition_actor is not None:
                repos_mean, _ = reposition_actor.get_mean_and_std(policy_obs, training=False)
                retract_action = torch.tanh(repos_mean)
            else:
                retract_action = torch.zeros((n, 21), device=dev)
                if not args_cli.no_retract:
                    retract_action[:, :7] = (
                        (home_arm - u.dof_targets[:, u._arm_ids_t]) / realizable_arm_step
                    ).clamp(-1.0, 1.0)
            action = torch.zeros((n, 21), device=dev)
            phase_actions = [
                (phase == PHASE_NUDGE, nudge_action),
                (phase == PHASE_RETRACT, retract_action),
                (phase == PHASE_GRASP, grasp_action),
            ]
            if inhand_actor is not None:
                ih_mean, _ = inhand_actor.get_mean_and_std(policy_obs, training=False)
                phase_actions.append((phase == PHASE_INHAND, torch.tanh(ih_mean)))
            for mask, act in phase_actions:
                action = torch.where(mask.unsqueeze(-1), act, action)
            in_grasp = phase == PHASE_GRASP
            g_action_arm_abs += torch.where(
                in_grasp, grasp_action[:, :7].abs().mean(dim=-1), torch.zeros_like(g_action_arm_abs)
            )
            g_steps += in_grasp.float()
            obs, _, _, _, _ = env.step(action)
            if capture:
                frames.append(env.unwrapped.render())

            # ---- strict success (lift phase only) ----
            post = u._compute_grasp_signals()
            post_force = post["force_magnitude"].max(dim=-1).values
            post_clear = u._object_true_min_z() - u._table_surface_z
            slow = (
                (u.object.data.root_com_lin_vel_w.norm(dim=-1) < cfg.success_max_obj_lin_speed)
                & (u.object.data.root_com_ang_vel_w.norm(dim=-1) < cfg.success_max_obj_ang_speed)
            )
            strict = (
                (phase == PHASE_LIFT)
                & (post_clear >= cfg.lift_success_height)
                & u._is_grasped
                & (post["grasp_quality"] >= cfg.grasp_quality_high)
                & (post["hold_quality"] >= cfg.close_option_min_hold_quality)
                & (post_force <= cfg.grasp_bonus_max_force)
                & slow
            )
            stable_count = torch.where(strict, stable_count + 1, torch.zeros_like(stable_count))
            newly = (~lifted) & (phase == PHASE_LIFT) & (stable_count >= args_cli.stable_steps)
            lifted |= newly
            if args_cli.capture_held is not None and bool(newly.any()):
                snap = capture_boundary(u)
                ids = newly.nonzero(as_tuple=False).squeeze(-1)
                held_snaps.append({k: v[ids].cpu() for k, v in snap.items()})
            if inhand_actor is None:
                success_step[newly] = step
                success |= newly
                phase[newly] = PHASE_DONE
            else:
                phase[newly] = PHASE_INHAND
                phase_entry[newly] = step

            # ---- IN-HAND stage: index fingertip onto the functional point ----
            if inhand_actor is not None:
                inhand = phase == PHASE_INHAND
                if bool(inhand.any()):
                    ih_dist = u._inhand_distance()
                    ih_min_dist = torch.where(
                        inhand, torch.minimum(ih_min_dist, ih_dist), ih_min_dist
                    )
                    at_point = inhand & (ih_dist <= args_cli.inhand_dist) & u._is_grasped
                    ih_hold = torch.where(at_point, ih_hold + 1, torch.zeros_like(ih_hold))
                    ih_lost = torch.where(
                        inhand & (~u._is_grasped), ih_lost + 1, torch.zeros_like(ih_lost)
                    )
                    pointed_now = inhand & (ih_hold >= args_cli.inhand_confirm)
                    if bool(pointed_now.any()):
                        # Freeze the exact posture for the post-success hold.
                        commanded[pointed_now] = u.dof_targets[pointed_now].detach().clone()
                        success_step[pointed_now] = step
                        success |= pointed_now
                        phase[pointed_now] = PHASE_DONE
                    ih_fail = inhand & (
                        (ih_lost >= 10)
                        | (post_clear < 0.10)
                        | (step - phase_entry >= args_cli.inhand_deadline)
                    ) & ~pointed_now
                    if bool(ih_fail.any()):
                        phase[ih_fail] = PHASE_DONE
                        failed |= ih_fail
            max_clearance = torch.maximum(
                max_clearance, torch.where(phase >= PHASE_LIFT, post_clear, max_clearance)
            )

        return {
            "nudged": int(nudged.sum()),
            "grasped": int(grasped.sum()),
            "lifted": int(lifted.sum()),
            "inhand_min_dist": _summary(ih_min_dist[lifted]) if bool(lifted.any()) else None,
            "success": int(success.sum()),
            "frames": frames,
            "success_flag": bool(success.any()),
            "nudge_step": _summary(nudge_step[nudged].float()) if bool(nudged.any()) else None,
            "grasp_step": _summary(grasp_step[grasped].float()) if bool(grasped.any()) else None,
            "success_step": _summary(success_step[success].float()) if bool(success.any()) else None,
            "max_clearance": _summary(max_clearance),
            "grasp_phase_ever_latch": int(g_ever_latch.sum()),
            "grasp_phase_max_quality": _summary(g_max_quality),
            "grasp_phase_max_proximity": _summary(g_max_proximity),
            "grasp_entry_palm_z": _summary(g_entry_palm_z),
            "grasp_entry_home_err": _summary(g_entry_home_err),
            "grasp_min_fingertip_dist": _summary(g_min_ft_dist),
            "grasp_action_arm_abs_mean": _summary(g_action_arm_abs / g_steps.clamp_min(1.0)),
            "cycles_used": _summary(cycles.float()),
            "success_by_cycle": {
                str(c): int((success & (cycles == c)).sum()) for c in range(args_cli.max_cycles)
            },
            "final_phase_counts": {
                name: int((phase == idx).sum())
                for idx, name in enumerate(("NUDGE", "RETRACT", "GRASP", "LIFT", "DONE"))
            },
        }

    base = {
        "nudge_checkpoint": str(args_cli.nudge_checkpoint.resolve()),
        "grasp_checkpoint": str(
            (args_cli.grasp_flashsac_checkpoint or args_cli.grasp_checkpoint).resolve()
        ),
        "grasp_actor_kind": "flashsac" if args_cli.grasp_flashsac_checkpoint else "ppo",
        "num_envs": n,
        "seed": args_cli.seed,
        "yaw_range": args_cli.yaw_range,
        "gates": {
            "nudge_hold": args_cli.nudge_hold,
            "grasp_confirm": args_cli.grasp_confirm,
            "stable_steps": args_cli.stable_steps,
        },
    }

    held_snaps: list[dict[str, torch.Tensor]] = []

    if not demo_mode:
        r = run_attempt(capture=False)
        metrics = {
            **base,
            "nudged_count": r["nudged"],
            "grasped_count": r["grasped"],
            "lifted_count": r["lifted"],
            "inhand_min_dist": r["inhand_min_dist"],
            "chain_success_count": r["success"],
            "chain_success_rate": r["success"] / n,
            "nudge_rate": r["nudged"] / n,
            "grasp_rate_given_nudge": (r["grasped"] / r["nudged"]) if r["nudged"] else 0.0,
            "lift_rate_given_grasp": (r["success"] / r["grasped"]) if r["grasped"] else 0.0,
            "nudge_step": r["nudge_step"],
            "grasp_step": r["grasp_step"],
            "success_step": r["success_step"],
            "max_clearance": r["max_clearance"],
            "grasp_phase_ever_latch": r["grasp_phase_ever_latch"],
            "grasp_phase_max_quality": r["grasp_phase_max_quality"],
            "grasp_phase_max_proximity": r["grasp_phase_max_proximity"],
            "grasp_entry_palm_z": r["grasp_entry_palm_z"],
            "grasp_entry_home_err": r["grasp_entry_home_err"],
            "grasp_min_fingertip_dist": r["grasp_min_fingertip_dist"],
            "grasp_action_arm_abs_mean": r["grasp_action_arm_abs_mean"],
            "final_phase_counts": r["final_phase_counts"],
            "cycles_used": r["cycles_used"],
            "success_by_cycle": r["success_by_cycle"],
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
        print(
            f"CHAIN: nudged={r['nudged']}/{n} -> grasped={r['grasped']} -> lifted={r['success']} "
            f"({r['success']/n*100:.1f}% end-to-end)",
            flush=True,
        )
        print(f"wrote {args_cli.output}", flush=True)
        if args_cli.capture_held is not None and held_snaps:
            boundary = {k: torch.cat([s[k] for s in held_snaps]) for k in held_snaps[0]}
            args_cli.capture_held.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "boundaries": {"inhand_start": boundary},
                    "meta": {
                        "format_version": 1,
                        "kind": "chain_held_lift_boundaries",
                        "num_states": int(boundary["joint_pos"].shape[0]),
                        "seed": args_cli.seed,
                    },
                },
                args_cli.capture_held,
            )
            print(
                f"captured {int(boundary['joint_pos'].shape[0])} held-lift states -> "
                f"{args_cli.capture_held}",
                flush=True,
            )
    else:
        os.makedirs(args_cli.video_folder, exist_ok=True)
        for _ in range(6):
            env.unwrapped.render()
        saved: list[str] = []
        for attempt in range(args_cli.attempts):
            r = run_attempt(capture=True)
            if r["success_flag"] and len(saved) < args_cli.max_clips:
                import imageio.v2 as imageio

                path = os.path.join(args_cli.video_folder, f"chain_success_{len(saved) + 1}.mp4")
                imageio.mimwrite(
                    path, [np.asarray(f) for f in r["frames"]], fps=args_cli.fps, macro_block_size=None
                )
                saved.append(path)
                print(f"[demo] attempt {attempt}: SUCCESS -> {path}", flush=True)
            elif attempt == 0 and not r["success_flag"]:
                # Always keep one failure clip for debugging the stage handoffs.
                import imageio.v2 as imageio

                path = os.path.join(args_cli.video_folder, "chain_fail_debug.mp4")
                imageio.mimwrite(
                    path, [np.asarray(f) for f in r["frames"]], fps=args_cli.fps, macro_block_size=None
                )
                print(f"[demo] attempt {attempt}: FAIL (debug clip) -> {path}", flush=True)
            else:
                print(
                    f"[demo] attempt {attempt}: nudged={r['nudged']} grasped={r['grasped']} "
                    f"lifted={r['success']}",
                    flush=True,
                )
            if len(saved) >= args_cli.max_clips:
                break
        print(f"[demo] saved {len(saved)} chain clip(s)", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
