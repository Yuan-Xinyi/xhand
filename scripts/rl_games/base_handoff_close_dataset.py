#!/usr/bin/env python3
"""Collect or evaluate an online base -> close -> lift option chain.

Each environment runs the frozen 115/21 base actor until a debounced, measured pregrasp gate opens,
then switches in-place to a close teacher or actor.  A stable close can be verified by a scripted
micro-lift, handed to a scripted lift teacher, or handed to a learned lift actor.  No state is ever
restored, so velocities, actuator targets, contact history, latch counters and the observable
previous action remain continuous.  Serialized close/lift rows come only from trajectories that
subsequently satisfy the selected load-bearing success contract.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base_checkpoint", required=True)
parser.add_argument(
    "--close_checkpoint",
    default=None,
    help="optional 115/21 close actor for online evaluation; omitted means collect teacher labels",
)
parser.add_argument(
    "--rollout_close_checkpoint",
    default=None,
    help="optional learner used to visit DAgger close states while stored actions remain teacher labels",
)
parser.add_argument(
    "--lift_checkpoint",
    default=None,
    help="optional 115/21 lift-option actor used after a stable close",
)
parser.add_argument("--teacher_probability", type=float, default=1.0)
parser.add_argument("--feasibility_json", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seed", type=int, default=130)
parser.add_argument("--approach_max_steps", type=int, default=500)
parser.add_argument("--handoff_score", type=float, default=0.25)
parser.add_argument("--handoff_hold_steps", type=int, default=4)
parser.add_argument(
    "--handoff_min_step",
    type=int,
    default=0,
    help="do not transfer control before this zero-based base-policy step",
)
parser.add_argument("--handoff_min_proximity", type=float, default=0.02)
parser.add_argument("--close_max_steps", type=int, default=240)
parser.add_argument("--close_arm_mode", choices=("zero", "base", "dls"), default="zero")
parser.add_argument("--hand_action_step", type=float, default=0.04)
parser.add_argument("--grip_force_target", type=float, default=3.0)
parser.add_argument("--grip_force_limit", type=float, default=20.0)
parser.add_argument("--grip_servo_step", type=float, default=0.006)
parser.add_argument("--grip_servo_range", type=float, default=0.60)
parser.add_argument("--verify_height", type=float, default=0.07)
parser.add_argument("--verify_clearance", type=float, default=0.05)
parser.add_argument("--verify_hand_mode", choices=("hold", "servo"), default="hold")
parser.add_argument("--verify_ramp_steps", type=int, default=50)
parser.add_argument("--verify_hold_steps", type=int, default=5)
parser.add_argument("--verify_max_steps", type=int, default=100)
parser.add_argument("--verify_loss_steps", type=int, default=4)
parser.add_argument(
    "--post_close_mode",
    choices=(
        "verify_only",
        "base_direct",
        "verify_then_base",
        "scripted_lift",
        "learned_lift",
    ),
    default="verify_only",
)
parser.add_argument("--post_base_max_steps", type=int, default=500)
parser.add_argument("--post_base_stable_steps", type=int, default=15)
parser.add_argument("--post_base_loss_steps", type=int, default=30)
parser.add_argument("--scripted_lift_height", type=float, default=0.24)
parser.add_argument("--scripted_lift_ramp_steps", type=int, default=240)
parser.add_argument(
    "--learned_lift_hand_mode",
    choices=("actor", "hold"),
    default="actor",
)
parser.add_argument(
    "--learned_lift_stop_clearance",
    type=float,
    default=0.0,
    help="zero learned lift arm increments at this true clearance; 0 disables the ceiling",
)
parser.add_argument(
    "--post_hand_mode",
    choices=("base", "close", "hold", "servo"),
    default="base",
    help="hand controller after close succeeds (scripted lift only overrides the arm)",
)
parser.add_argument("--damping", type=float, default=0.05)
parser.add_argument("--max_cart_step", type=float, default=0.004)
parser.add_argument("--max_rot_step", type=float, default=0.05)
parser.add_argument("--max_joint_step", type=float, default=0.04)
parser.add_argument("--min_successes", type=int, default=1)
parser.add_argument(
    "--dataset_phase",
    choices=("close", "lift"),
    default="close",
    help="successful option rows to serialize; lift requires scripted_lift",
)
parser.add_argument("--output", default="/tmp/pick_tool_base_handoff_close.pt")
parser.add_argument("--metrics", default="/tmp/pick_tool_base_handoff_close.json")
parser.add_argument(
    "--transition_output",
    default=None,
    help=(
        "optional FlashSAC one-step transition dataset; uses the normal task horizon/termination "
        "and retains only strict, latched 20 cm successes"
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 1 or args_cli.min_successes < 0:
    parser.error("--num_envs must be positive and --min_successes non-negative")
finite_values = {
    "handoff_score": args_cli.handoff_score,
    "handoff_min_proximity": args_cli.handoff_min_proximity,
    "teacher_probability": args_cli.teacher_probability,
    "hand_action_step": args_cli.hand_action_step,
    "grip_force_target": args_cli.grip_force_target,
    "grip_force_limit": args_cli.grip_force_limit,
    "grip_servo_step": args_cli.grip_servo_step,
    "grip_servo_range": args_cli.grip_servo_range,
    "verify_height": args_cli.verify_height,
    "verify_clearance": args_cli.verify_clearance,
    "scripted_lift_height": args_cli.scripted_lift_height,
    "learned_lift_stop_clearance": args_cli.learned_lift_stop_clearance,
    "damping": args_cli.damping,
    "max_cart_step": args_cli.max_cart_step,
    "max_rot_step": args_cli.max_rot_step,
    "max_joint_step": args_cli.max_joint_step,
}
not_finite = [name for name, value in finite_values.items() if not math.isfinite(value)]
if not_finite:
    parser.error(f"non-finite numeric arguments: {', '.join(not_finite)}")
if not 0 <= args_cli.handoff_min_step < args_cli.approach_max_steps:
    parser.error("--handoff_min_step must be in [0, approach_max_steps)")
if not 0.0 <= args_cli.handoff_score <= 1.0:
    parser.error("--handoff_score must be in [0, 1]")
if not 0.0 <= args_cli.handoff_min_proximity <= 1.0:
    parser.error("--handoff_min_proximity must be in [0, 1]")
for value in (
    args_cli.approach_max_steps,
    args_cli.handoff_hold_steps,
    args_cli.close_max_steps,
    args_cli.verify_ramp_steps,
    args_cli.verify_hold_steps,
    args_cli.verify_max_steps,
    args_cli.verify_loss_steps,
    args_cli.post_base_max_steps,
    args_cli.post_base_stable_steps,
    args_cli.post_base_loss_steps,
    args_cli.scripted_lift_ramp_steps,
):
    if value < 1:
        parser.error("step counts must be positive")
if (
    args_cli.hand_action_step <= 0.0
    or args_cli.grip_servo_step <= 0.0
    or args_cli.grip_servo_range <= 0.0
):
    parser.error("hand step, grip servo step and grip servo range must be positive")
if not 0.0 <= args_cli.grip_force_target < args_cli.grip_force_limit:
    parser.error("require 0 <= grip_force_target < grip_force_limit")
if args_cli.verify_height <= args_cli.verify_clearance or args_cli.verify_clearance <= 0.0:
    parser.error("verify height must exceed a positive verify clearance")
if not 0.0 <= args_cli.teacher_probability <= 1.0:
    parser.error("--teacher_probability must be in [0, 1]")
if args_cli.close_checkpoint and args_cli.rollout_close_checkpoint:
    parser.error("--close_checkpoint and --rollout_close_checkpoint are mutually exclusive")
if not args_cli.rollout_close_checkpoint and args_cli.teacher_probability != 1.0:
    parser.error("--teacher_probability differs from 1 but no rollout close checkpoint was supplied")
if (
    args_cli.post_close_mode not in ("verify_only", "scripted_lift")
    and not args_cli.close_checkpoint
):
    parser.error("post-base integration modes require --close_checkpoint")
if args_cli.scripted_lift_height < 0.20:
    parser.error("--scripted_lift_height must be at least 0.20m")
if min(
    args_cli.damping,
    args_cli.max_cart_step,
    args_cli.max_rot_step,
    args_cli.max_joint_step,
) <= 0.0:
    parser.error("DLS damping and step limits must be positive")
if args_cli.dataset_phase == "lift" and args_cli.post_close_mode != "scripted_lift":
    parser.error("--dataset_phase lift requires --post_close_mode scripted_lift")
if args_cli.post_close_mode == "learned_lift" and not args_cli.lift_checkpoint:
    parser.error("--post_close_mode learned_lift requires --lift_checkpoint")
if args_cli.lift_checkpoint and args_cli.post_close_mode != "learned_lift":
    parser.error("--lift_checkpoint requires --post_close_mode learned_lift")
if args_cli.transition_output and args_cli.post_close_mode == "verify_only":
    parser.error("--transition_output requires a full 20 cm post-close mode")
if args_cli.post_hand_mode == "close" and not args_cli.close_checkpoint:
    parser.error("--post_hand_mode close requires --close_checkpoint")
if (
    args_cli.learned_lift_stop_clearance != 0.0
    and args_cli.learned_lift_stop_clearance < 0.20
):
    parser.error("--learned_lift_stop_clearance must be 0 or at least 0.20m")
if (
    args_cli.learned_lift_stop_clearance != 0.0
    and args_cli.post_close_mode != "learned_lift"
):
    parser.error("--learned_lift_stop_clearance is only valid with learned_lift")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

# Sibling module (scripts/rl_games is sys.path[0] for direct-script execution): single source of
# truth for the pregrasp score, boundary schema and force constants shared with the evaluator.
from pick_tool_shared import capture_boundary, limit_norm, pregrasp_score, sha256

from isaaclab.utils.math import compute_pose_error, quat_apply
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401
from bc_pick_tool import MigratedActor, clone_state, load_torch
from xhand_inhand.tasks.direct.pick_tool_token.grasp_signals import update_close_option_state
from xhand_inhand.tasks.direct.pick_tool_token.hybrid_action import (
    apply_asymmetric_joint_residual,
    invert_asymmetric_joint_residual,
)

_FLASHSAC_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "flashsac"
if str(_FLASHSAC_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_FLASHSAC_SCRIPT_DIR))
from adapter import PickToolIsaacLabAdapter, build_replay_transition  # noqa: E402


SEARCH, CLOSE, VERIFY, POST_BASE, SUCCESS, FAILURE = range(6)
PHASE_CLOSE = 1
PHASE_LIFT = 3


# Bound to the shared implementations so collector and evaluator cannot drift apart.
_sha256 = sha256
_limit_norm = limit_norm
_pregrasp_score = pregrasp_score
_capture_boundary = capture_boundary


def _checkpoint_model(path: Path) -> dict[str, torch.Tensor]:
    raw = load_torch(path)
    if not isinstance(raw, dict):
        raise TypeError("checkpoint root is not a dictionary")
    if isinstance(raw.get("model"), dict):
        payload = raw
    else:
        payload = raw[0] if 0 in raw else raw.get("0")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise KeyError("checkpoint must contain {'model': state_dict}, optionally below root key 0")
    return clone_state(payload["model"])


def _object_com_position_w(u) -> torch.Tensor:
    return u.object.data.root_link_pos_w + quat_apply(
        u.object.data.root_link_quat_w,
        u.object.data.body_com_pos_b[:, 0],
    )


def _summary(value: torch.Tensor) -> dict[str, float] | None:
    flat = value.detach().float().flatten()
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return None
    quantiles = torch.quantile(
        flat, torch.tensor((0.0, 0.1, 0.5, 0.9, 1.0), device=flat.device)
    )
    return dict(
        zip(("min", "p10", "median", "p90", "max"), (float(x) for x in quantiles), strict=True)
    )


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(args_cli.seed)
    checkpoint_path = Path(args_cli.base_checkpoint)
    artifact_path = Path(args_cli.feasibility_json)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    hybrid = artifact["results"]["hybrid14"]
    if not hybrid.get("robust_grasp_pass", False):
        raise RuntimeError("the selected hybrid14 feasibility result is not a robust grasp")

    actor = MigratedActor(_checkpoint_model(checkpoint_path)).to(args_cli.device).eval()
    if actor.observation_dim != 115 or actor.action_dim != 21:
        raise RuntimeError(
            f"base actor must use 115 observations and 21 actions, got "
            f"{actor.observation_dim}/{actor.action_dim}"
        )
    close_checkpoint_path = Path(args_cli.close_checkpoint) if args_cli.close_checkpoint else None
    close_actor = None
    if close_checkpoint_path is not None:
        close_actor = MigratedActor(_checkpoint_model(close_checkpoint_path)).to(args_cli.device).eval()
        if close_actor.observation_dim != 115 or close_actor.action_dim != 21:
            raise RuntimeError(
                f"close actor must use 115 observations and 21 actions, got "
                f"{close_actor.observation_dim}/{close_actor.action_dim}"
            )
    lift_checkpoint_path = Path(args_cli.lift_checkpoint) if args_cli.lift_checkpoint else None
    lift_actor = None
    if lift_checkpoint_path is not None:
        lift_actor = MigratedActor(_checkpoint_model(lift_checkpoint_path)).to(args_cli.device).eval()
        if lift_actor.observation_dim != 115 or lift_actor.action_dim != 21:
            raise RuntimeError(
                f"lift actor must use 115 observations and 21 actions, got "
                f"{lift_actor.observation_dim}/{lift_actor.action_dim}"
            )
    rollout_checkpoint_path = (
        Path(args_cli.rollout_close_checkpoint) if args_cli.rollout_close_checkpoint else None
    )
    rollout_close_actor = None
    if rollout_checkpoint_path is not None:
        rollout_close_actor = MigratedActor(
            _checkpoint_model(rollout_checkpoint_path)
        ).to(args_cli.device).eval()
        if rollout_close_actor.observation_dim != 115 or rollout_close_actor.action_dim != 21:
            raise RuntimeError(
                f"DAgger rollout close actor must use 115 observations and 21 actions, got "
                f"{rollout_close_actor.observation_dim}/{rollout_close_actor.action_dim}"
            )

    cfg = parse_env_cfg(
        "Pick-Tool-Token-Direct-v0", device=args_cli.device, num_envs=args_cli.num_envs
    )
    cfg.seed = args_cli.seed
    total_max_steps = (
        args_cli.approach_max_steps + args_cli.close_max_steps + args_cli.verify_max_steps
    )
    if args_cli.post_close_mode != "verify_only":
        total_max_steps += args_cli.post_base_max_steps
    if args_cli.transition_output is None:
        # The option-label collector measures terminal states itself.  A reset
        # would splice a new episode into one option row.  FlashSAC transition
        # collection takes the other branch and intentionally keeps the target
        # MDP's normal 20 s/drop/force/success semantics.
        cfg.episode_length_s = 120.0
        cfg.terminate_on_drop = False
        cfg.success_hold_steps = 100000
        cfg.tactile_hard_terminate_steps = total_max_steps + 1
        cfg.tactile_terminate_steps = total_max_steps + 1
    env = gym.make("Pick-Tool-Token-Direct-v0", cfg=cfg)
    u = env.unwrapped
    transition_adapter = None
    if args_cli.transition_output is None:
        obs, _ = env.reset()
    else:
        transition_adapter = PickToolIsaacLabAdapter(
            env,
            strict=True,
            require_cuda=True,
            validate_finite=True,
        )
        policy_obs, _ = transition_adapter.reset()
        obs = {"policy": policy_obs}
    dev = u.device
    n = u.num_envs

    hand_names = [u.robot.joint_names[index] for index in u._hand_ids_t.tolist()]
    if hand_names != artifact["hand_joint_names"]:
        raise RuntimeError("artifact hand-joint order differs from the runtime articulation")
    hand_lower = u.dof_lower[:, u._hand_ids_t]
    hand_upper = u.dof_upper[:, u._hand_ids_t]
    latent = torch.tensor(hybrid["latent"], device=dev).repeat(n, 1)
    token = latent[:, : u._n_tokens]
    token_base = u.retarget.retarget_from_unit_action(token)[:, u._retarget2isaac]
    token_base = torch.maximum(torch.minimum(token_base, hand_upper), hand_lower)
    expected_target = torch.tensor(hybrid["target"], device=dev).repeat(n, 1)
    decoded_target, _ = apply_asymmetric_joint_residual(
        token_base,
        hand_lower,
        hand_upper,
        latent[:, u._n_tokens :],
        u._distal_hand_ids,
    )
    if float((decoded_target - expected_target).abs().max()) >= 1.0e-6:
        raise RuntimeError("hybrid artifact no longer decodes to its saved hand target")

    servo_lower = expected_target.index_select(1, u._distal_hand_ids)
    servo_upper = torch.minimum(
        servo_lower + args_cli.grip_servo_range,
        hand_upper.index_select(1, u._distal_hand_ids),
    )
    # Reuse the env's fingertip->distal routing rather than re-deriving it, so the servo cannot
    # drift from the shield the deployment env applies.
    force_to_distal = u._force_to_distal_index.tolist()

    def servo_hand_action(force: torch.Tensor) -> torch.Tensor:
        previous = u.actions[:, u._n_arm :]
        previous_token_target = u.retarget.retarget_from_unit_action(
            previous[:, : u._n_tokens]
        )[:, u._retarget2isaac]
        previous_token_target = torch.maximum(
            torch.minimum(previous_token_target, hand_upper), hand_lower
        )
        previous_target, _ = apply_asymmetric_joint_residual(
            previous_token_target,
            hand_lower,
            hand_upper,
            previous[:, u._n_tokens :],
            u._distal_hand_ids,
        )
        previous_distal = previous_target.index_select(1, u._distal_hand_ids)
        delta = torch.zeros_like(previous_distal)
        for force_index, distal_index in enumerate(force_to_distal):
            delta[:, distal_index] = args_cli.grip_servo_step * (
                (force[:, force_index] < args_cli.grip_force_target).float()
                - (force[:, force_index] > args_cli.grip_force_limit).float()
            )
        desired_distal = torch.maximum(
            torch.minimum(previous_distal + delta, servo_upper), servo_lower
        )
        desired = token_base.clone()
        desired.index_copy_(1, u._distal_hand_ids, desired_distal)
        residual = invert_asymmetric_joint_residual(
            desired, token_base, hand_lower, hand_upper, u._distal_hand_ids
        )
        return torch.cat((token, residual), dim=-1).clamp(-1.0, 1.0)

    eye6 = torch.eye(6, device=dev).unsqueeze(0)

    def dls_pose_action(desired_pos: torch.Tensor, desired_quat: torch.Tensor) -> torch.Tensor:
        u._compute_intermediate_values()
        pos_error, rot_error = compute_pose_error(
            u.robot.data.body_pos_w[:, u.palm_idx],
            u.robot.data.body_quat_w[:, u.palm_idx],
            desired_pos,
            desired_quat,
            rot_error_type="axis_angle",
        )
        delta = torch.cat(
            (
                _limit_norm(pos_error, args_cli.max_cart_step),
                _limit_norm(rot_error, args_cli.max_rot_step),
            ),
            dim=-1,
        )
        jacobian = u.robot.root_physx_view.get_jacobians()
        jacobian = jacobian[:, u._palm_jac_idx, :, :][:, :, u._arm_ids_t]
        jt = jacobian.transpose(1, 2)
        system = jacobian @ jt + (args_cli.damping**2) * eye6
        delta_q = (jt @ torch.linalg.solve(system, delta.unsqueeze(-1))).squeeze(-1)
        delta_q = delta_q.clamp(-args_cli.max_joint_step, args_cli.max_joint_step)
        return (delta_q / (cfg.act_moving_average * cfg.action_scale)).clamp(-1.0, 1.0)

    state = torch.full((n,), SEARCH, dtype=torch.long, device=dev)
    ready_steps = torch.zeros(n, dtype=torch.long, device=dev)
    close_age = torch.zeros(n, dtype=torch.long, device=dev)
    verify_age = torch.zeros(n, dtype=torch.long, device=dev)
    post_base_age = torch.zeros(n, dtype=torch.long, device=dev)
    close_stable_steps = torch.zeros(n, dtype=torch.long, device=dev)
    close_lost_steps = torch.zeros(n, dtype=torch.long, device=dev)
    verify_stable_steps = torch.zeros(n, dtype=torch.long, device=dev)
    verify_lost_steps = torch.zeros(n, dtype=torch.long, device=dev)
    post_base_stable_steps = torch.zeros(n, dtype=torch.long, device=dev)
    post_base_lost_steps = torch.zeros(n, dtype=torch.long, device=dev)
    base_only_stable_steps = torch.zeros(n, dtype=torch.long, device=dev)
    overforce_steps = torch.zeros(n, dtype=torch.long, device=dev)
    hard_force_steps = torch.zeros(n, dtype=torch.long, device=dev)
    close_start_xy = torch.zeros((n, 2), device=dev)
    verify_anchor_pos = torch.zeros((n, 3), device=dev)
    verify_anchor_quat = torch.zeros((n, 4), device=dev)
    verify_anchor_quat[:, 0] = 1.0
    verify_hand_action = torch.zeros((n, u._n_tokens + u._n_distal_residuals), device=dev)
    post_hand_action = torch.zeros((n, u._n_tokens + u._n_distal_residuals), device=dev)
    close_anchor_pos = torch.zeros((n, 3), device=dev)
    close_anchor_quat = torch.zeros((n, 4), device=dev)
    close_anchor_quat[:, 0] = 1.0
    lift_anchor_pos = torch.zeros((n, 3), device=dev)
    lift_anchor_quat = torch.zeros((n, 4), device=dev)
    lift_anchor_quat[:, 0] = 1.0

    handoff_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    handoff_score = torch.zeros(n, device=dev)
    handoff_proximity = torch.zeros(n, device=dev)
    close_success_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    verify_success_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    final_success_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    base_only_success_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    max_verify_clearance = torch.full((n,), -float("inf"), device=dev)
    max_post_base_clearance = torch.full((n,), -float("inf"), device=dev)
    force_peak = torch.zeros(n, device=dev)
    force_peak_per_finger = torch.zeros((n, len(u.ee_names)), device=dev)
    max_close_stable_steps = torch.zeros(n, dtype=torch.long, device=dev)
    max_close_quality = torch.zeros(n, device=dev)
    max_wrap_quality = torch.zeros(n, device=dev)
    max_grasp_quality = torch.zeros(n, device=dev)
    max_hold_quality = torch.zeros(n, device=dev)
    max_other_contact_count = torch.zeros(n, dtype=torch.long, device=dev)
    ever_thumb_contact = torch.zeros(n, dtype=torch.bool, device=dev)
    ever_valid_contact = torch.zeros(n, dtype=torch.bool, device=dev)
    ever_latch = torch.zeros(n, dtype=torch.bool, device=dev)
    min_hand_action_distance = torch.full((n,), float("inf"), device=dev)
    max_horizontal_drift = torch.zeros(n, device=dev)

    close_failure = torch.zeros(n, dtype=torch.bool, device=dev)
    close_timeout = torch.zeros(n, dtype=torch.bool, device=dev)
    close_unlatched_lift = torch.zeros(n, dtype=torch.bool, device=dev)
    close_horizontal_escape = torch.zeros(n, dtype=torch.bool, device=dev)
    close_lost_window = torch.zeros(n, dtype=torch.bool, device=dev)
    close_unsafe = torch.zeros(n, dtype=torch.bool, device=dev)
    search_timeout = torch.zeros(n, dtype=torch.bool, device=dev)
    verify_timeout = torch.zeros(n, dtype=torch.bool, device=dev)
    verify_lost = torch.zeros(n, dtype=torch.bool, device=dev)
    verify_unsafe = torch.zeros(n, dtype=torch.bool, device=dev)
    post_base_timeout = torch.zeros(n, dtype=torch.bool, device=dev)
    post_base_lost = torch.zeros(n, dtype=torch.bool, device=dev)
    post_base_unsafe = torch.zeros(n, dtype=torch.bool, device=dev)
    post_base_fling = torch.zeros(n, dtype=torch.bool, device=dev)
    post_unlatched_5cm = torch.zeros(n, dtype=torch.bool, device=dev)
    post_unlatched_20cm = torch.zeros(n, dtype=torch.bool, device=dev)
    environment_failure = torch.zeros(n, dtype=torch.bool, device=dev)

    initial_boundary = _capture_boundary(u)
    close_start = {key: torch.zeros_like(value) for key, value in initial_boundary.items()}
    lift_start = {key: torch.zeros_like(value) for key, value in initial_boundary.items()}
    handoff_obs = torch.zeros((n, 115), device=dev)

    record_env: list[torch.Tensor] = []
    record_obs: list[torch.Tensor] = []
    record_action: list[torch.Tensor] = []
    record_step: list[torch.Tensor] = []
    executed_teacher_rows = 0
    executed_close_rows = 0

    # Full MDP transition rows are recorded step-major and filtered by strict
    # final outcome only after every parallel trajectory is complete.  The
    # adapter supplies reset-before terminal observations for done rows.
    transition_open = torch.ones(n, dtype=torch.bool, device=dev)
    transition_unlatched_seen = torch.zeros(n, dtype=torch.bool, device=dev)
    transition_strict_success = torch.zeros(n, dtype=torch.bool, device=dev)
    transition_assisted_success = torch.zeros(n, dtype=torch.bool, device=dev)
    transition_terminal_snapshot = {
        "true_clearance": torch.full((n,), float("nan"), device=dev),
        "is_grasped": torch.zeros(n, dtype=torch.bool, device=dev),
        "grasp_quality": torch.full((n,), float("nan"), device=dev),
        "hold_quality": torch.full((n,), float("nan"), device=dev),
        "max_force": torch.full((n,), float("nan"), device=dev),
        "object_lin_speed": torch.full((n,), float("nan"), device=dev),
        "object_ang_speed": torch.full((n,), float("nan"), device=dev),
        "success_steps": torch.zeros(n, dtype=torch.long, device=dev),
    }
    transition_env: list[torch.Tensor] = []
    transition_observation: list[torch.Tensor] = []
    transition_action: list[torch.Tensor] = []
    transition_reward: list[torch.Tensor] = []
    transition_next_observation: list[torch.Tensor] = []
    transition_terminated: list[torch.Tensor] = []
    transition_truncated: list[torch.Tensor] = []
    transition_phase: list[torch.Tensor] = []
    transition_step: list[torch.Tensor] = []

    print(
        f"online base->close collector: envs={n} score>={args_cli.handoff_score:.3f} "
        f"x{args_cli.handoff_hold_steps} after step {args_cli.handoff_min_step}, "
        f"close stable={cfg.close_option_confirm_steps}, "
        f"verify clearance={args_cli.verify_clearance:.3f}m",
        flush=True,
    )
    for global_step in range(total_max_steps):
        active = (
            (state == SEARCH) | (state == CLOSE) | (state == VERIFY) | (state == POST_BASE)
        )
        if not bool(active.any()):
            break
        policy_obs = obs["policy"]
        if policy_obs.shape != (n, 115):
            raise RuntimeError(f"environment observation changed: got {tuple(policy_obs.shape)}")
        u._compute_intermediate_values()
        signals_before = u._compute_grasp_signals()
        force_before = signals_before["force_magnitude"]
        force_max_before = force_before.max(dim=-1).values
        clearance_before = u._object_true_min_z() - u._table_surface_z
        score = _pregrasp_score(u)
        searching = state == SEARCH
        ready = (
            searching
            & (global_step >= args_cli.handoff_min_step)
            & ~u._is_grasped
            & (score >= args_cli.handoff_score)
            & (signals_before["proximity_quality"] >= args_cli.handoff_min_proximity)
            & (force_max_before <= cfg.grasp_bonus_max_force)
        )
        ready_steps = torch.where(ready, ready_steps + 1, torch.zeros_like(ready_steps))
        enter_close = searching & (ready_steps >= args_cli.handoff_hold_steps)
        if bool(enter_close.any()):
            boundary = _capture_boundary(u)
            for key in close_start:
                close_start[key][enter_close] = boundary[key][enter_close]
            handoff_obs[enter_close] = policy_obs[enter_close]
            handoff_step[enter_close] = global_step
            handoff_score[enter_close] = score[enter_close]
            handoff_proximity[enter_close] = signals_before["proximity_quality"][enter_close]
            close_start_xy[enter_close] = _object_com_position_w(u)[enter_close, :2]
            close_anchor_pos[enter_close] = u.robot.data.body_pos_w[enter_close, u.palm_idx]
            close_anchor_quat[enter_close] = u.robot.data.body_quat_w[enter_close, u.palm_idx]
            close_age[enter_close] = 0
            close_stable_steps[enter_close] = 0
            close_lost_steps[enter_close] = 0
            state[enter_close] = CLOSE

        close_mask = state == CLOSE
        verify_mask = state == VERIFY
        post_base_mask = state == POST_BASE
        base_search_mask = state == SEARCH
        action = actor(policy_obs).clamp(-1.0, 1.0)
        previous_hand = u.actions[:, u._n_arm :]
        ramped_hand = previous_hand + torch.clamp(
            latent - previous_hand,
            -args_cli.hand_action_step,
            args_cli.hand_action_step,
        )
        servo_hand = servo_hand_action(force_before)
        servo_ready = (latent - previous_hand).abs().max(dim=-1).values <= args_cli.hand_action_step
        close_hand = torch.where(servo_ready.unsqueeze(-1), servo_hand, ramped_hand)
        if close_actor is None:
            if args_cli.close_arm_mode == "zero":
                action[close_mask, : u._n_arm] = 0.0
            elif args_cli.close_arm_mode == "dls" and bool(close_mask.any()):
                close_arm = dls_pose_action(close_anchor_pos, close_anchor_quat)
                action[close_mask, : u._n_arm] = close_arm[close_mask]
            action[close_mask, u._n_arm :] = close_hand[close_mask]
        teacher_action = action.clone()
        close_policy_mask = close_mask | (
            post_base_mask & (args_cli.post_hand_mode == "close")
        )
        if close_actor is not None and bool(close_policy_mask.any()):
            learner_close_action = close_actor(policy_obs).clamp(-1.0, 1.0)
            action[close_mask] = learner_close_action[close_mask]
            if args_cli.post_hand_mode == "close":
                action[post_base_mask, u._n_arm :] = learner_close_action[
                    post_base_mask, u._n_arm :
                ]
        elif rollout_close_actor is not None and bool(close_mask.any()):
            learner_close_action = rollout_close_actor(policy_obs).clamp(-1.0, 1.0)
            use_teacher = (
                torch.rand(n, device=dev) < args_cli.teacher_probability
            ) & close_mask
            use_learner = close_mask & ~use_teacher
            action[use_learner] = learner_close_action[use_learner]
            executed_teacher_rows += int(use_teacher.sum())
            executed_close_rows += int(close_mask.sum())

        if args_cli.post_hand_mode == "hold" and bool(post_base_mask.any()):
            action[post_base_mask, u._n_arm :] = post_hand_action[post_base_mask]
        elif args_cli.post_hand_mode == "servo" and bool(post_base_mask.any()):
            action[post_base_mask, u._n_arm :] = servo_hand[post_base_mask]

        if args_cli.post_close_mode == "learned_lift" and bool(post_base_mask.any()):
            learned_lift_action = lift_actor(policy_obs).clamp(-1.0, 1.0)
            action[post_base_mask] = learned_lift_action[post_base_mask]
            if args_cli.learned_lift_hand_mode == "hold":
                action[post_base_mask, u._n_arm :] = post_hand_action[post_base_mask]
            if args_cli.learned_lift_stop_clearance > 0.0:
                stop_lift = post_base_mask & (
                    clearance_before >= args_cli.learned_lift_stop_clearance
                )
                action[stop_lift, : u._n_arm] = 0.0

        if args_cli.post_close_mode == "scripted_lift" and bool(post_base_mask.any()):
            lift_fraction = torch.clamp(
                (post_base_age.float() + 1.0) / args_cli.scripted_lift_ramp_steps,
                0.0,
                1.0,
            )
            lift_pos = lift_anchor_pos.clone()
            lift_pos[:, 2] += args_cli.scripted_lift_height * lift_fraction
            lift_arm = dls_pose_action(lift_pos, lift_anchor_quat)
            action[post_base_mask, : u._n_arm] = lift_arm[post_base_mask]

        if bool(verify_mask.any()):
            verify_fraction = torch.clamp(
                (verify_age.float() + 1.0) / args_cli.verify_ramp_steps, 0.0, 1.0
            )
            desired_pos = verify_anchor_pos.clone()
            desired_pos[:, 2] += args_cli.verify_height * verify_fraction
            verify_arm = dls_pose_action(desired_pos, verify_anchor_quat)
            action[verify_mask, : u._n_arm] = verify_arm[verify_mask]
            if args_cli.verify_hand_mode == "hold":
                action[verify_mask, u._n_arm :] = verify_hand_action[verify_mask]
            else:
                action[verify_mask, u._n_arm :] = servo_hand[verify_mask]

        terminal = (state == SUCCESS) | (state == FAILURE)
        action[terminal] = 0.0
        record_mask = torch.zeros_like(close_mask)
        record_source_action = action
        if args_cli.dataset_phase == "close" and close_actor is None:
            record_mask = close_mask
            record_source_action = teacher_action
        elif args_cli.dataset_phase == "lift" and args_cli.post_close_mode == "scripted_lift":
            record_mask = post_base_mask
        if bool(record_mask.any()):
            ids = record_mask.nonzero(as_tuple=False).squeeze(-1)
            record_env.append(ids.cpu())
            record_obs.append(policy_obs[ids].clamp(-5.0, 5.0).cpu())
            record_action.append(record_source_action[ids].cpu())
            option_age = close_age if args_cli.dataset_phase == "close" else post_base_age
            record_step.append(option_age[ids].cpu())
            if (
                args_cli.dataset_phase == "close"
                and args_cli.close_arm_mode == "zero"
                and float(record_source_action[ids, : u._n_arm].abs().max()) != 0.0
            ):
                raise RuntimeError("close teacher emitted a non-zero arm action")

        transition_mask = active & transition_open
        transition_ids = transition_mask.nonzero(as_tuple=False).squeeze(-1)
        if transition_adapter is None:
            obs, reward, terminated, truncated, step_info = env.step(action)
            transition_values = None
            terminal_failure_now = active & (terminated | truncated)
        else:
            next_policy_obs, reward, terminated, truncated, step_info = transition_adapter.step(action)
            replay_transition = build_replay_transition(
                policy_obs,
                action,
                reward,
                terminated,
                truncated,
                step_info,
            )
            if transition_ids.numel() > 0:
                transition_env.append(transition_ids.cpu())
                transition_observation.append(
                    replay_transition["observation"][transition_ids].cpu()
                )
                transition_action.append(replay_transition["action"][transition_ids].cpu())
                transition_reward.append(replay_transition["reward"][transition_ids].cpu())
                transition_next_observation.append(
                    replay_transition["next_observation"][transition_ids].cpu()
                )
                transition_terminated.append(
                    replay_transition["terminated"][transition_ids].cpu()
                )
                transition_truncated.append(
                    replay_transition["truncated"][transition_ids].cpu()
                )
                transition_phase.append(state[transition_ids].to(torch.uint8).cpu())
                transition_step.append(
                    torch.full(
                        (transition_ids.numel(),),
                        global_step,
                        dtype=torch.int32,
                    )
                )
            transition_values = step_info.get("pick_tool_terminal")
            if not isinstance(transition_values, dict):
                raise KeyError("adapter step has no pick_tool_terminal ground truth")
            required_terminal_values = (
                "success",
                "failure",
                "time_out",
                "unlatched_clearance_ge_5cm",
                "true_clearance",
                "is_grasped",
                "grasp_quality",
                "hold_quality",
                "max_force",
                "object_lin_speed",
                "object_ang_speed",
                "success_steps",
            )
            for name in required_terminal_values:
                value = transition_values.get(name)
                if not isinstance(value, torch.Tensor) or value.shape != (n,):
                    raise RuntimeError(
                        f"pick_tool_terminal[{name!r}] must be a tensor with shape ({n},)"
                    )
            transition_unlatched_seen |= transition_mask & transition_values[
                "unlatched_clearance_ge_5cm"
            ]
            done_now = terminated | truncated
            strict_success_now = (
                transition_mask
                & done_now
                & transition_values["success"]
                & ~transition_values["failure"]
                & ~transition_values["time_out"]
            )
            terminal_failure_now = transition_mask & done_now & ~strict_success_now
            assisted_success_now = strict_success_now & post_base_mask
            base_success_now = strict_success_now & base_search_mask
            transition_strict_success |= strict_success_now
            transition_assisted_success |= assisted_success_now
            for name, storage in transition_terminal_snapshot.items():
                storage[strict_success_now] = transition_values[name][strict_success_now]
            final_success_step[assisted_success_now] = global_step
            base_only_success_step[base_success_now] = global_step
            state[strict_success_now] = SUCCESS
            state[terminal_failure_now] = FAILURE
            transition_open &= ~done_now
            # Done rows have already auto-reset.  Their replay next observation
            # is correct, but reset-state physics must not enter the hierarchy's
            # post-step controller diagnostics or overwrite the terminal route.
            continuing = ~done_now
            active &= continuing
            searching &= continuing
            close_mask &= continuing
            verify_mask &= continuing
            post_base_mask &= continuing
            base_search_mask &= continuing
            obs = {"policy": next_policy_obs}
        done = terminated | truncated
        u._compute_intermediate_values()
        signals = u._compute_grasp_signals()
        force = signals["force_magnitude"]
        force_max = force.max(dim=-1).values
        clearance = u._object_true_min_z() - u._table_surface_z
        force_peak = torch.maximum(force_peak, torch.where(active, force_max, force_peak))
        force_peak_per_finger = torch.maximum(
            force_peak_per_finger,
            torch.where(close_mask.unsqueeze(-1), force, force_peak_per_finger),
        )
        # Mirror the env's full sustained-force safety contract (both cutoffs, cfg-driven), not
        # just the 60 N rule with a hard-coded count.  The tactile terminations are disabled during
        # collection, so a demo that violates either rule must be rejected here or it would be a
        # trajectory the deployment/eval env terminates but the dataset admits.
        overforce = force_max > cfg.tactile_terminate_force_limit
        overforce_steps = torch.where(
            active & overforce, overforce_steps + 1, torch.zeros_like(overforce_steps)
        )
        hard_force = force_max > cfg.tactile_hard_force_limit
        hard_force_steps = torch.where(
            active & hard_force, hard_force_steps + 1, torch.zeros_like(hard_force_steps)
        )
        unsafe = (
            (overforce_steps >= cfg.tactile_terminate_steps)
            | (hard_force_steps >= cfg.tactile_hard_terminate_steps)
            | terminal_failure_now
        )
        dropped = clearance < -0.03
        environment_failure |= terminal_failure_now
        search_environment_failure = searching & terminal_failure_now
        state[search_environment_failure] = FAILURE

        close_age[close_mask] += 1
        horizontal_drift = (_object_com_position_w(u)[:, :2] - close_start_xy).norm(dim=-1)
        max_horizontal_drift = torch.maximum(
            max_horizontal_drift,
            torch.where(close_mask, horizontal_drift, max_horizontal_drift),
        )
        max_close_quality = torch.maximum(
            max_close_quality,
            torch.where(close_mask, signals["close_quality"], max_close_quality),
        )
        max_wrap_quality = torch.maximum(
            max_wrap_quality,
            torch.where(close_mask, signals["quality"], max_wrap_quality),
        )
        max_grasp_quality = torch.maximum(
            max_grasp_quality,
            torch.where(close_mask, signals["grasp_quality"], max_grasp_quality),
        )
        max_hold_quality = torch.maximum(
            max_hold_quality,
            torch.where(close_mask, signals["hold_quality"], max_hold_quality),
        )
        max_other_contact_count = torch.maximum(
            max_other_contact_count,
            torch.where(close_mask, signals["other_contact_count"], max_other_contact_count),
        )
        ever_thumb_contact |= close_mask & signals["thumb_contact"]
        ever_valid_contact |= close_mask & signals["thumb_contact"] & (
            signals["other_contact_count"] >= 2
        )
        ever_latch |= close_mask & u._is_grasped
        min_hand_action_distance = torch.minimum(
            min_hand_action_distance,
            torch.where(
                close_mask,
                (u.actions[:, u._n_arm :] - latent).abs().max(dim=-1).values,
                min_hand_action_distance,
            ),
        )
        contract = update_close_option_state(
            signals["grasp_quality"],
            signals["hold_quality"],
            force_max,
            u._is_grasped,
            close_stable_steps,
            clearance,
            horizontal_drift,
            signals["proximity_quality"],
            close_lost_steps,
            unsafe | dropped,
            grasp_quality_threshold=cfg.grasp_quality_high,
            hold_quality_threshold=cfg.close_option_min_hold_quality,
            safe_force_limit=cfg.grasp_bonus_max_force,
            confirm_steps=cfg.close_option_confirm_steps,
            unlatched_lift_limit=cfg.close_option_unlatched_lift_limit,
            horizontal_drift_limit=cfg.close_option_horizontal_drift_limit,
            min_proximity=cfg.close_option_min_proximity,
            lost_window_steps=cfg.close_option_lost_window_steps,
        )
        close_stable_steps = torch.where(
            close_mask, contract["stable_count"], close_stable_steps
        )
        close_lost_steps = torch.where(
            close_mask, contract["lost_window_count"], close_lost_steps
        )
        max_close_stable_steps = torch.maximum(max_close_stable_steps, close_stable_steps)
        close_succeeded_now = close_mask & contract["success"]
        close_failed_now = close_mask & contract["failure"]
        close_success_step[close_succeeded_now] = global_step
        close_failure |= close_failed_now
        close_unlatched_lift |= close_failed_now & contract["unlatched_lift"]
        close_horizontal_escape |= close_failed_now & contract["horizontal_escape"]
        close_lost_window |= close_failed_now & contract["lost_window"]
        close_unsafe |= close_failed_now & (unsafe | dropped)
        state[close_failed_now] = FAILURE

        if bool(close_succeeded_now.any()) and args_cli.post_close_mode in (
            "verify_only",
            "verify_then_base",
        ):
            verify_anchor_pos[close_succeeded_now] = u.robot.data.body_pos_w[
                close_succeeded_now, u.palm_idx
            ]
            verify_anchor_quat[close_succeeded_now] = u.robot.data.body_quat_w[
                close_succeeded_now, u.palm_idx
            ]
            verify_hand_action[close_succeeded_now] = u.actions[
                close_succeeded_now, u._n_arm :
            ]
            verify_age[close_succeeded_now] = 0
            verify_stable_steps[close_succeeded_now] = 0
            verify_lost_steps[close_succeeded_now] = 0
            state[close_succeeded_now] = VERIFY
        elif bool(close_succeeded_now.any()):
            if args_cli.post_close_mode == "scripted_lift":
                lift_anchor_pos[close_succeeded_now] = u.robot.data.body_pos_w[
                    close_succeeded_now, u.palm_idx
                ]
                lift_anchor_quat[close_succeeded_now] = u.robot.data.body_quat_w[
                    close_succeeded_now, u.palm_idx
                ]
                boundary = _capture_boundary(u)
                for key in lift_start:
                    lift_start[key][close_succeeded_now] = boundary[key][close_succeeded_now]
            post_hand_action[close_succeeded_now] = u.actions[
                close_succeeded_now, u._n_arm :
            ]
            post_base_age[close_succeeded_now] = 0
            post_base_stable_steps[close_succeeded_now] = 0
            post_base_lost_steps[close_succeeded_now] = 0
            state[close_succeeded_now] = POST_BASE

        close_timed_out_now = (state == CLOSE) & (close_age >= args_cli.close_max_steps)
        close_timeout |= close_timed_out_now
        state[close_timed_out_now] = FAILURE

        verify_age[verify_mask] += 1
        max_verify_clearance = torch.maximum(
            max_verify_clearance,
            torch.where(verify_mask, clearance, max_verify_clearance),
        )
        verify_good = (
            verify_mask
            & u._is_grasped
            & (signals["grasp_quality"] >= cfg.grasp_quality_high)
            & (signals["hold_quality"] >= cfg.close_option_min_hold_quality)
            & (force_max <= cfg.grasp_bonus_max_force)
            & (clearance >= args_cli.verify_clearance)
        )
        verify_stable_steps = torch.where(
            verify_good, verify_stable_steps + 1, torch.zeros_like(verify_stable_steps)
        )
        verify_bad = verify_mask & (
            (~u._is_grasped)
            | (signals["grasp_quality"] < cfg.grasp_quality_low)
            | (signals["hold_quality"] < cfg.close_option_min_hold_quality)
        )
        verify_lost_steps = torch.where(
            verify_bad, verify_lost_steps + 1, torch.zeros_like(verify_lost_steps)
        )
        verify_succeeded_now = verify_mask & (
            verify_stable_steps >= args_cli.verify_hold_steps
        )
        verify_failed_now = verify_mask & (
            unsafe | dropped | (verify_lost_steps >= args_cli.verify_loss_steps)
        )
        verify_success_step[verify_succeeded_now] = global_step
        verify_lost |= verify_failed_now & (verify_lost_steps >= args_cli.verify_loss_steps)
        verify_unsafe |= verify_failed_now & (unsafe | dropped)
        if args_cli.post_close_mode == "verify_only":
            state[verify_succeeded_now] = SUCCESS
        else:
            post_hand_action[verify_succeeded_now] = u.actions[
                verify_succeeded_now, u._n_arm :
            ]
            post_base_age[verify_succeeded_now] = 0
            post_base_stable_steps[verify_succeeded_now] = 0
            post_base_lost_steps[verify_succeeded_now] = 0
            state[verify_succeeded_now] = POST_BASE
        state[verify_failed_now & ~verify_succeeded_now] = FAILURE
        verify_timed_out_now = (state == VERIFY) & (
            verify_age >= args_cli.verify_max_steps
        )
        verify_timeout |= verify_timed_out_now
        state[verify_timed_out_now] = FAILURE

        post_base_age[post_base_mask] += 1
        post_unlatched_5cm |= post_base_mask & ~u._is_grasped & (clearance >= 0.05)
        post_unlatched_20cm |= post_base_mask & ~u._is_grasped & (
            clearance >= cfg.lift_success_height
        )
        max_post_base_clearance = torch.maximum(
            max_post_base_clearance,
            torch.where(post_base_mask, clearance, max_post_base_clearance),
        )
        slow = (
            u.object.data.root_com_lin_vel_w.norm(dim=-1) < cfg.success_max_obj_lin_speed
        ) & (u.object.data.root_com_ang_vel_w.norm(dim=-1) < cfg.success_max_obj_ang_speed)
        scripted_lift_settled = torch.ones_like(post_base_mask)
        if args_cli.post_close_mode == "scripted_lift":
            scripted_lift_settled = post_base_age >= args_cli.scripted_lift_ramp_steps
        post_base_good = (
            post_base_mask
            & scripted_lift_settled
            & (clearance >= cfg.lift_success_height)
            & u._is_grasped
            & (signals["grasp_quality"] >= cfg.grasp_quality_high)
            & (signals["hold_quality"] >= cfg.close_option_min_hold_quality)
            & (force_max <= cfg.grasp_bonus_max_force)
            & slow
        )
        post_base_stable_steps = torch.where(
            post_base_good,
            post_base_stable_steps + 1,
            torch.zeros_like(post_base_stable_steps),
        )
        post_base_bad = post_base_mask & (
            (~u._is_grasped)
            | (signals["grasp_quality"] < cfg.grasp_quality_low)
            | (signals["hold_quality"] < cfg.close_option_min_hold_quality)
        )
        post_base_lost_steps = torch.where(
            post_base_bad,
            post_base_lost_steps + 1,
            torch.zeros_like(post_base_lost_steps),
        )
        post_base_succeeded_now = post_base_mask & (
            post_base_stable_steps >= args_cli.post_base_stable_steps
        )
        post_fling_now = post_base_mask & ~u._is_grasped & (clearance >= 0.05)
        post_base_failed_now = post_base_mask & (
            unsafe
            | dropped
            | post_fling_now
            | (post_base_lost_steps >= args_cli.post_base_loss_steps)
        )
        final_success_step[post_base_succeeded_now] = global_step
        post_base_lost |= post_base_failed_now & (
            post_base_lost_steps >= args_cli.post_base_loss_steps
        )
        post_base_unsafe |= post_base_failed_now & (unsafe | dropped)
        post_base_fling |= post_base_failed_now & post_fling_now
        state[post_base_succeeded_now] = SUCCESS
        state[post_base_failed_now & ~post_base_succeeded_now] = FAILURE
        post_base_timed_out_now = (state == POST_BASE) & (
            post_base_age >= args_cli.post_base_max_steps
        )
        post_base_timeout |= post_base_timed_out_now
        state[post_base_timed_out_now] = FAILURE

        if args_cli.post_close_mode != "verify_only" and close_actor is not None:
            base_only_good = (
                base_search_mask
                & (state == SEARCH)
                & (clearance >= cfg.lift_success_height)
                & u._is_grasped
                & (signals["grasp_quality"] >= cfg.grasp_quality_high)
                & (signals["hold_quality"] >= cfg.close_option_min_hold_quality)
                & (force_max <= cfg.grasp_bonus_max_force)
                & slow
            )
            base_only_stable_steps = torch.where(
                base_only_good,
                base_only_stable_steps + 1,
                torch.zeros_like(base_only_stable_steps),
            )
            base_only_succeeded_now = base_search_mask & (state == SEARCH) & (
                base_only_stable_steps >= args_cli.post_base_stable_steps
            )
            base_only_success_step[base_only_succeeded_now] = global_step
            state[base_only_succeeded_now] = SUCCESS

        search_timed_out_now = (state == SEARCH) & (
            global_step + 1 >= args_cli.approach_max_steps
        )
        search_timeout |= search_timed_out_now
        state[search_timed_out_now] = FAILURE

        if global_step % 100 == 0 or not bool(
            (
                (state == SEARCH)
                | (state == CLOSE)
                | (state == VERIFY)
                | (state == POST_BASE)
            ).any()
        ):
            print(
                f"step={global_step:4d} search/close/verify/post/success/failure="
                f"{[int((state == value).sum()) for value in range(6)]}",
                flush=True,
            )

    success_ids = (state == SUCCESS).nonzero(as_tuple=False).squeeze(-1)
    handoff_ids = (handoff_step >= 0).nonzero(as_tuple=False).squeeze(-1)
    close_success_ids = (close_success_step >= 0).nonzero(as_tuple=False).squeeze(-1)
    verified_ids = (verify_success_step >= 0).nonzero(as_tuple=False).squeeze(-1)
    assisted_success_ids = (final_success_step >= 0).nonzero(as_tuple=False).squeeze(-1)
    base_only_success_ids = (base_only_success_step >= 0).nonzero(as_tuple=False).squeeze(-1)
    post_base_entry_ids = (
        close_success_ids
        if args_cli.post_close_mode in ("base_direct", "scripted_lift", "learned_lift")
        else verified_ids
    )

    output_path = Path(args_cli.output)
    metrics_path = Path(args_cli.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "num_envs": n,
        "seed": args_cli.seed,
        "handoff_min_step": args_cli.handoff_min_step,
        "handoff_count": int(handoff_ids.numel()),
        "close_contract_success_count": int(close_success_ids.numel()),
        "verified_5cm_success_count": int(verified_ids.numel()),
        "final_20cm_success_count": (
            int(success_ids.numel()) if args_cli.post_close_mode != "verify_only" else None
        ),
        "base_only_20cm_success_count": (
            int(base_only_success_ids.numel())
            if args_cli.post_close_mode != "verify_only"
            else None
        ),
        "close_assisted_20cm_success_count": (
            int(assisted_success_ids.numel())
            if args_cli.post_close_mode != "verify_only"
            else None
        ),
        "handoff_to_close_success_rate": (
            float(close_success_ids.numel() / handoff_ids.numel()) if handoff_ids.numel() else 0.0
        ),
        "close_to_verified_success_rate": (
            float(verified_ids.numel() / close_success_ids.numel())
            if close_success_ids.numel()
            else 0.0
        ),
        "post_base_entry_count": int(post_base_entry_ids.numel()),
        "post_base_to_20cm_success_rate": (
            float(assisted_success_ids.numel() / post_base_entry_ids.numel())
            if args_cli.post_close_mode != "verify_only" and post_base_entry_ids.numel()
            else None
        ),
        "failure": {
            "search_timeout": int(search_timeout.sum()),
            "close_contract": int(close_failure.sum()),
            "close_timeout": int(close_timeout.sum()),
            "unlatched_lift": int(close_unlatched_lift.sum()),
            "horizontal_escape": int(close_horizontal_escape.sum()),
            "lost_window": int(close_lost_window.sum()),
            "close_unsafe": int(close_unsafe.sum()),
            "verify_timeout": int(verify_timeout.sum()),
            "verify_lost": int(verify_lost.sum()),
            "verify_unsafe": int(verify_unsafe.sum()),
            "post_base_timeout": int(post_base_timeout.sum()),
            "post_base_lost": int(post_base_lost.sum()),
            "post_base_unsafe": int(post_base_unsafe.sum()),
            "post_base_fling": int(post_base_fling.sum()),
            "post_unlatched_5cm": int(post_unlatched_5cm.sum()),
            "post_unlatched_20cm": int(post_unlatched_20cm.sum()),
            "environment_termination": int(environment_failure.sum()),
        },
        "handoff_score": _summary(handoff_score[handoff_ids]),
        "handoff_step": _summary(handoff_step[handoff_ids].float()),
        "close_length": _summary(close_age[handoff_ids].float()),
        "max_close_stable_steps": _summary(max_close_stable_steps[handoff_ids].float()),
        "close_funnel": {
            "ever_thumb_contact": int(ever_thumb_contact[handoff_ids].sum()),
            "ever_two_sided_contact": int(ever_valid_contact[handoff_ids].sum()),
            "ever_latch": int(ever_latch[handoff_ids].sum()),
            "max_other_contact_count": _summary(max_other_contact_count[handoff_ids].float()),
            "max_close_quality": _summary(max_close_quality[handoff_ids]),
            "max_wrap_quality": _summary(max_wrap_quality[handoff_ids]),
            "max_grasp_quality": _summary(max_grasp_quality[handoff_ids]),
            "max_hold_quality": _summary(max_hold_quality[handoff_ids]),
            "min_hand_action_linf_to_hybrid": _summary(min_hand_action_distance[handoff_ids]),
            "max_com_horizontal_drift": _summary(max_horizontal_drift[handoff_ids]),
        },
        "max_verify_clearance": _summary(max_verify_clearance[close_success_ids]),
        "max_post_base_clearance": _summary(max_post_base_clearance[post_base_entry_ids]),
        "force_peak_handoff": _summary(force_peak[handoff_ids]),
        "force_peak_per_finger": {
            name: _summary(force_peak_per_finger[handoff_ids, index])
            for index, name in enumerate(u.ee_names)
        },
        "force_peak_verified_success": _summary(force_peak[verified_ids]),
        "force_peak_final_20cm_success": (
            _summary(force_peak[success_ids]) if args_cli.post_close_mode != "verify_only" else None
        ),
        "post_close_mode": args_cli.post_close_mode,
        "post_hand_mode": args_cli.post_hand_mode,
        "learned_lift_hand_mode": args_cli.learned_lift_hand_mode,
        "learned_lift_stop_clearance": args_cli.learned_lift_stop_clearance,
        "dataset_phase": args_cli.dataset_phase,
        "dataset": None,
        "transition_dataset": None,
        "close_checkpoint": str(close_checkpoint_path.resolve()) if close_checkpoint_path else None,
        "close_checkpoint_sha256": _sha256(close_checkpoint_path) if close_checkpoint_path else None,
        "lift_checkpoint": str(lift_checkpoint_path.resolve()) if lift_checkpoint_path else None,
        "lift_checkpoint_sha256": _sha256(lift_checkpoint_path) if lift_checkpoint_path else None,
        "rollout_close_checkpoint": (
            str(rollout_checkpoint_path.resolve()) if rollout_checkpoint_path else None
        ),
        "rollout_close_checkpoint_sha256": (
            _sha256(rollout_checkpoint_path) if rollout_checkpoint_path else None
        ),
        "teacher_probability": args_cli.teacher_probability,
        "executed_teacher_fraction": (
            executed_teacher_rows / executed_close_rows if executed_close_rows else 1.0
        ),
    }

    dataset_success_ids = assisted_success_ids if args_cli.dataset_phase == "lift" else success_ids
    can_write_dataset = (
        args_cli.dataset_phase == "lift"
        or (args_cli.dataset_phase == "close" and close_actor is None)
    )
    if can_write_dataset and dataset_success_ids.numel() > 0:
        option_name = args_cli.dataset_phase
        phase_id = PHASE_CLOSE if option_name == "close" else PHASE_LIFT
        boundary_name = "close_start" if option_name == "close" else "lift_start"
        boundary_storage = close_start if option_name == "close" else lift_start
        if not record_env:
            raise RuntimeError(f"successful {option_name} episodes have no recorded rows")
        all_env = torch.cat(record_env)
        all_obs = torch.cat(record_obs)
        all_action = torch.cat(record_action)
        all_step = torch.cat(record_step)
        episode_obs = []
        episode_action = []
        episode_id = []
        episode_step = []
        offsets = [0]
        first_obs = []
        for episode, env_id in enumerate(dataset_success_ids.cpu().tolist()):
            selected = all_env == env_id
            ep_obs = all_obs[selected]
            ep_action = all_action[selected]
            ep_source_step = all_step[selected]
            if ep_obs.shape[0] == 0 or int(ep_source_step[0]) != 0:
                raise RuntimeError(
                    f"successful env {env_id} is missing its first {option_name} row"
                )
            expected_steps = torch.arange(ep_source_step.numel(), dtype=ep_source_step.dtype)
            if not torch.equal(ep_source_step, expected_steps):
                raise RuntimeError(
                    f"successful env {env_id} has non-contiguous {option_name} rows"
                )
            episode_obs.append(ep_obs)
            episode_action.append(ep_action)
            episode_id.append(torch.full((ep_obs.shape[0],), episode, dtype=torch.int32))
            episode_step.append(torch.arange(ep_obs.shape[0], dtype=torch.int16))
            offsets.append(offsets[-1] + ep_obs.shape[0])
            first_obs.append(ep_obs[0])

        first_obs_t = torch.stack(first_obs)
        success_cpu = dataset_success_ids.cpu()
        boundary_last_action = boundary_storage["last_action"][dataset_success_ids].cpu()
        prefix_error = (first_obs_t[:, 70:86] - boundary_last_action[:, :16]).abs().max()
        residual_error = (first_obs_t[:, 87:92] - boundary_last_action[:, 16:21]).abs().max()
        continuity_error = torch.maximum(prefix_error, residual_error)
        if not bool(torch.isfinite(continuity_error)):
            raise RuntimeError(f"non-finite {option_name} boundary continuity error")
        if float(continuity_error) > 1.0e-6:
            raise RuntimeError(
                f"first {option_name} observation is not continuous with "
                f"{boundary_name}.last_action: "
                f"prefix={float(prefix_error):.3g}, residual={float(residual_error):.3g}"
            )

        dataset = {
            "obs": torch.cat(episode_obs).float(),
            "action": torch.cat(episode_action).float(),
            "phase": torch.full((offsets[-1],), phase_id, dtype=torch.uint8),
            "episode_id": torch.cat(episode_id),
            "step": torch.cat(episode_step),
            "episode_offsets": torch.tensor(offsets, dtype=torch.int64),
            "episode_success": torch.ones(dataset_success_ids.numel(), dtype=torch.bool),
            "episode_source_env_id": success_cpu,
            "episode_handoff_step": handoff_step[dataset_success_ids].cpu(),
            "episode_handoff_score": handoff_score[dataset_success_ids].cpu(),
            "episode_handoff_proximity": handoff_proximity[dataset_success_ids].cpu(),
            "episode_close_success_step": close_success_step[dataset_success_ids].cpu(),
            "episode_verify_success_step": verify_success_step[dataset_success_ids].cpu(),
            "episode_final_success_step": final_success_step[dataset_success_ids].cpu(),
            "boundaries": {
                boundary_name: {
                    key: value[dataset_success_ids].cpu()
                    for key, value in boundary_storage.items()
                }
            },
            "meta": {
                "format_version": 1,
                "action_layout": "arm_delta7|crossdex_token9|distal_residual5",
                "observation_layout": "legacy_prefix87|distal_action5|grasp_transport23",
                "phase_names": ["approach", "close", "micro", "lift", "settle"],
                "collector": (
                    "online_frozen_base_to_close_teacher"
                    if option_name == "close"
                    else "online_base_close_to_scripted_lift_teacher"
                ),
                "dataset_phase": option_name,
                "close_arm_mode": args_cli.close_arm_mode,
                "rollout_close_checkpoint": (
                    str(rollout_checkpoint_path.resolve()) if rollout_checkpoint_path else None
                ),
                "rollout_close_checkpoint_sha256": (
                    _sha256(rollout_checkpoint_path) if rollout_checkpoint_path else None
                ),
                "teacher_probability": args_cli.teacher_probability,
                "executed_teacher_fraction": (
                    executed_teacher_rows / executed_close_rows if executed_close_rows else 1.0
                ),
                "base_checkpoint": str(checkpoint_path.resolve()),
                "base_checkpoint_sha256": _sha256(checkpoint_path),
                "feasibility_json": str(artifact_path.resolve()),
                "feasibility_sha256": _sha256(artifact_path),
                "seed": args_cli.seed,
                "handoff": {
                    "score": args_cli.handoff_score,
                    "hold_steps": args_cli.handoff_hold_steps,
                    "min_step": args_cli.handoff_min_step,
                    "min_proximity": args_cli.handoff_min_proximity,
                    "requires_unlatched": True,
                    "max_force": cfg.grasp_bonus_max_force,
                    "continuous_simulation_without_restore": True,
                },
                "close_contract": {
                    "confirm_steps": cfg.close_option_confirm_steps,
                    "grasp_quality": cfg.grasp_quality_high,
                    "hold_quality": cfg.close_option_min_hold_quality,
                    "max_force": cfg.grasp_bonus_max_force,
                    "unlatched_lift_limit": cfg.close_option_unlatched_lift_limit,
                    "horizontal_drift_limit": cfg.close_option_horizontal_drift_limit,
                    "min_proximity": cfg.close_option_min_proximity,
                    "lost_window_steps": cfg.close_option_lost_window_steps,
                },
                "verification": {
                    "height": args_cli.verify_height,
                    "clearance": args_cli.verify_clearance,
                    "hold_steps": args_cli.verify_hold_steps,
                    "hand_mode": args_cli.verify_hand_mode,
                },
                "scripted_lift": {
                    "enabled": args_cli.post_close_mode == "scripted_lift",
                    "height": args_cli.scripted_lift_height,
                    "ramp_steps": args_cli.scripted_lift_ramp_steps,
                    "hand_mode": args_cli.post_hand_mode,
                    "success_clearance": cfg.lift_success_height,
                    "success_stable_steps": args_cli.post_base_stable_steps,
                },
                "first_observation_last_action_max_error": float(
                    continuity_error
                ),
                "option_teacher_arm_action_abs_max": float(
                    torch.cat(episode_action)[:, :7].abs().max()
                ),
            },
        }
        if not bool(torch.isfinite(dataset["obs"]).all()):
            raise RuntimeError("teacher dataset contains a non-finite observation")
        if not bool(torch.isfinite(dataset["action"]).all()):
            raise RuntimeError("teacher dataset contains a non-finite action")
        if float(dataset["action"].abs().max()) > 1.0001:
            raise RuntimeError("teacher dataset contains an action outside [-1, 1]")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, output_path)
        metrics["dataset"] = str(output_path.resolve())
        metrics["retained_episodes"] = int(dataset_success_ids.numel())
        metrics["transitions"] = int(dataset["obs"].shape[0])
        metrics["continuity_max_error"] = float(continuity_error)

    transition_dataset_success_ids = torch.empty(0, dtype=torch.long, device=dev)
    if args_cli.transition_output is not None:
        terminal = transition_terminal_snapshot
        qualified_transition_success = (
            transition_assisted_success
            & ~transition_unlatched_seen
            & terminal["is_grasped"]
            & (terminal["true_clearance"] >= cfg.lift_success_height)
            & (terminal["grasp_quality"] >= cfg.grasp_quality_high)
            & (terminal["hold_quality"] >= cfg.close_option_min_hold_quality)
            & (terminal["max_force"] <= cfg.grasp_bonus_max_force)
            & (terminal["object_lin_speed"] < cfg.success_max_obj_lin_speed)
            & (terminal["object_ang_speed"] < cfg.success_max_obj_ang_speed)
            & (terminal["success_steps"] >= cfg.success_hold_steps)
        )
        transition_dataset_success_ids = qualified_transition_success.nonzero(
            as_tuple=False
        ).squeeze(-1)
        metrics["transition_strict_success_count"] = int(transition_strict_success.sum())
        metrics["transition_assisted_success_count"] = int(transition_assisted_success.sum())
        metrics["transition_qualified_success_count"] = int(
            transition_dataset_success_ids.numel()
        )
        metrics["transition_rejected_unlatched_count"] = int(
            (transition_assisted_success & transition_unlatched_seen).sum()
        )

        if transition_dataset_success_ids.numel() > 0:
            if not transition_env:
                raise RuntimeError("strict transition successes have no recorded MDP rows")
            all_transition_env = torch.cat(transition_env)
            all_transition_observation = torch.cat(transition_observation)
            all_transition_action = torch.cat(transition_action)
            all_transition_reward = torch.cat(transition_reward)
            all_transition_next_observation = torch.cat(transition_next_observation)
            all_transition_terminated = torch.cat(transition_terminated)
            all_transition_truncated = torch.cat(transition_truncated)
            all_transition_phase = torch.cat(transition_phase)
            all_transition_step = torch.cat(transition_step)

            output_transition: dict[str, list[torch.Tensor]] = {
                "observation": [],
                "action": [],
                "reward": [],
                "next_observation": [],
                "terminated": [],
                "truncated": [],
                "phase": [],
                "episode_id": [],
                "step": [],
            }
            transition_offsets = [0]
            for episode, env_id in enumerate(
                transition_dataset_success_ids.cpu().tolist()
            ):
                selected = all_transition_env == env_id
                episode_fields = {
                    "observation": all_transition_observation[selected],
                    "action": all_transition_action[selected],
                    "reward": all_transition_reward[selected],
                    "next_observation": all_transition_next_observation[selected],
                    "terminated": all_transition_terminated[selected],
                    "truncated": all_transition_truncated[selected],
                    "phase": all_transition_phase[selected],
                    "source_step": all_transition_step[selected],
                }
                length = episode_fields["observation"].shape[0]
                expected_step = torch.arange(length, dtype=torch.int32)
                if length < 1 or not torch.equal(episode_fields["source_step"], expected_step):
                    raise RuntimeError(
                        f"successful env {env_id} has missing/non-contiguous transition rows"
                    )
                terminated_rows = episode_fields["terminated"].nonzero(
                    as_tuple=False
                ).squeeze(-1)
                if terminated_rows.tolist() != [length - 1]:
                    raise RuntimeError(
                        f"successful env {env_id} must have exactly one final MDP termination, "
                        f"got {terminated_rows.tolist()}"
                    )
                if bool(episode_fields["truncated"].any()):
                    raise RuntimeError(f"successful env {env_id} contains a timeout transition")
                if not bool((episode_fields["phase"] == CLOSE).any()) or not bool(
                    (episode_fields["phase"] == POST_BASE).any()
                ):
                    raise RuntimeError(
                        f"successful env {env_id} did not traverse both close and lift phases"
                    )
                for name in (
                    "observation",
                    "action",
                    "reward",
                    "next_observation",
                    "terminated",
                    "truncated",
                    "phase",
                ):
                    output_transition[name].append(episode_fields[name])
                output_transition["episode_id"].append(
                    torch.full((length,), episode, dtype=torch.int64)
                )
                output_transition["step"].append(expected_step)
                transition_offsets.append(transition_offsets[-1] + length)

            transition_dataset = {
                name: torch.cat(parts) for name, parts in output_transition.items()
            }
            transition_dataset.update(
                {
                    "episode_offsets": torch.tensor(transition_offsets, dtype=torch.int64),
                    "episode_route": torch.ones(
                        transition_dataset_success_ids.numel(), dtype=torch.uint8
                    ),
                    "episode_success": torch.ones(
                        transition_dataset_success_ids.numel(), dtype=torch.bool
                    ),
                    "episode_source_env_id": transition_dataset_success_ids.cpu(),
                    **{
                        f"episode_terminal_{name}": value[
                            transition_dataset_success_ids
                        ].cpu()
                        for name, value in transition_terminal_snapshot.items()
                    },
                    "meta": {
                        "format_version": 1,
                        "transition_horizon": 1,
                        "terminal_observation": "adapter_captured_pre_reset",
                        "normal_task_termination": True,
                        "observation_dim": 115,
                        "action_dim": 21,
                        "action_layout": "arm_delta7|crossdex_token9|distal_residual5",
                        "phase_names": ["search", "close", "verify", "post_base"],
                        "episode_route_names": ["base_only", "close_assisted"],
                        "collector": "base_close_lift_hierarchy_strict_success",
                        "reject_unlatched_clearance_ge_5cm": True,
                        "base_checkpoint": str(checkpoint_path.resolve()),
                        "base_checkpoint_sha256": _sha256(checkpoint_path),
                        "close_checkpoint": (
                            str(close_checkpoint_path.resolve()) if close_checkpoint_path else None
                        ),
                        "close_checkpoint_sha256": (
                            _sha256(close_checkpoint_path) if close_checkpoint_path else None
                        ),
                        "lift_checkpoint": (
                            str(lift_checkpoint_path.resolve()) if lift_checkpoint_path else None
                        ),
                        "lift_checkpoint_sha256": (
                            _sha256(lift_checkpoint_path) if lift_checkpoint_path else None
                        ),
                        "feasibility_json": str(artifact_path.resolve()),
                        "feasibility_sha256": _sha256(artifact_path),
                        "seed": args_cli.seed,
                    },
                }
            )
            for name in ("observation", "action", "reward", "next_observation"):
                if not bool(torch.isfinite(transition_dataset[name]).all()):
                    raise RuntimeError(f"transition dataset contains non-finite {name}")
            if float(transition_dataset["action"].abs().max()) > 1.0001:
                raise RuntimeError("transition dataset contains an action outside [-1, 1]")
            if transition_dataset["episode_offsets"][-1] != transition_dataset["observation"].shape[0]:
                raise RuntimeError("transition episode offsets do not cover all rows")
            transition_output_path = Path(args_cli.transition_output)
            transition_output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(transition_dataset, transition_output_path)
            metrics["transition_dataset"] = str(transition_output_path.resolve())
            metrics["transition_rows"] = int(transition_dataset["observation"].shape[0])

    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(
        f"handoff={handoff_ids.numel()}/{n}, close={close_success_ids.numel()}, "
        f"verified5cm={verified_ids.numel()}, final20cm="
        f"{success_ids.numel() if args_cli.post_close_mode != 'verify_only' else 'n/a'}; "
        f"wrote {metrics_path.resolve()}",
        flush=True,
    )
    env.close()
    required_success_ids = (
        transition_dataset_success_ids
        if args_cli.transition_output is not None
        else (
            dataset_success_ids
            if can_write_dataset or args_cli.post_close_mode == "verify_only"
            else assisted_success_ids
        )
    )
    if required_success_ids.numel() < args_cli.min_successes:
        raise RuntimeError(
            f"qualified successes={required_success_ids.numel()}, below --min_successes="
            f"{args_cli.min_successes}"
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
