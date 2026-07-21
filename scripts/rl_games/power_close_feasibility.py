#!/usr/bin/env python3
"""Search arm7 plus hybrid14 actions for a strict thumb-plus-three power close.

The input is a recoverable fixed-pregrasp artifact emitted by
``hand_space_feasibility.py``.  Unlike the historical benchmark, this search
optimizes the independent power-close signals: thumb plus three legal opposed
non-thumb contacts, rigid hold, a 30 N force ceiling, and a 15-frame stable
window after the four-frame power latch.  Arm commands are zero by default.
An optional bounded seven-joint arm-target search is a reachability oracle only;
its winner must reproduce through the public incremental controller before use
as a teacher.  With ``--coupled_power_contract``, candidates execute through the
native 131-D ALIGN->CLOSE task, its phase shields and its terminal telemetry.
Every CEM iteration restores the same native reset boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import types
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True, help="artifact supplying the fixed pregrasp")
parser.add_argument(
    "--initial_action_input",
    default=None,
    help="optional artifact supplying the initial coupled or legacy hybrid14 latent",
)
parser.add_argument("--population", type=int, default=512)
parser.add_argument(
    "--replicates_per_candidate",
    type=int,
    default=1,
    help="physical clones sharing each CEM parameter vector; a pass requires every clone",
)
parser.add_argument("--iterations", type=int, default=12)
parser.add_argument(
    "--arm_delta_limit_rad",
    type=float,
    default=0.0,
    help=(
        "when positive, jointly search seven bounded arm-joint target offsets before "
        "closing the hand; zero preserves the hand-only formal-action search"
    ),
)
parser.add_argument(
    "--align_steps",
    type=int,
    default=0,
    help="simulation frames used to blend to a searched arm target before hand closure",
)
parser.add_argument("--arm_delta_penalty", type=float, default=0.20)
parser.add_argument(
    "--public_controller",
    action="store_true",
    help=(
        "evaluate CEM candidates through the formal incremental-arm and hybrid14 hand "
        "controller, including EMA, slew and tactile shields"
    ),
)
parser.add_argument(
    "--coupled_power_contract",
    action="store_true",
    help=(
        "run through the native 131-D coupled ALIGN->CLOSE environment contract; "
        "requires --public_controller and a one-state static curriculum"
    ),
)
parser.add_argument(
    "--curriculum_dataset",
    type=Path,
    default=None,
    help="coupled_power_static_close_start_v1 dataset used for exact native resets",
)
parser.add_argument("--close_steps", type=int, default=48)
parser.add_argument("--eval_steps", type=int, default=24)
parser.add_argument("--elite_frac", type=float, default=0.10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", default="/tmp/pick_tool_power_close_feasibility.json")
parser.add_argument("--require_pass", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.population < 32:
    parser.error("--population must be at least 32")
if args_cli.replicates_per_candidate < 1:
    parser.error("--replicates_per_candidate must be positive")
if args_cli.population % args_cli.replicates_per_candidate != 0:
    parser.error("--population must be divisible by --replicates_per_candidate")
if args_cli.iterations < 1:
    parser.error("--iterations must be positive")
if args_cli.close_steps < 1 or args_cli.eval_steps < 15:
    parser.error("--close_steps must be positive and --eval_steps must be at least 15")
if args_cli.arm_delta_limit_rad < 0.0 or args_cli.arm_delta_limit_rad > 0.35:
    parser.error("--arm_delta_limit_rad must lie in [0, 0.35]")
if args_cli.align_steps < 0:
    parser.error("--align_steps must be non-negative")
if args_cli.arm_delta_limit_rad > 0.0 and args_cli.align_steps < 1:
    parser.error("arm-micro search requires --align_steps >= 1")
if args_cli.arm_delta_limit_rad == 0.0 and args_cli.align_steps != 0:
    parser.error("--align_steps requires a positive --arm_delta_limit_rad")
if args_cli.arm_delta_penalty < 0.0:
    parser.error("--arm_delta_penalty must be non-negative")
if args_cli.public_controller and args_cli.arm_delta_limit_rad <= 0.0:
    parser.error("--public_controller currently requires a positive arm micro-adjustment limit")
if not 0.02 <= args_cli.elite_frac <= 0.5:
    parser.error("--elite_frac must lie in [0.02, 0.5]")
if args_cli.coupled_power_contract:
    if not args_cli.public_controller:
        parser.error("--coupled_power_contract requires --public_controller")
    if args_cli.curriculum_dataset is None:
        parser.error("--coupled_power_contract requires --curriculum_dataset")
    if args_cli.align_steps != 24:
        parser.error("coupled power v1 requires --align_steps 24")
    if not 0.0 < args_cli.arm_delta_limit_rad <= 0.12:
        parser.error("coupled power v1 requires --arm_delta_limit_rad in (0, 0.12]")
elif args_cli.curriculum_dataset is not None:
    parser.error("--curriculum_dataset is only valid with --coupled_power_contract")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401
from xhand_inhand.tasks.direct.pick_tool_token.hybrid_action import (
    apply_asymmetric_joint_residual,
    invert_asymmetric_joint_residual,
)
from power_close_search_contract import (
    POWER_CLEARANCE_LIMIT,
    POWER_FORCE_LIMIT,
    POWER_GRASP_QUALITY_MIN,
    POWER_HOLD_QUALITY_MIN,
    POWER_LATCH_CONFIRM_FRAMES,
    POWER_LATCH_RELEASE_FRAMES,
    POWER_REQUIRED_OTHER_CONTACTS,
    POWER_ROTATION_DRIFT_LIMIT,
    POWER_STABLE_FRAMES,
    POWER_XY_DRIFT_LIMIT,
    aggregate_replicated_candidates,
    conservative_coupled_teacher_pass,
    power_close_candidate_score,
    power_close_stable_frame,
    simultaneous_power_contact_stages,
    strict_power_close_pass,
    update_power_grasp_latch,
    update_stable_streak,
)


ARM_DIM = 7
HAND_DIM = 14


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_list(mapping: dict, name: str, length: int) -> list:
    value = mapping.get(name)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a list of length {length}")
    return value


def _quat_angle(reference: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Shortest sign-invariant quaternion distance in radians."""

    dot = (reference * current).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


@torch.inference_mode()
def main() -> None:
    input_path = Path(args_cli.input).resolve()
    artifact = json.loads(input_path.read_text(encoding="utf-8"))
    pregrasp = (
        artifact.get("effective_static_pregrasp")
        if args_cli.coupled_power_contract
        else artifact.get("pregrasp")
    )
    if pregrasp is None:
        pregrasp = artifact.get("pregrasp")
    if not isinstance(pregrasp, dict):
        raise ValueError("input artifact has no pregrasp mapping")
    source_pregrasp = artifact.get("source_pregrasp", artifact.get("pregrasp", pregrasp))
    joint_pos = _require_list(pregrasp, "joint_pos", 19)
    object_local_pos = _require_list(pregrasp, "object_local_pos", 3)
    object_quat = _require_list(pregrasp, "object_quat", 4)

    initial_path = (
        Path(args_cli.initial_action_input).resolve()
        if args_cli.initial_action_input is not None
        else input_path
    )
    initial_artifact = json.loads(initial_path.read_text(encoding="utf-8"))
    try:
        initial_results = initial_artifact["results"]
        initial_result = (
            initial_results.get("coupled_align_close21")
            if args_cli.coupled_power_contract
            else None
        )
        if initial_result is None:
            initial_result = initial_results["hybrid14"]
        initial_latent = initial_result["latent"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(
            "initial action artifact has no coupled/legacy hybrid14 latent"
        ) from exc
    if not isinstance(initial_latent, list) or len(initial_latent) != HAND_DIM:
        raise ValueError("initial hybrid14 latent must contain 14 values")

    curriculum_path: Path | None = None
    curriculum_sha256: str | None = None
    curriculum_boundary: dict[str, torch.Tensor] | None = None
    if args_cli.coupled_power_contract:
        assert args_cli.curriculum_dataset is not None
        curriculum_path = args_cli.curriculum_dataset.expanduser().resolve()
        if not curriculum_path.is_file():
            raise FileNotFoundError(f"curriculum dataset does not exist: {curriculum_path}")
        curriculum = torch.load(curriculum_path, map_location="cpu", weights_only=False)
        if not isinstance(curriculum, dict):
            raise TypeError("coupled curriculum must be a dictionary")
        metadata = curriculum.get("meta")
        if not isinstance(metadata, dict) or metadata.get("contract") != (
            "coupled_power_static_close_start_v1"
        ):
            raise ValueError(
                "coupled curriculum must use coupled_power_static_close_start_v1"
            )
        try:
            curriculum_boundary = curriculum["boundaries"]["close_start"]
        except (KeyError, TypeError) as exc:
            raise ValueError("coupled curriculum lacks boundaries.close_start") from exc
        if not isinstance(curriculum_boundary, dict):
            raise TypeError("coupled curriculum close_start boundary must be a dictionary")
        boundary_joint = curriculum_boundary.get("joint_pos")
        if not isinstance(boundary_joint, torch.Tensor) or boundary_joint.shape != (1, 19):
            shape = tuple(boundary_joint.shape) if isinstance(boundary_joint, torch.Tensor) else None
            raise ValueError(f"coupled CEM requires exactly one joint state, got {shape}")
        for name, expected in (
            ("object_local_pos", (1, 3)),
            ("object_quat", (1, 4)),
        ):
            value = curriculum_boundary.get(name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
                shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
                raise ValueError(f"coupled curriculum {name} expected {expected}, got {shape}")
        artifact_joint = torch.tensor(joint_pos, dtype=torch.float32)
        artifact_object_pos = torch.tensor(object_local_pos, dtype=torch.float32)
        artifact_object_quat = torch.tensor(object_quat, dtype=torch.float32)
        joint_error = float((boundary_joint[0].float() - artifact_joint).abs().max())
        object_pos_error = float(
            (curriculum_boundary["object_local_pos"][0].float() - artifact_object_pos)
            .abs()
            .max()
        )
        curriculum_quat = curriculum_boundary["object_quat"][0].float()
        quat_error = float(
            torch.minimum(
                (curriculum_quat - artifact_object_quat).abs().max(),
                (curriculum_quat + artifact_object_quat).abs().max(),
            )
        )
        if max(joint_error, object_pos_error, quat_error) > 1.0e-5:
            raise ValueError(
                "input pregrasp does not match the one-state coupled curriculum: "
                f"joint={joint_error:.3g} object_pos={object_pos_error:.3g} "
                f"quat={quat_error:.3g}"
            )
        curriculum_sha256 = _sha256(curriculum_path)

    torch.manual_seed(args_cli.seed)
    num_envs = args_cli.population
    replicates = args_cli.replicates_per_candidate
    num_candidates = num_envs // replicates
    cfg = parse_env_cfg(
        "Pick-Tool-Token-Direct-v0", device=args_cli.device, num_envs=num_envs
    )
    cfg.seed = args_cli.seed
    if args_cli.coupled_power_contract:
        assert curriculum_path is not None
        cfg.close_option_mode = True
        cfg.power_close_option_mode = True
        cfg.coupled_power_align_close_option_mode = True
        cfg.observation_space = 131
        cfg.state_space = 131
        cfg.episode_length_s = 3.0
        cfg.curriculum_dataset = str(curriculum_path)
        cfg.curriculum_boundary = "close_start"
        cfg.curriculum_reset_probability = 1.0
        cfg.curriculum_joint_noise = 0.0
    else:
        cfg.episode_length_s = 120.0
        cfg.terminate_on_drop = False
        cfg.success_hold_steps = 100000
    env = gym.make("Pick-Tool-Token-Direct-v0", cfg=cfg)
    u = env.unwrapped
    observations, _ = env.reset()
    dev = u.device
    all_ids = u.robot._ALL_INDICES

    expected_observation_dim = 131 if args_cli.coupled_power_contract else cfg.observation_space
    policy_observation = observations.get("policy")
    if not isinstance(policy_observation, torch.Tensor) or policy_observation.shape != (
        num_envs,
        expected_observation_dim,
    ):
        shape = (
            tuple(policy_observation.shape)
            if isinstance(policy_observation, torch.Tensor)
            else None
        )
        raise RuntimeError(
            f"expected policy observation {(num_envs, expected_observation_dim)}, got {shape}"
        )
    if args_cli.coupled_power_contract and not bool(u._curriculum_reset_mask.all()):
        raise RuntimeError("one or more coupled environments missed the mandatory curriculum reset")

    if cfg.action_space != ARM_DIM + HAND_DIM:
        raise RuntimeError(f"expected formal 21-D action space, got {cfg.action_space}")
    if u._n_arm != ARM_DIM or u._n_tokens + u._n_distal_residuals != HAND_DIM:
        raise RuntimeError("runtime hybrid action partition is not arm7|hand14")

    # Search diagnostics own failure accounting.  Preserve the native done truth but prevent
    # DirectRLEnv from auto-resetting a rejected candidate before its trajectory score is read.
    native_get_dones = u._get_dones
    native_terminated_seen = torch.zeros(
        num_envs, dtype=torch.bool, device=dev
    )
    native_timeout_seen = torch.zeros_like(native_terminated_seen)
    native_success_seen = torch.zeros_like(native_terminated_seen)
    native_failure_seen = torch.zeros_like(native_terminated_seen)
    native_terminal_seen = torch.zeros_like(native_terminated_seen)
    native_first_terminal_step = torch.full(
        (num_envs,), -1, dtype=torch.long, device=dev
    )
    native_terminal_stable_steps = torch.zeros(
        num_envs, dtype=torch.long, device=dev
    )
    native_terminal_power_is_grasped = torch.zeros_like(native_terminated_seen)
    native_terminal_thumb_contact = torch.zeros_like(native_terminated_seen)
    native_terminal_legal_other = torch.zeros(
        num_envs, dtype=torch.long, device=dev
    )
    native_terminal_latch_confirm_steps = torch.zeros(
        num_envs, dtype=torch.long, device=dev
    )
    native_terminal_power_grasp_quality = torch.zeros(num_envs, device=dev)
    native_terminal_hold_quality = torch.zeros(num_envs, device=dev)
    native_terminal_max_force = torch.zeros(num_envs, device=dev)
    native_terminal_align_active = torch.zeros_like(native_terminated_seen)
    rollout_active = torch.ones_like(native_terminated_seen)
    native_reason_keys = (
        "dropped",
        "unsafe_force",
        "power_close_option_failure",
        "power_close_option_timeout",
        "close_option_unlatched_lift",
        "close_option_horizontal_escape",
        "close_option_lost_window",
        "coupled_power_pose_escape",
        "coupled_power_arm_target_saturated",
    )
    native_terminal_reasons = {
        key: torch.zeros_like(native_terminated_seen) for key in native_reason_keys
    }

    def no_auto_reset(self):
        terminated, time_out = native_get_dones()
        if args_cli.coupled_power_contract:
            terminal = self.extras.get("pick_tool_terminal")
            if not isinstance(terminal, dict):
                raise RuntimeError("native coupled dones omitted pick_tool_terminal")

            def require_terminal_tensor(name: str, dtype: torch.dtype) -> torch.Tensor:
                value = terminal.get(name)
                if not isinstance(value, torch.Tensor) or value.shape != (num_envs,):
                    shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
                    raise RuntimeError(f"invalid native terminal field {name!r}: {shape}")
                if value.dtype != dtype or value.device != torch.device(dev):
                    raise RuntimeError(
                        f"invalid native terminal field {name!r}: "
                        f"dtype={value.dtype} device={value.device}"
                    )
                return value

            success = require_terminal_tensor(
                "power_close_option_success", torch.bool
            )
            failure = require_terminal_tensor(
                "power_close_option_failure", torch.bool
            )
            timeout = require_terminal_tensor(
                "power_close_option_timeout", torch.bool
            )
            stable_steps = require_terminal_tensor(
                "power_close_option_stable_steps", torch.long
            )
            terminal_power_is_grasped = require_terminal_tensor(
                "power_is_grasped", torch.bool
            )
            terminal_thumb_contact = require_terminal_tensor(
                "power_thumb_contact", torch.bool
            )
            terminal_legal_other = require_terminal_tensor(
                "power_legal_other_contact_count", torch.long
            )
            terminal_latch_confirm_steps = require_terminal_tensor(
                "power_grasp_latch_confirm_steps", torch.long
            )
            terminal_power_grasp_quality = require_terminal_tensor(
                "power_grasp_quality", torch.float32
            )
            terminal_hold_quality = require_terminal_tensor(
                "hold_quality", torch.float32
            )
            terminal_max_force = require_terminal_tensor("max_force", torch.float32)
            terminal_align_active = require_terminal_tensor(
                "coupled_power_align_active", torch.bool
            )
            generic_success = require_terminal_tensor("success", torch.bool)
            generic_failure = require_terminal_tensor("failure", torch.bool)
            generic_timeout = require_terminal_tensor("time_out", torch.bool)
            dropped = require_terminal_tensor("dropped", torch.bool)
            unsafe_force = require_terminal_tensor("unsafe_force", torch.bool)
            pose_escape = require_terminal_tensor(
                "coupled_power_pose_escape", torch.bool
            )
            if not torch.equal(terminated, success | failure):
                raise RuntimeError("native coupled terminated classification is inconsistent")
            if not torch.equal(time_out, timeout):
                raise RuntimeError("native coupled timeout classification is inconsistent")
            if not (
                torch.equal(generic_success, success)
                and torch.equal(generic_failure, failure)
                and torch.equal(generic_timeout, timeout)
            ):
                raise RuntimeError("native coupled generic terminal aliases are inconsistent")
            if bool(((success & failure) | (success & timeout) | (failure & timeout)).any()):
                raise RuntimeError("native coupled terminal classes are not mutually exclusive")
            if bool((success & (dropped | unsafe_force | pose_escape)).any()):
                raise RuntimeError("native coupled success violated safety precedence")
            done = terminated | time_out
            newly_done = done & (~native_terminal_seen)
            native_terminated_seen.logical_or_(terminated & newly_done)
            native_timeout_seen.logical_or_(time_out & newly_done)
            native_success_seen.logical_or_(success & newly_done)
            native_failure_seen.logical_or_(failure & newly_done)
            native_terminal_stable_steps.copy_(
                torch.where(newly_done, stable_steps, native_terminal_stable_steps)
            )
            native_first_terminal_step.copy_(
                torch.where(
                    newly_done,
                    self.episode_length_buf,
                    native_first_terminal_step,
                )
            )
            native_terminal_power_is_grasped.copy_(
                torch.where(
                    newly_done,
                    terminal_power_is_grasped,
                    native_terminal_power_is_grasped,
                )
            )
            native_terminal_thumb_contact.copy_(
                torch.where(
                    newly_done,
                    terminal_thumb_contact,
                    native_terminal_thumb_contact,
                )
            )
            native_terminal_legal_other.copy_(
                torch.where(
                    newly_done,
                    terminal_legal_other,
                    native_terminal_legal_other,
                )
            )
            native_terminal_latch_confirm_steps.copy_(
                torch.where(
                    newly_done,
                    terminal_latch_confirm_steps,
                    native_terminal_latch_confirm_steps,
                )
            )
            native_terminal_power_grasp_quality.copy_(
                torch.where(
                    newly_done,
                    terminal_power_grasp_quality,
                    native_terminal_power_grasp_quality,
                )
            )
            native_terminal_hold_quality.copy_(
                torch.where(
                    newly_done,
                    terminal_hold_quality,
                    native_terminal_hold_quality,
                )
            )
            native_terminal_max_force.copy_(
                torch.where(
                    newly_done,
                    terminal_max_force,
                    native_terminal_max_force,
                )
            )
            native_terminal_align_active.copy_(
                torch.where(
                    newly_done,
                    terminal_align_active,
                    native_terminal_align_active,
                )
            )
            for key, captured in native_terminal_reasons.items():
                value = require_terminal_tensor(key, torch.bool)
                captured.logical_or_(value & newly_done)
            native_terminal_seen.logical_or_(done)
            rollout_active.logical_and_(~done)
        else:
            native_terminated_seen.logical_or_(terminated)
            native_timeout_seen.logical_or_(time_out)
        zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return zeros, zeros

    u._get_dones = types.MethodType(no_auto_reset, u)

    snapshot_joint = torch.tensor(joint_pos, dtype=torch.float32, device=dev)
    snapshot_object_local = torch.tensor(
        object_local_pos, dtype=torch.float32, device=dev
    )
    snapshot_object_quat = torch.tensor(object_quat, dtype=torch.float32, device=dev)
    repeated_joint = snapshot_joint.unsqueeze(0).repeat(num_envs, 1)
    start_com_xy = torch.zeros((num_envs, 2), dtype=torch.float32, device=dev)
    snapshot_quat_batch = snapshot_object_quat.unsqueeze(0).repeat(num_envs, 1)
    search_arm_micro = args_cli.arm_delta_limit_rad > 0.0
    search_joint_target = repeated_joint.clone()

    if search_arm_micro and not args_cli.public_controller:
        # CEM searches a bounded arm *target offset*, not a constant incremental
        # command that would integrate without bound.  The rollout below blends
        # to this target, then closes the hand.  A later replay/teacher converts
        # the selected offset into the public incremental action sequence.
        def apply_search_target(self, actions: torch.Tensor) -> None:
            self.actions = torch.zeros_like(actions)
            self.dof_targets.copy_(search_joint_target)
            self.dof_targets[:] = torch.clamp(
                self.dof_targets, self.dof_lower, self.dof_upper
            )

        u._pre_physics_step = types.MethodType(apply_search_target, u)

    @torch.inference_mode()
    def restore_snapshot() -> None:
        if args_cli.coupled_power_contract:
            reset_observations, _ = env.reset()
            policy = reset_observations.get("policy")
            if not isinstance(policy, torch.Tensor) or policy.shape != (num_envs, 131):
                raise RuntimeError("native coupled reset changed the 131-D observation contract")
            if not bool(u._curriculum_reset_mask.all()):
                raise RuntimeError("native coupled reset missed a curriculum row")
            if bool((u.episode_length_buf != 0).any()):
                raise RuntimeError("native coupled reset did not clear the episode clock")
            repeated_joint.copy_(u.dof_targets)
            search_joint_target.copy_(u.dof_targets)
            start_com_xy.copy_(u._close_option_start_xy)
            snapshot_quat_batch.copy_(u._close_option_start_quat)
        else:
            u._reset_idx(all_ids)
            u.robot.write_joint_state_to_sim(
                repeated_joint, torch.zeros_like(repeated_joint), env_ids=all_ids
            )
            u.robot.set_joint_position_target(repeated_joint, env_ids=all_ids)
            u.dof_targets.copy_(repeated_joint)
            search_joint_target.copy_(repeated_joint)
            pose = torch.zeros((num_envs, 7), dtype=torch.float32, device=dev)
            pose[:, :3] = snapshot_object_local + u.scene.env_origins
            pose[:, 3:7] = snapshot_object_quat
            u.object.write_root_pose_to_sim(pose, env_ids=all_ids)
            u.object.write_root_velocity_to_sim(
                torch.zeros((num_envs, 6), device=dev), env_ids=all_ids
            )
            u.episode_length_buf.zero_()
            u.actions.zero_()
            u.prev_actions.zero_()
            u._compute_intermediate_values()
            start_com_xy.copy_(u._object_com_position_w()[:, :2])
        native_terminated_seen.zero_()
        native_timeout_seen.zero_()
        native_success_seen.zero_()
        native_failure_seen.zero_()
        native_terminal_seen.zero_()
        native_first_terminal_step.fill_(-1)
        native_terminal_stable_steps.zero_()
        native_terminal_power_is_grasped.zero_()
        native_terminal_thumb_contact.zero_()
        native_terminal_legal_other.zero_()
        native_terminal_latch_confirm_steps.zero_()
        native_terminal_power_grasp_quality.zero_()
        native_terminal_hold_quality.zero_()
        native_terminal_max_force.zero_()
        native_terminal_align_active.zero_()
        rollout_active.fill_(True)
        for captured in native_terminal_reasons.values():
            captured.zero_()

    hand_lower = u.dof_lower[:, u._hand_ids_t]
    hand_upper = u.dof_upper[:, u._hand_ids_t]

    def decode_hybrid14(latent: torch.Tensor) -> torch.Tensor:
        token_target = u.retarget.retarget_from_unit_action(latent[:, : u._n_tokens])[
            :, u._retarget2isaac
        ]
        target, _ = apply_asymmetric_joint_residual(
            token_target,
            hand_lower[: latent.shape[0]],
            hand_upper[: latent.shape[0]],
            latent[:, u._n_tokens :],
            u._distal_hand_ids,
        )
        return target

    initial_hold_action: torch.Tensor | None = None
    initial_hold_decode_error = 0.0
    if args_cli.public_controller and not args_cli.coupled_power_contract:
        pregrasp_token = pregrasp.get("token")
        if not isinstance(pregrasp_token, list) or len(pregrasp_token) != u._n_tokens:
            raise ValueError("public-controller alignment requires pregrasp.token[9]")
        seed_token = torch.tensor(
            pregrasp_token, dtype=torch.float32, device=dev
        )
        fit_population = 4096
        fit_elites = 256
        fit_generator = torch.Generator(device=dev).manual_seed(99173)
        fit_mean = seed_token.clone()
        fit_std = torch.full_like(fit_mean, 0.30)
        fit_lower = hand_lower[:1].expand(fit_population, -1)
        fit_upper = hand_upper[:1].expand(fit_population, -1)
        fit_target = repeated_joint[:1, u._hand_ids_t].expand(
            fit_population, -1
        )
        best_fit_action = None
        best_fit_error = float("inf")
        for _ in range(15):
            candidate_token = (
                fit_mean
                + fit_std
                * torch.randn(
                    (fit_population, u._n_tokens),
                    device=dev,
                    generator=fit_generator,
                )
            ).clamp(-1.0, 1.0)
            candidate_token[0] = fit_mean.clamp(-1.0, 1.0)
            candidate_token[1] = seed_token
            candidate_base = u.retarget.retarget_from_unit_action(candidate_token)[
                :, u._retarget2isaac
            ]
            candidate_residual = invert_asymmetric_joint_residual(
                fit_target,
                candidate_base,
                fit_lower,
                fit_upper,
                u._distal_hand_ids,
            )
            candidate_decoded, _ = apply_asymmetric_joint_residual(
                candidate_base,
                fit_lower,
                fit_upper,
                candidate_residual,
                u._distal_hand_ids,
            )
            candidate_error = candidate_decoded - fit_target
            fit_loss = candidate_error.square().mean(dim=-1) + 0.25 * (
                candidate_error.abs().max(dim=-1).values.square()
            )
            best_index = int(torch.argmin(fit_loss).item())
            best_error = float(candidate_error[best_index].abs().max().item())
            if best_error < best_fit_error:
                best_fit_error = best_error
                best_fit_action = torch.cat(
                    (candidate_token[best_index], candidate_residual[best_index]),
                    dim=0,
                ).clone()
            elite_indices = torch.topk(
                fit_loss, k=fit_elites, largest=False
            ).indices
            elite = candidate_token[elite_indices]
            fit_mean = elite.mean(dim=0)
            fit_std = elite.std(dim=0, unbiased=False).clamp(0.015, 0.50)
        assert best_fit_action is not None
        initial_hold_action = best_fit_action.unsqueeze(0).repeat(num_envs, 1)
        initial_hold_decode_error = best_fit_error

    hand_mean = torch.tensor(initial_latent, dtype=torch.float32, device=dev)
    hand_std = torch.cat(
        (
            torch.full((u._n_tokens,), 0.35, device=dev),
            torch.full((u._n_distal_residuals,), 0.55, device=dev),
        )
    )
    if search_arm_micro:
        initial_arm_delta = initial_result.get("arm_delta_target_rad")
        if initial_arm_delta is None:
            arm_mean = torch.zeros(ARM_DIM, device=dev)
        elif not isinstance(initial_arm_delta, list) or len(initial_arm_delta) != ARM_DIM:
            raise ValueError("initial arm_delta_target_rad must contain 7 values")
        else:
            arm_mean = (
                torch.tensor(initial_arm_delta, dtype=torch.float32, device=dev)
                / args_cli.arm_delta_limit_rad
            ).clamp(-1.0, 1.0)
        mean = torch.cat((arm_mean, hand_mean))
        std = torch.cat((torch.full((ARM_DIM,), 0.40, device=dev), hand_std))
    else:
        mean = hand_mean
        std = hand_std
    original_seed = mean.clone()
    parameter_dim = int(mean.numel())
    elite_count = min(
        num_candidates,
        max(8, int(round(num_candidates * args_cli.elite_frac))),
    )
    best_any: dict | None = None
    best_pass: dict | None = None
    best_native: dict | None = None
    search_trace: list[dict] = []
    controller_label = (
        "coupled-native"
        if args_cli.coupled_power_contract
        else ("public" if args_cli.public_controller else "oracle")
    )

    print(
        f"POWER CEM envs={num_envs} candidates={num_candidates} "
        f"replicates={replicates} iterations={args_cli.iterations} "
        f"align={args_cli.align_steps} close={args_cli.close_steps} "
        f"eval={args_cli.eval_steps} elites={elite_count} "
        f"arm_delta_limit={args_cli.arm_delta_limit_rad:.3f}rad "
        f"controller={controller_label} "
        f"initial_hold_error={initial_hold_decode_error:.4f}rad",
        flush=True,
    )

    for iteration in range(args_cli.iterations):
        candidate_parameter = (
            mean + std * torch.randn((num_candidates, parameter_dim), device=dev)
        ).clamp(
            -1.0, 1.0
        )
        candidate_parameter[0] = mean.clamp(-1.0, 1.0)
        if num_candidates > 1:
            candidate_parameter[1] = original_seed
        parameter = candidate_parameter.repeat_interleave(replicates, dim=0)
        if search_arm_micro:
            arm_parameter = parameter[:, :ARM_DIM]
            hand_latent = parameter[:, ARM_DIM:]
        else:
            arm_parameter = torch.zeros((num_envs, ARM_DIM), device=dev)
            hand_latent = parameter
        action = torch.zeros((num_envs, ARM_DIM + HAND_DIM), device=dev)
        action[:, ARM_DIM:] = hand_latent

        # Coupled search must rebuild every candidate rollout from the environment's native
        # close-start reset.  This initializes the phase clock, latch counters, potentials and
        # target anchors that a hand-written joint/object restore cannot reproduce.
        restore_snapshot()
        candidate_joint_target = repeated_joint.clone()
        candidate_hand_target = decode_hybrid14(hand_latent)
        candidate_joint_target[:, u._hand_ids_t] = candidate_hand_target
        if search_arm_micro:
            raw_arm_target = repeated_joint[:, u._arm_ids_t] + (
                args_cli.arm_delta_limit_rad * arm_parameter
            )
            arm_lower = u.dof_lower[:, u._arm_ids_t]
            arm_upper = u.dof_upper[:, u._arm_ids_t]
            candidate_arm_target = torch.maximum(
                torch.minimum(raw_arm_target, arm_upper), arm_lower
            )
            candidate_joint_target[:, u._arm_ids_t] = candidate_arm_target
            arm_delta_actual = (
                candidate_arm_target - repeated_joint[:, u._arm_ids_t]
            )
        else:
            arm_delta_actual = torch.zeros((num_envs, ARM_DIM), device=dev)

        q_close_sum = torch.zeros(num_envs, device=dev)
        q_wrap_sum = torch.zeros(num_envs, device=dev)
        q_grasp_sum = torch.zeros(num_envs, device=dev)
        q_close_peak = torch.zeros(num_envs, device=dev)
        q_wrap_peak = torch.zeros(num_envs, device=dev)
        q_grasp_peak = torch.zeros(num_envs, device=dev)
        hold_sum = torch.zeros(num_envs, device=dev)
        hold_peak = torch.zeros(num_envs, device=dev)
        thumb_sum = torch.zeros(num_envs, device=dev)
        legal_other_sum = torch.zeros(num_envs, device=dev)
        legal_other_peak = torch.zeros(num_envs, dtype=torch.long, device=dev)
        thumb_and_third_sum = torch.zeros(num_envs, device=dev)
        thumb_and_fourth_sum = torch.zeros(num_envs, device=dev)
        thumb_and_three_seen = torch.zeros(
            num_envs, dtype=torch.bool, device=dev
        )
        stable_sum = torch.zeros(num_envs, device=dev)
        stable_streak = torch.zeros(num_envs, dtype=torch.long, device=dev)
        stable_streak_peak = torch.zeros_like(stable_streak)
        first_success_step = torch.full(
            (num_envs,), -1, dtype=torch.long, device=dev
        )
        power_is_grasped = torch.zeros(num_envs, dtype=torch.bool, device=dev)
        power_latch_confirm = torch.zeros(num_envs, dtype=torch.long, device=dev)
        power_latch_release = torch.zeros(num_envs, dtype=torch.long, device=dev)
        power_latch_confirm_peak = torch.zeros_like(power_latch_confirm)
        force_peak = torch.zeros((num_envs, len(u.ee_names)), device=dev)
        clearance_peak = torch.full((num_envs,), -float("inf"), device=dev)
        clearance_min = torch.full((num_envs,), float("inf"), device=dev)
        xy_drift_peak = torch.zeros(num_envs, device=dev)
        rotation_drift_peak = torch.zeros(num_envs, device=dev)
        arm_table_clearance_min = torch.full(
            (num_envs,), float("inf"), device=dev
        )
        arm_tracking_error_peak = torch.zeros(num_envs, device=dev)
        arm_target_saturated_seen = torch.zeros(
            num_envs, dtype=torch.bool, device=dev
        )
        native_stable_streak_peak = torch.zeros(
            num_envs, dtype=torch.long, device=dev
        )
        native_power_latched_seen = torch.zeros(
            num_envs, dtype=torch.bool, device=dev
        )
        native_latch_confirm_steps_peak = torch.zeros(
            num_envs, dtype=torch.long, device=dev
        )
        eval_count = torch.zeros(num_envs, device=dev)

        # Teacher safety is trajectory-wide, including the reset boundary before action 1.
        initial_signals = u._compute_grasp_signals()
        force_peak.copy_(initial_signals["force_magnitude"])
        initial_clearance = u._object_true_min_z() - u._table_surface_z
        clearance_peak.copy_(initial_clearance)
        clearance_min.copy_(initial_clearance)
        initial_com_xy = u._object_com_position_w()[:, :2]
        xy_drift_peak.copy_((initial_com_xy - start_com_xy).norm(dim=-1))
        rotation_drift_peak.copy_(
            _quat_angle(snapshot_quat_batch, u.object.data.root_quat_w)
        )
        initial_arm_clearance = (
            u.robot.data.body_pos_w[:, u._arm_body_ids, 2]
            - u.scene.env_origins[:, 2].unsqueeze(-1)
            - u._table_surface_z
        ).min(dim=-1).values
        arm_table_clearance_min.copy_(initial_arm_clearance)

        eval_start = (
            args_cli.align_steps
            if args_cli.coupled_power_contract
            else args_cli.align_steps + args_cli.close_steps
        )
        total_steps = (
            int(u.max_episode_length)
            if args_cli.coupled_power_contract
            else args_cli.align_steps + args_cli.close_steps + args_cli.eval_steps
        )
        rollout_steps_executed = 0
        for step in range(total_steps):
            if args_cli.coupled_power_contract and not bool(rollout_active.any()):
                break
            rollout_steps_executed = step + 1
            step_active = (
                rollout_active.clone()
                if args_cli.coupled_power_contract
                else torch.ones(num_envs, dtype=torch.bool, device=dev)
            )
            desired_arm_target = candidate_joint_target[:, u._arm_ids_t]
            if args_cli.public_controller:
                step_action = torch.zeros_like(action)
                if step < args_cli.align_steps:
                    x = float(step + 1) / float(args_cli.align_steps)
                    arm_blend = x * x * (3.0 - 2.0 * x)
                    desired_arm_target = repeated_joint[:, u._arm_ids_t] + (
                        arm_blend * arm_delta_actual
                    )
                    if args_cli.coupled_power_contract:
                        # The native phase shield holds the captured hand target during ALIGN.
                        # Supplying the candidate throughout also handles an early power latch:
                        # the environment then atomically removes arm authority and enables hand.
                        step_action[:, ARM_DIM:] = hand_latent
                    else:
                        assert initial_hold_action is not None
                        step_action[:, ARM_DIM:] = initial_hold_action
                else:
                    step_action[:, ARM_DIM:] = hand_latent
                current_arm_target = u.dof_targets[:, u._arm_ids_t]
                arm_multiplier = (
                    cfg.coupled_power_arm_action_multiplier
                    if args_cli.coupled_power_contract
                    else 1.0
                )
                step_action[:, :ARM_DIM] = torch.clamp(
                    (desired_arm_target - current_arm_target)
                    / (cfg.action_scale * cfg.act_moving_average * arm_multiplier),
                    -1.0,
                    1.0,
                )
                step_action[~step_active] = 0.0
                env.step(step_action)
                tracking_error = (
                    (u.dof_targets[:, u._arm_ids_t] - desired_arm_target)
                    .abs()
                    .max(dim=-1)
                    .values
                )
                arm_tracking_error_peak = torch.where(
                    step_active,
                    torch.maximum(arm_tracking_error_peak, tracking_error),
                    arm_tracking_error_peak,
                )
            elif search_arm_micro:
                search_joint_target.copy_(repeated_joint)
                if step < args_cli.align_steps:
                    x = float(step + 1) / float(args_cli.align_steps)
                    arm_blend = x * x * (3.0 - 2.0 * x)
                    hand_blend = 0.0
                elif step < eval_start:
                    arm_blend = 1.0
                    x = float(step - args_cli.align_steps + 1) / float(
                        args_cli.close_steps
                    )
                    hand_blend = x * x * (3.0 - 2.0 * x)
                else:
                    arm_blend = 1.0
                    hand_blend = 1.0
                search_joint_target[:, u._arm_ids_t] = repeated_joint[
                    :, u._arm_ids_t
                ] + arm_blend * arm_delta_actual
                search_joint_target[:, u._hand_ids_t] = repeated_joint[
                    :, u._hand_ids_t
                ] + hand_blend * (
                    candidate_hand_target - repeated_joint[:, u._hand_ids_t]
                )
                env.step(action)
            else:
                env.step(action)
            signals = u._compute_grasp_signals()
            max_force = signals["force_magnitude"].max(dim=-1).values
            force_peak = torch.where(
                step_active.unsqueeze(-1),
                torch.maximum(force_peak, signals["force_magnitude"]),
                force_peak,
            )
            clearance = u._object_true_min_z() - u._table_surface_z
            clearance_peak = torch.where(
                step_active,
                torch.maximum(clearance_peak, clearance),
                clearance_peak,
            )
            clearance_min = torch.where(
                step_active,
                torch.minimum(clearance_min, clearance),
                clearance_min,
            )
            object_com_xy = u._object_com_position_w()[:, :2]
            xy_drift = (object_com_xy - start_com_xy).norm(dim=-1)
            xy_drift_peak = torch.where(
                step_active, torch.maximum(xy_drift_peak, xy_drift), xy_drift_peak
            )
            rotation_drift = _quat_angle(
                snapshot_quat_batch, u.object.data.root_quat_w
            )
            rotation_drift_peak = torch.where(
                step_active,
                torch.maximum(rotation_drift_peak, rotation_drift),
                rotation_drift_peak,
            )
            arm_clearance = (
                u.robot.data.body_pos_w[:, u._arm_body_ids, 2]
                - u.scene.env_origins[:, 2].unsqueeze(-1)
                - u._table_surface_z
            ).min(dim=-1).values
            arm_table_clearance_min = torch.where(
                step_active,
                torch.minimum(arm_table_clearance_min, arm_clearance),
                arm_table_clearance_min,
            )
            if args_cli.coupled_power_contract:
                arm_target_saturated_seen.logical_or_(
                    step_active & u._coupled_arm_target_saturated_ever
                )
                native_stable_streak_peak.copy_(
                    torch.where(
                        step_active,
                        torch.maximum(
                            native_stable_streak_peak,
                            u._close_option_stable_steps,
                        ),
                        native_stable_streak_peak,
                    )
                )
                native_latch_state = torch.where(
                    native_terminal_seen,
                    native_terminal_power_is_grasped,
                    u._power_is_grasped,
                )
                native_confirm_steps = torch.where(
                    native_terminal_seen,
                    native_terminal_latch_confirm_steps,
                    u._power_contact_steps,
                )
                native_power_latched_seen.logical_or_(step_active & native_latch_state)
                native_latch_confirm_steps_peak.copy_(
                    torch.where(
                        step_active,
                        torch.maximum(
                            native_latch_confirm_steps_peak,
                            native_confirm_steps,
                        ),
                        native_latch_confirm_steps_peak,
                    )
                )

            stable = power_close_stable_frame(
                power_is_grasped,
                signals["power_thumb_contact"],
                signals["power_legal_other_contact_count"],
                signals["power_grasp_quality"],
                signals["hold_quality"],
                max_force,
            )
            next_stable_streak, next_stable_streak_peak = update_stable_streak(
                stable_streak, stable_streak_peak, stable
            )
            stable_streak = torch.where(
                step_active, next_stable_streak, stable_streak
            )
            stable_streak_peak = torch.where(
                step_active, next_stable_streak_peak, stable_streak_peak
            )
            newly_successful = (stable_streak >= POWER_STABLE_FRAMES) & (
                first_success_step < 0
            ) & step_active
            first_success_step = torch.where(
                newly_successful,
                torch.full_like(first_success_step, step + 1),
                first_success_step,
            )
            # Install the latch after evaluating this action's option state.  This mirrors
            # DirectRLEnv's dones-before-reward ordering and preserves action 19 as the earliest
            # possible strict completion.
            (
                next_power_is_grasped,
                next_power_latch_confirm,
                next_power_latch_release,
            ) = update_power_grasp_latch(
                signals["power_grasp_quality"],
                power_is_grasped,
                power_latch_confirm,
                power_latch_release,
            )
            power_is_grasped = torch.where(
                step_active, next_power_is_grasped, power_is_grasped
            )
            power_latch_confirm = torch.where(
                step_active, next_power_latch_confirm, power_latch_confirm
            )
            power_latch_release = torch.where(
                step_active, next_power_latch_release, power_latch_release
            )
            power_latch_confirm_peak = torch.maximum(
                power_latch_confirm_peak, power_latch_confirm
            )

            if args_cli.coupled_power_contract:
                eval_mask = step_active & (~u._coupled_align_active)
            else:
                eval_mask = step_active & (step >= eval_start)
            if bool(eval_mask.any()):
                q_close_sum[eval_mask] += signals["power_close_quality"][eval_mask]
                q_wrap_sum[eval_mask] += signals["power_wrap_quality"][eval_mask]
                q_grasp_sum[eval_mask] += signals["power_grasp_quality"][eval_mask]
                hold_sum[eval_mask] += signals["hold_quality"][eval_mask]
                q_close_peak[eval_mask] = torch.maximum(
                    q_close_peak[eval_mask],
                    signals["power_close_quality"][eval_mask],
                )
                q_wrap_peak[eval_mask] = torch.maximum(
                    q_wrap_peak[eval_mask],
                    signals["power_wrap_quality"][eval_mask],
                )
                q_grasp_peak[eval_mask] = torch.maximum(
                    q_grasp_peak[eval_mask],
                    signals["power_grasp_quality"][eval_mask],
                )
                hold_peak[eval_mask] = torch.maximum(
                    hold_peak[eval_mask], signals["hold_quality"][eval_mask]
                )
                thumb_sum[eval_mask] += signals["power_thumb_contact"][eval_mask].float()
                legal_other_sum[eval_mask] += signals[
                    "power_legal_other_contact_count"
                ][eval_mask].float()
                legal_other = signals["power_legal_other_contact_count"]
                legal_other_peak[eval_mask] = torch.maximum(
                    legal_other_peak[eval_mask], legal_other[eval_mask]
                )
                thumb_and_third, thumb_and_fourth = simultaneous_power_contact_stages(
                    signals["power_thumb_contact"], legal_other
                )
                thumb_and_third_sum[eval_mask] += thumb_and_third[eval_mask].float()
                thumb_and_fourth_sum[eval_mask] += thumb_and_fourth[eval_mask].float()
                thumb_and_three_seen.logical_or_(eval_mask & thumb_and_third)
                stable_sum[eval_mask] += stable[eval_mask].float()
                eval_count[eval_mask] += 1

        if args_cli.coupled_power_contract and not bool(native_terminal_seen.all()):
            missing = int((~native_terminal_seen).sum().item())
            raise RuntimeError(
                f"native coupled rollout ended without a terminal classification for {missing} rows"
            )

        eval_denominator = eval_count.clamp_min(1.0)
        q_close_mean = q_close_sum / eval_denominator
        q_wrap_mean = q_wrap_sum / eval_denominator
        q_grasp_mean = q_grasp_sum / eval_denominator
        hold_mean = hold_sum / eval_denominator
        thumb_frac = thumb_sum / eval_denominator
        legal_other_mean = legal_other_sum / eval_denominator
        thumb_and_third_frac = thumb_and_third_sum / eval_denominator
        thumb_and_fourth_frac = thumb_and_fourth_sum / eval_denominator
        stable_frac = stable_sum / eval_denominator
        force_peak_max = force_peak.max(dim=-1).values
        if args_cli.coupled_power_contract:
            native_stable_at_end = torch.where(
                native_terminal_seen,
                native_terminal_stable_steps,
                u._close_option_stable_steps,
            )
            first_success_step = torch.where(
                native_success_seen,
                native_first_terminal_step,
                torch.full_like(native_first_terminal_step, -1),
            )
            stable_streak = native_stable_at_end
            stable_streak_peak = native_stable_streak_peak
            power_is_grasped = torch.where(
                native_terminal_seen,
                native_terminal_power_is_grasped,
                u._power_is_grasped,
            )
            unexpected_done = native_failure_seen | native_timeout_seen
            strict_env_pass = conservative_coupled_teacher_pass(
                native_success=native_success_seen,
                native_failure=native_failure_seen,
                native_timeout=native_timeout_seen,
                terminal_stable_steps=native_terminal_stable_steps,
                terminal_power_is_grasped=native_terminal_power_is_grasped,
                terminal_thumb_contact=native_terminal_thumb_contact,
                terminal_legal_other_contacts=native_terminal_legal_other,
                terminal_power_grasp_quality=native_terminal_power_grasp_quality,
                terminal_hold_quality=native_terminal_hold_quality,
                terminal_max_force=native_terminal_max_force,
                terminal_align_active=native_terminal_align_active,
                trajectory_force_peak=force_peak_max,
                trajectory_xy_drift_peak=xy_drift_peak,
                trajectory_rotation_drift_peak=rotation_drift_peak,
                trajectory_clearance_peak=clearance_peak,
                arm_target_saturated=arm_target_saturated_seen,
            )
        else:
            unexpected_done = native_terminated_seen | native_timeout_seen
            strict_env_pass = strict_power_close_pass(
                stable_streak,
                force_peak_max,
                xy_drift_peak,
                rotation_drift_peak,
                clearance_peak,
                unexpected_done,
            )
        env_score = power_close_candidate_score(
            power_close_mean=q_close_mean,
            thumb_fraction=thumb_frac,
            thumb_and_third_fraction=thumb_and_third_frac,
            thumb_and_fourth_fraction=thumb_and_fourth_frac,
            power_wrap_mean=q_wrap_mean,
            power_grasp_mean=q_grasp_mean,
            hold_mean=hold_mean,
            stable_fraction=stable_frac,
            stable_streak_at_end=stable_streak,
            stable_streak_peak=stable_streak_peak,
            force_peak=force_peak_max,
            xy_drift_peak=xy_drift_peak,
            rotation_drift_peak=rotation_drift_peak,
            clearance_peak=clearance_peak,
            # A single lucky clone must not receive the lexicographic pass bonus.  The bonus is
            # installed only after all physical replicas of a candidate pass independently.
            strict_pass=torch.zeros_like(strict_env_pass),
            unexpected_done=unexpected_done,
        )
        if search_arm_micro:
            normalized_actual_arm_delta = (
                arm_delta_actual / args_cli.arm_delta_limit_rad
            )
            env_score -= (
                args_cli.arm_delta_penalty
                * normalized_actual_arm_delta.square().mean(dim=-1)
            )
        if args_cli.coupled_power_contract:
            # Drop and tactile-force termination are irrecoverable hard rejects.  Pose-window
            # failures retain the continuous drift/rotation/clearance penalties above: otherwise
            # a nearly stable grasp just beyond the boundary is exactly tied at -1e6 and CEM cannot
            # move back toward the safe side.  Native/teacher success gates remain unchanged.
            irrecoverable_failure = (
                native_terminal_reasons["dropped"]
                | native_terminal_reasons["unsafe_force"]
            )
            env_score = torch.where(
                irrecoverable_failure,
                -1.0e6 + torch.clamp(env_score, -1000.0, 1000.0),
                env_score,
            )
        finite = torch.stack(
            (
                q_close_mean,
                q_wrap_mean,
                q_grasp_mean,
                hold_mean,
                force_peak_max,
                clearance_peak,
                clearance_min,
                xy_drift_peak,
                rotation_drift_peak,
                arm_table_clearance_min,
                arm_tracking_error_peak,
                env_score,
            ),
            dim=-1,
        ).isfinite().all(dim=-1)
        if not bool(finite.any()):
            raise RuntimeError(f"all CEM candidates became non-finite at iteration {iteration + 1}")
        strict_env_pass &= finite
        env_score = torch.where(
            finite, env_score, torch.full_like(env_score, -1.0e9)
        )
        candidate_score, strict_group_pass, finite_group = (
            aggregate_replicated_candidates(
                env_score, strict_env_pass, finite, replicates
            )
        )
        native_group_pass = (
            native_success_seen.reshape(num_candidates, replicates).all(dim=1)
            & finite_group
            if args_cli.coupled_power_contract
            else strict_group_pass
        )

        legal_peak_matrix = legal_other_peak.reshape(num_candidates, replicates)
        q_grasp_peak_matrix = q_grasp_peak.reshape(num_candidates, replicates)
        native_latch_matrix = native_power_latched_seen.reshape(
            num_candidates, replicates
        )
        thumb_three_matrix = thumb_and_three_seen.reshape(
            num_candidates, replicates
        )
        progress_env_score = (
            100.0 * native_stable_streak_peak.float()
            + 20.0 * native_power_latched_seen.float()
            + 12.0 * thumb_and_three_seen.float()
            + 2.0 * legal_other_peak.float()
            + q_grasp_peak
        )
        progress_matrix = progress_env_score.reshape(num_candidates, replicates)
        progress_candidate_score = (
            0.5 * progress_matrix.mean(dim=1) + 0.5 * progress_matrix.amin(dim=1)
        )
        progress_index = int(torch.argmax(progress_candidate_score).item())
        progress_slice = slice(
            progress_index * replicates, (progress_index + 1) * replicates
        )
        search_trace.append(
            {
                "iteration": iteration + 1,
                "strict_teacher_pass_candidates": int(
                    strict_group_pass.sum().item()
                ),
                "native_success_candidates": int(native_group_pass.sum().item()),
                "legal_other_contact_peak_max": int(legal_other_peak.max().item()),
                "environments_reaching_three_legal_contacts": int(
                    (legal_other_peak >= POWER_REQUIRED_OTHER_CONTACTS).sum().item()
                ),
                "environments_reaching_simultaneous_thumb_and_three": int(
                    thumb_and_three_seen.sum().item()
                ),
                "candidates_reaching_thumb_and_three_any_replicate": int(
                    thumb_three_matrix.any(dim=1).sum().item()
                ),
                "candidates_reaching_thumb_and_three_all_replicates": int(
                    thumb_three_matrix.all(dim=1).sum().item()
                ),
                "candidates_reaching_three_contacts_any_replicate": int(
                    (legal_peak_matrix.amax(dim=1) >= POWER_REQUIRED_OTHER_CONTACTS)
                    .sum()
                    .item()
                ),
                "candidates_reaching_three_contacts_all_replicates": int(
                    (legal_peak_matrix.amin(dim=1) >= POWER_REQUIRED_OTHER_CONTACTS)
                    .sum()
                    .item()
                ),
                "power_grasp_quality_peak_max": float(q_grasp_peak.max().item()),
                "power_grasp_quality_peak_robust_max": float(
                    q_grasp_peak_matrix.amin(dim=1).max().item()
                ),
                "native_latch_environments": int(
                    native_power_latched_seen.sum().item()
                ),
                "candidates_with_native_latch_any_replicate": int(
                    native_latch_matrix.any(dim=1).sum().item()
                ),
                "candidates_with_native_latch_all_replicates": int(
                    native_latch_matrix.all(dim=1).sum().item()
                ),
                "native_stable_streak_peak_max": int(
                    native_stable_streak_peak.max().item()
                ),
                "native_latch_confirm_steps_peak_max": int(
                    native_latch_confirm_steps_peak.max().item()
                ),
                "native_failure_reason_counts": {
                    key: int(value.sum().item())
                    for key, value in native_terminal_reasons.items()
                },
                "native_failure_reason_counts_after_latch": {
                    key: int((value & native_power_latched_seen).sum().item())
                    for key, value in native_terminal_reasons.items()
                },
                "best_progress_candidate": {
                    "candidate_index": progress_index,
                    "progress_score": float(
                        progress_candidate_score[progress_index].item()
                    ),
                    "search_parameter": candidate_parameter[progress_index]
                    .detach()
                    .cpu()
                    .tolist(),
                    "native_stable_streak_peak_per_replicate": (
                        native_stable_streak_peak[progress_slice].cpu().tolist()
                    ),
                    "native_power_latched_seen_per_replicate": (
                        native_power_latched_seen[progress_slice].cpu().tolist()
                    ),
                    "native_latch_confirm_steps_peak_per_replicate": (
                        native_latch_confirm_steps_peak[progress_slice]
                        .cpu()
                        .tolist()
                    ),
                    "legal_other_contact_peak_per_replicate": (
                        legal_other_peak[progress_slice].cpu().tolist()
                    ),
                    "thumb_and_three_seen_per_replicate": (
                        thumb_and_three_seen[progress_slice].cpu().tolist()
                    ),
                    "power_grasp_quality_peak_per_replicate": (
                        q_grasp_peak[progress_slice].cpu().tolist()
                    ),
                    "native_failure_per_replicate": (
                        native_failure_seen[progress_slice].cpu().tolist()
                    ),
                    "native_timeout_per_replicate": (
                        native_timeout_seen[progress_slice].cpu().tolist()
                    ),
                    "dense_score_per_replicate": (
                        env_score[progress_slice].cpu().tolist()
                    ),
                    "native_terminal_reasons": {
                        key: value[progress_slice].cpu().tolist()
                        for key, value in native_terminal_reasons.items()
                    },
                },
            }
        )

        top = torch.topk(candidate_score, k=elite_count, largest=True)
        elite = candidate_parameter[top.indices]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0, unbiased=False).clamp(0.035, 0.75)

        def candidate_record(index: int) -> dict:
            start = index * replicates
            stop = start + replicates
            sl = slice(start, stop)
            first_steps = first_success_step[sl]
            robust_first_step = (
                int(first_steps.max().item())
                if bool((first_steps >= 0).all())
                else -1
            )
            robust_force_peak = force_peak[sl].amax(dim=0)
            record = {
                "iteration": iteration + 1,
                "score": float(candidate_score[index].item()),
                "replicates": replicates,
                "strict_replicates": int(strict_env_pass[sl].sum().item()),
                "replicate_strict_pass": strict_env_pass[sl].cpu().tolist(),
                "latent": hand_latent[start].detach().cpu().tolist(),
                "target": candidate_hand_target[start]
                .detach()
                .cpu()
                .tolist(),
                "search_parameter": candidate_parameter[index].detach().cpu().tolist(),
                "arm_parameter": arm_parameter[start].detach().cpu().tolist(),
                "arm_delta_target_rad": arm_delta_actual[start]
                .detach()
                .cpu()
                .tolist(),
                "arm_target": candidate_joint_target[start, u._arm_ids_t]
                .detach()
                .cpu()
                .tolist(),
                "arm_delta_abs_max_rad": float(
                    arm_delta_actual[start].abs().max().item()
                ),
                "strict_power_close_pass": bool(strict_group_pass[index].item()),
                "first_success_step": robust_first_step,
                "stable_streak_at_end": int(stable_streak[sl].min().item()),
                "stable_streak_at_end_per_replicate": stable_streak[sl].cpu().tolist(),
                "stable_streak_peak": int(stable_streak_peak[sl].min().item()),
                "stable_streak_peak_per_replicate": stable_streak_peak[sl]
                .cpu()
                .tolist(),
                "power_latch_confirm_peak": int(
                    power_latch_confirm_peak[sl].min().item()
                ),
                "power_latched_at_end": bool(power_is_grasped[sl].all().item()),
                "stable_fraction": float(stable_frac[sl].mean().item()),
                "q_close": float(q_close_mean[sl].mean().item()),
                "q_close_peak_per_replicate": q_close_peak[sl].cpu().tolist(),
                "q_wrap": float(q_wrap_mean[sl].mean().item()),
                "q_wrap_peak_per_replicate": q_wrap_peak[sl].cpu().tolist(),
                "q_grasp": float(q_grasp_mean[sl].mean().item()),
                "q_grasp_per_replicate": q_grasp_mean[sl].cpu().tolist(),
                "q_grasp_peak_min_replicate": float(
                    q_grasp_peak[sl].min().item()
                ),
                "q_grasp_peak_max_replicate": float(
                    q_grasp_peak[sl].max().item()
                ),
                "q_grasp_peak_per_replicate": q_grasp_peak[sl].cpu().tolist(),
                "hold_quality": float(hold_mean[sl].mean().item()),
                "hold_quality_peak_per_replicate": hold_peak[sl].cpu().tolist(),
                "thumb_fraction": float(thumb_frac[sl].mean().item()),
                "legal_other_mean": float(legal_other_mean[sl].mean().item()),
                "legal_other_contact_peak_min_replicate": int(
                    legal_other_peak[sl].min().item()
                ),
                "legal_other_contact_peak_max_replicate": int(
                    legal_other_peak[sl].max().item()
                ),
                "legal_other_contact_peak_per_replicate": (
                    legal_other_peak[sl].cpu().tolist()
                ),
                "thumb_and_third_fraction": float(
                    thumb_and_third_frac[sl].mean().item()
                ),
                "thumb_and_fourth_fraction": float(
                    thumb_and_fourth_frac[sl].mean().item()
                ),
                "thumb_and_three_seen_per_replicate": (
                    thumb_and_three_seen[sl].cpu().tolist()
                ),
                "clearance_peak": float(clearance_peak[sl].max().item()),
                "clearance_min": float(clearance_min[sl].min().item()),
                "xy_drift_peak": float(xy_drift_peak[sl].max().item()),
                "rotation_drift_peak_rad": float(
                    rotation_drift_peak[sl].max().item()
                ),
                "arm_table_clearance_min": float(
                    arm_table_clearance_min[sl].min().item()
                ),
                "arm_tracking_error_peak_rad": float(
                    arm_tracking_error_peak[sl].max().item()
                ),
                "native_terminated_seen": bool(
                    native_terminated_seen[sl].any().item()
                ),
                "native_timeout_seen": bool(native_timeout_seen[sl].any().item()),
                "force_peak": robust_force_peak.detach().cpu().tolist(),
            }
            if args_cli.coupled_power_contract:
                record.update(
                    {
                        "pass_authority": (
                            "conservative_coupled_teacher_audit_v1"
                        ),
                        "native_option_success_all_replicates": bool(
                            native_success_seen[sl].all().item()
                        ),
                        "conservative_teacher_audit_pass": bool(
                            strict_group_pass[index].item()
                        ),
                        "native_success_replicates": int(
                            native_success_seen[sl].sum().item()
                        ),
                        "native_success_per_replicate": (
                            native_success_seen[sl].cpu().tolist()
                        ),
                        "native_power_latched_seen_per_replicate": (
                            native_power_latched_seen[sl].cpu().tolist()
                        ),
                        "native_latch_confirm_steps_peak_per_replicate": (
                            native_latch_confirm_steps_peak[sl].cpu().tolist()
                        ),
                        "native_failure_per_replicate": (
                            native_failure_seen[sl].cpu().tolist()
                        ),
                        "native_timeout_per_replicate": (
                            native_timeout_seen[sl].cpu().tolist()
                        ),
                        "native_terminal_seen_per_replicate": (
                            native_terminal_seen[sl].cpu().tolist()
                        ),
                        "native_first_terminal_step_per_replicate": (
                            native_first_terminal_step[sl].cpu().tolist()
                        ),
                        "native_terminal_stable_steps_per_replicate": (
                            native_terminal_stable_steps[sl].cpu().tolist()
                        ),
                        "native_terminal_power_is_grasped_per_replicate": (
                            native_terminal_power_is_grasped[sl].cpu().tolist()
                        ),
                        "native_terminal_thumb_contact_per_replicate": (
                            native_terminal_thumb_contact[sl].cpu().tolist()
                        ),
                        "native_terminal_legal_other_per_replicate": (
                            native_terminal_legal_other[sl].cpu().tolist()
                        ),
                        "native_terminal_latch_confirm_steps_per_replicate": (
                            native_terminal_latch_confirm_steps[sl].cpu().tolist()
                        ),
                        "native_terminal_power_grasp_quality_per_replicate": (
                            native_terminal_power_grasp_quality[sl].cpu().tolist()
                        ),
                        "native_terminal_hold_quality_per_replicate": (
                            native_terminal_hold_quality[sl].cpu().tolist()
                        ),
                        "native_terminal_max_force_per_replicate": (
                            native_terminal_max_force[sl].cpu().tolist()
                        ),
                        "native_terminal_align_active_per_replicate": (
                            native_terminal_align_active[sl].cpu().tolist()
                        ),
                        "evaluation_frames_per_replicate": (
                            eval_count[sl].long().cpu().tolist()
                        ),
                        "rollout_steps_executed": rollout_steps_executed,
                        "arm_target_saturated_per_replicate": (
                            arm_target_saturated_seen[sl].cpu().tolist()
                        ),
                        "native_terminal_reasons": {
                            key: value[sl].cpu().tolist()
                            for key, value in native_terminal_reasons.items()
                        },
                    }
                )
            return record

        best_index = int(torch.argmax(candidate_score).item())
        current_any = candidate_record(best_index)
        if best_any is None or current_any["score"] > best_any["score"]:
            best_any = current_any
        pass_ids = strict_group_pass.nonzero(as_tuple=False).squeeze(-1)
        if pass_ids.numel() > 0:
            local = int(torch.argmax(candidate_score[pass_ids]).item())
            current_pass = candidate_record(int(pass_ids[local].item()))
            if best_pass is None or current_pass["score"] > best_pass["score"]:
                best_pass = current_pass
        native_ids = native_group_pass.nonzero(as_tuple=False).squeeze(-1)
        if args_cli.coupled_power_contract and native_ids.numel() > 0:
            local = int(torch.argmax(candidate_score[native_ids]).item())
            current_native = candidate_record(int(native_ids[local].item()))
            if best_native is None or current_native["score"] > best_native["score"]:
                best_native = current_native

        report = best_pass if best_pass is not None else best_any
        assert report is not None
        print(
            f"iter {iteration + 1:02d}: pass={int(strict_group_pass.sum().item())}/"
            f"{num_candidates} native={int(native_group_pass.sum().item())}/"
            f"{num_candidates} "
            f"best_score={report['score']:.3f} stable={report['stable_streak_peak']} "
            f"legal={report['legal_other_mean']:.2f} q={report['q_grasp']:.3f} "
            f"thumb+3={report['thumb_and_third_fraction']:.3f} "
            f"pop_legal_peak={search_trace[-1]['legal_other_contact_peak_max']} "
            f"latch_envs={search_trace[-1]['native_latch_environments']} "
            f"F={max(report['force_peak']):.2f}N "
            f"xy={report['xy_drift_peak']:.4f}m "
            f"rot={report['rotation_drift_peak_rad']:.3f}rad "
            f"arm_d={report['arm_delta_abs_max_rad']:.3f}rad "
            f"track={report['arm_tracking_error_peak_rad']:.3f}rad",
            flush=True,
        )

    search_seed_result = best_pass or best_native or best_any
    assert search_seed_result is not None
    authoritative_result = (
        best_pass if args_cli.coupled_power_contract else search_seed_result
    )
    effective_static_pregrasp = None
    if args_cli.coupled_power_contract:
        assert curriculum_boundary is not None
        effective_static_pregrasp = {}
        for key, value in curriculum_boundary.items():
            if not isinstance(value, torch.Tensor) or value.shape[0] != 1:
                continue
            row = value[0]
            effective_static_pregrasp[key] = (
                row.item() if row.ndim == 0 else row.tolist()
            )
    output = {
        "format_version": 1,
        "contract": (
            "strict_power_close_coupled_teacher_v1"
            if args_cli.coupled_power_contract
            else (
                "strict_power_close_arm_micro7_hybrid14_public_v1"
                if args_cli.public_controller
                else (
                    "strict_power_close_arm_micro7_hybrid14_oracle_v1"
                    if search_arm_micro
                    else "strict_power_close_hybrid14_v1"
                )
            )
        ),
        "controller": (
            "coupled_native_align_close_with_runtime_shields"
            if args_cli.coupled_power_contract
            else (
                "public_incremental_arm_hybrid14_with_runtime_shields"
                if args_cli.public_controller
                else (
                    "oracle_direct_joint_target_requires_public_replay"
                    if search_arm_micro
                    else "public_hybrid14_zero_arm"
                )
            )
        ),
        "task_mode": (
            "coupled_power_align_close_option_v1"
            if args_cli.coupled_power_contract
            else None
        ),
        "observation_contract": (
            "pick_tool_coupled_power_align_close_state131_v1"
            if args_cli.coupled_power_contract
            else None
        ),
        "observation_dim": expected_observation_dim,
        "action_dim": ARM_DIM + HAND_DIM,
        "action_layout": "arm_delta7|crossdex_token9|distal_residual5",
        "action_projection": "identity_v1",
        "pass_authority": (
            "conservative_coupled_teacher_audit_v1"
            if args_cli.coupled_power_contract
            else "power_close_search_contract.strict_power_close_pass"
        ),
        "native_success_authority": (
            "pick_tool_terminal.power_close_option_success"
            if args_cli.coupled_power_contract
            else None
        ),
        "search_parameter_layout": (
            "normalized_arm_target_offset7|hybrid14"
            if search_arm_micro
            else "hybrid14"
        ),
        "search_parameter_is_environment_action": False,
        "search_parameter_projection": (
            "time_varying_arm_feedback_plus_hybrid14"
            if search_arm_micro
            else "prepend_zero_arm7"
        ),
        "arm_feedback_formula": (
            "clip((smoothstep_target-current_target)/(action_scale*ema*0.2),-1,1)"
            if args_cli.coupled_power_contract
            else None
        ),
        "seed": args_cli.seed,
        "population": num_envs,
        "candidate_count": num_candidates,
        "replicates_per_candidate": replicates,
        "iterations": args_cli.iterations,
        "align_steps": args_cli.align_steps,
        "arm_delta_limit_rad": args_cli.arm_delta_limit_rad,
        "arm_delta_penalty": args_cli.arm_delta_penalty,
        "initial_hold_latent": (
            initial_hold_action[0].detach().cpu().tolist()
            if initial_hold_action is not None
            else None
        ),
        "initial_hold_decode_error_rad": initial_hold_decode_error,
        "close_steps": args_cli.close_steps,
        "eval_steps": args_cli.eval_steps,
        "rollout_step_limit": (
            int(u.max_episode_length)
            if args_cli.coupled_power_contract
            else args_cli.align_steps + args_cli.close_steps + args_cli.eval_steps
        ),
        "coupled_dense_score_phase": (
            "all_native_close_frames_until_first_terminal"
            if args_cli.coupled_power_contract
            else None
        ),
        "coupled_failure_ranking": (
            "drop_or_unsafe_force_hard_reject_pose_escape_soft_rank_v1"
            if args_cli.coupled_power_contract
            else None
        ),
        "coupled_close_eval_args_control_rollout": (
            False if args_cli.coupled_power_contract else None
        ),
        "episode_length_s": cfg.episode_length_s,
        "coupled_phase_contract": (
            {
                "align_steps": cfg.coupled_power_align_steps,
                "arm_action_multiplier": cfg.coupled_power_arm_action_multiplier,
                "arm_target_limit_rad": cfg.coupled_power_arm_target_limit,
                "align_hand_action": "masked_hold_target",
                "close_arm_action": "masked_frozen_target",
            }
            if args_cli.coupled_power_contract
            else None
        ),
        "curriculum_dataset": (
            str(curriculum_path) if curriculum_path is not None else None
        ),
        "curriculum_sha256": curriculum_sha256,
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "initial_action_input": str(initial_path),
        "initial_action_sha256": _sha256(initial_path),
        "pregrasp": pregrasp if not args_cli.coupled_power_contract else None,
        "source_pregrasp": source_pregrasp if args_cli.coupled_power_contract else None,
        "effective_static_pregrasp": effective_static_pregrasp,
        "hand_joint_names": [u.robot.joint_names[i] for i in u._hand_ids_t.tolist()],
        "arm_joint_names": [u.robot.joint_names[i] for i in u._arm_ids_t.tolist()],
        "fingertip_force_order": list(u.ee_names),
        "thresholds": {
            "required_legal_other_contacts": POWER_REQUIRED_OTHER_CONTACTS,
            "power_grasp_quality": POWER_GRASP_QUALITY_MIN,
            "hold_quality": POWER_HOLD_QUALITY_MIN,
            "safe_force_n": POWER_FORCE_LIMIT,
            "confirm_steps": POWER_STABLE_FRAMES,
            "power_latch_confirm_steps": POWER_LATCH_CONFIRM_FRAMES,
            "power_latch_release_steps": POWER_LATCH_RELEASE_FRAMES,
            "unlatched_lift_m": POWER_CLEARANCE_LIMIT,
            "horizontal_drift_m": POWER_XY_DRIFT_LIMIT,
            "rotation_drift_rad": POWER_ROTATION_DRIFT_LIMIT,
        },
        "search_trace": search_trace,
        # Coupled ``result`` is deliberately fail-closed: only a conservative teacher pass may
        # occupy it.  Native-only success is retained separately, while ``search_seed_result`` may
        # continue CEM but must never be promoted to behavioral-cloning supervision.
        "result": authoritative_result,
        "teacher_result": best_pass if args_cli.coupled_power_contract else None,
        "native_result": best_native,
        "search_seed_result": (
            search_seed_result if args_cli.coupled_power_contract else None
        ),
        # Coupled parameters require a time-varying arm feedback controller and must never be
        # consumed by the legacy hybrid14 replay path.
        "results": (
            {"coupled_align_close21": search_seed_result}
            if args_cli.coupled_power_contract
            else {"hybrid14": search_seed_result}
        ),
    }
    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        f"wrote {output_path} strict_pass={best_pass is not None}",
        flush=True,
    )
    env.close()
    if args_cli.require_pass and best_pass is None:
        raise RuntimeError("CEM did not find a strict power-close action")


if __name__ == "__main__":
    main()
    simulation_app.close()
