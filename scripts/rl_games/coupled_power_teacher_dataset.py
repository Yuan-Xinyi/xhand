#!/usr/bin/env python3
"""Collect strict 131-D/21-D ALIGN->CLOSE power-grasp demonstrations.

The collector replays one fail-closed coupled CEM teacher from the native
``close_start`` curriculum boundary.  It saves only episodes that terminate in
the environment's power-close success and independently pass the conservative
trajectory audit.  The returned observation of a terminal step is an automatic
reset observation, so it is deliberately never serialized; the pre-step
observation and the terminal-producing action are retained.

This dataset bootstraps the missing grasp-closure option.  It is not evidence
of, and does not contain supervision for, a 20 cm lift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--teacher_artifact",
    type=Path,
    required=True,
    help="strict_power_close_coupled_teacher_v1 CEM artifact",
)
parser.add_argument(
    "--curriculum_dataset",
    type=Path,
    required=True,
    help="one-state coupled_power_static_close_start_v1 dataset",
)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--minimum_successes", type=int, default=16)
parser.add_argument("--seed", type=int, default=226)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("/tmp/pick_tool_coupled_power_teacher.pt"),
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="replace an existing output file atomically",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs < 1:
    parser.error("--num_envs must be positive")
if args_cli.minimum_successes < 1 or args_cli.minimum_successes > args_cli.num_envs:
    parser.error("--minimum_successes must lie in [1, num_envs]")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401
from power_close_search_contract import (
    POWER_CLEARANCE_LIMIT,
    POWER_FORCE_LIMIT,
    POWER_GRASP_QUALITY_MIN,
    POWER_HOLD_QUALITY_MIN,
    POWER_REQUIRED_OTHER_CONTACTS,
    POWER_ROTATION_DRIFT_LIMIT,
    POWER_STABLE_FRAMES,
    POWER_XY_DRIFT_LIMIT,
    conservative_coupled_teacher_pass,
)
from xhand_inhand.tasks.direct.pick_tool_token.hybrid_action import (
    coupled_align_phase_state,
)


ARM_DIM = 7
HAND_DIM = 14
ACTION_DIM = ARM_DIM + HAND_DIM
OBSERVATION_DIM = 131
ALIGN_STEPS = 24

ARTIFACT_CONTRACT = "strict_power_close_coupled_teacher_v1"
CURRICULUM_CONTRACT = "coupled_power_static_close_start_v1"
TASK_MODE = "coupled_power_align_close_option_v1"
OBSERVATION_CONTRACT = "pick_tool_coupled_power_align_close_state131_v1"
OBSERVATION_LAYOUT = (
    "legacy_prefix87|distal_action5|grasp_transport23|coupled_state16"
)
ACTION_LAYOUT = "arm_delta7|crossdex_token9|distal_residual5"
PASS_AUTHORITY = "conservative_coupled_teacher_audit_v1"
PHASE_NAMES = ["align", "close_unlatched", "hold_latched"]

PHASE_ALIGN = 0
PHASE_CLOSE_UNLATCHED = 1
PHASE_HOLD_LATCHED = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact(mapping: Mapping, key: str, expected: object, source: str) -> None:
    value = mapping.get(key)
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{source}.{key}={value!r}, expected {expected!r}")


def _require_finite_list(
    mapping: Mapping,
    key: str,
    length: int,
    source: str,
) -> list[float]:
    value = mapping.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{source}.{key} must be a list of length {length}")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value
    ):
        raise ValueError(f"{source}.{key} must contain only finite numbers")
    return [float(item) for item in value]


def _require_lower_sha256(value: object, source: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{source} must be a lowercase hexadecimal SHA256")
    return value


def _validate_teacher_artifact(
    path: Path,
    *,
    curriculum_sha256: str,
) -> tuple[dict, list[float], list[float]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise TypeError("teacher artifact root must be a mapping")
    expected = {
        "format_version": 1,
        "contract": ARTIFACT_CONTRACT,
        "controller": "coupled_native_align_close_with_runtime_shields",
        "task_mode": TASK_MODE,
        "observation_contract": OBSERVATION_CONTRACT,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "action_layout": ACTION_LAYOUT,
        "action_projection": "identity_v1",
        "pass_authority": PASS_AUTHORITY,
        "native_success_authority": (
            "pick_tool_terminal.power_close_option_success"
        ),
        "search_parameter_layout": "normalized_arm_target_offset7|hybrid14",
        "search_parameter_is_environment_action": False,
        "search_parameter_projection": (
            "time_varying_arm_feedback_plus_hybrid14"
        ),
        "align_steps": ALIGN_STEPS,
        "arm_delta_limit_rad": 0.12,
        "episode_length_s": 3.0,
    }
    for key, value in expected.items():
        _require_exact(artifact, key, value, "artifact")
    if _require_lower_sha256(
        artifact.get("curriculum_sha256"), "artifact.curriculum_sha256"
    ) != curriculum_sha256:
        raise ValueError("teacher artifact and supplied curriculum SHA256 disagree")

    phase_contract = artifact.get("coupled_phase_contract")
    if not isinstance(phase_contract, Mapping):
        raise TypeError("artifact.coupled_phase_contract must be a mapping")
    for key, value in {
        "align_steps": ALIGN_STEPS,
        "arm_action_multiplier": 0.2,
        "arm_target_limit_rad": 0.12,
        "align_hand_action": "masked_hold_target",
        "close_arm_action": "masked_frozen_target",
    }.items():
        _require_exact(phase_contract, key, value, "artifact.coupled_phase_contract")

    thresholds = artifact.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise TypeError("artifact.thresholds must be a mapping")
    for key, value in {
        "required_legal_other_contacts": POWER_REQUIRED_OTHER_CONTACTS,
        "power_grasp_quality": POWER_GRASP_QUALITY_MIN,
        "hold_quality": POWER_HOLD_QUALITY_MIN,
        "safe_force_n": POWER_FORCE_LIMIT,
        "confirm_steps": POWER_STABLE_FRAMES,
        "unlatched_lift_m": POWER_CLEARANCE_LIMIT,
        "horizontal_drift_m": POWER_XY_DRIFT_LIMIT,
        "rotation_drift_rad": POWER_ROTATION_DRIFT_LIMIT,
    }.items():
        _require_exact(thresholds, key, value, "artifact.thresholds")

    teacher = artifact.get("teacher_result")
    if not isinstance(teacher, dict):
        raise ValueError("teacher artifact is fail-closed: teacher_result is absent")
    if artifact.get("result") != teacher:
        raise ValueError("artifact.result must equal its authoritative teacher_result")
    for key in (
        "strict_power_close_pass",
        "conservative_teacher_audit_pass",
        "native_option_success_all_replicates",
    ):
        _require_exact(teacher, key, True, "artifact.teacher_result")
    _require_exact(teacher, "pass_authority", PASS_AUTHORITY, "artifact.teacher_result")
    replicates = teacher.get("replicates")
    strict_replicates = teacher.get("strict_replicates")
    if (
        not isinstance(replicates, int)
        or isinstance(replicates, bool)
        or replicates < 1
        or strict_replicates != replicates
    ):
        raise ValueError("teacher_result must pass every recorded physical replicate")
    for key in ("replicate_strict_pass", "native_success_per_replicate"):
        values = teacher.get(key)
        if (
            not isinstance(values, list)
            or len(values) != replicates
            or any(value is not True for value in values)
        ):
            raise ValueError(f"teacher_result.{key} must be all true")

    latent = _require_finite_list(teacher, "latent", HAND_DIM, "artifact.teacher_result")
    arm_delta = _require_finite_list(
        teacher, "arm_delta_target_rad", ARM_DIM, "artifact.teacher_result"
    )
    if max(abs(value) for value in latent) > 1.0:
        raise ValueError("teacher latent exceeds normalized [-1, 1]")
    if max(abs(value) for value in arm_delta) > 0.12 + 1.0e-6:
        raise ValueError("teacher arm target offset exceeds the coupled 0.12 rad envelope")
    declared_max = teacher.get("arm_delta_abs_max_rad")
    if not isinstance(declared_max, (int, float)) or isinstance(declared_max, bool):
        raise TypeError("teacher arm_delta_abs_max_rad must be numeric")
    if abs(float(declared_max) - max(abs(value) for value in arm_delta)) > 1.0e-6:
        raise ValueError("teacher arm_delta_abs_max_rad disagrees with arm_delta_target_rad")
    return artifact, latent, arm_delta


def _validate_curriculum(path: Path) -> dict:
    curriculum = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(curriculum, dict):
        raise TypeError("curriculum root must be a dictionary")
    metadata = curriculum.get("meta")
    if not isinstance(metadata, Mapping):
        raise TypeError("curriculum metadata must be a mapping")
    for key, value in {
        "format_version": 1,
        "contract": CURRICULUM_CONTRACT,
        "task_mode": TASK_MODE,
        "observation_contract": OBSERVATION_CONTRACT,
        "action_layout": ACTION_LAYOUT,
        "top_k": 1,
    }.items():
        _require_exact(metadata, key, value, "curriculum.meta")
    boundaries = curriculum.get("boundaries")
    if not isinstance(boundaries, Mapping):
        raise TypeError("curriculum.boundaries must be a mapping")
    boundary = boundaries.get("close_start")
    if not isinstance(boundary, dict):
        raise TypeError("curriculum.boundaries.close_start must be a dictionary")
    tensor_shapes = {
        "joint_pos": (1, 19),
        "joint_vel": (1, 19),
        "dof_targets": (1, 19),
        "object_local_pos": (1, 3),
        "object_quat": (1, 4),
        "object_velocity": (1, 6),
        "last_action": (1, ACTION_DIM),
        "contact_steps": (1,),
        "lost_contact_steps": (1,),
        "is_grasped": (1,),
    }
    for key, shape in tensor_shapes.items():
        value = boundary.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            actual = tuple(value.shape) if isinstance(value, torch.Tensor) else None
            raise ValueError(f"curriculum close_start {key} expected {shape}, got {actual}")
    if not bool(torch.isfinite(boundary["joint_pos"]).all()):
        raise ValueError("curriculum close_start joint_pos contains non-finite values")
    if not torch.equal(boundary["joint_pos"], boundary["dof_targets"]):
        raise ValueError("static curriculum requires dof_targets == joint_pos")
    for key in ("joint_vel", "object_velocity", "last_action"):
        if bool((boundary[key] != 0).any()):
            raise ValueError(f"static curriculum requires exact-zero {key}")
    for key in ("contact_steps", "lost_contact_steps", "is_grasped"):
        if bool(boundary[key].any()):
            raise ValueError(f"static curriculum requires cleared {key}")
    return boundary


def _require_vector(
    mapping: Mapping,
    name: str,
    *,
    count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    value = mapping.get(name)
    if not isinstance(value, torch.Tensor) or value.shape != (count,):
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
        raise RuntimeError(f"invalid terminal field {name!r}: shape={shape}")
    if value.dtype != dtype or value.device != device:
        raise RuntimeError(
            f"invalid terminal field {name!r}: dtype={value.dtype} device={value.device}"
        )
    return value


def _policy_observation(observations: object, count: int) -> torch.Tensor:
    if not isinstance(observations, Mapping):
        raise TypeError("environment observation must be a mapping")
    value = observations.get("policy")
    if not isinstance(value, torch.Tensor) or value.shape != (
        count,
        OBSERVATION_DIM,
    ):
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
        raise RuntimeError(
            f"expected policy observation {(count, OBSERVATION_DIM)}, got {shape}"
        )
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("policy observation contains NaN or infinity")
    return value


def _atomic_torch_save(payload: dict, output: Path, *, overwrite: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {output}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


@torch.inference_mode()
def main() -> None:
    teacher_path = args_cli.teacher_artifact.expanduser().resolve()
    curriculum_path = args_cli.curriculum_dataset.expanduser().resolve()
    output_path = args_cli.output.expanduser().resolve()
    if not teacher_path.is_file():
        raise FileNotFoundError(teacher_path)
    if not curriculum_path.is_file():
        raise FileNotFoundError(curriculum_path)
    if output_path.exists() and not args_cli.overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {output_path}")

    teacher_sha256 = _sha256(teacher_path)
    curriculum_sha256 = _sha256(curriculum_path)
    boundary = _validate_curriculum(curriculum_path)
    artifact, latent_values, arm_delta_values = _validate_teacher_artifact(
        teacher_path,
        curriculum_sha256=curriculum_sha256,
    )

    torch.manual_seed(args_cli.seed)
    cfg = parse_env_cfg(
        "Pick-Tool-Token-Direct-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    cfg.seed = args_cli.seed
    cfg.close_option_mode = True
    cfg.power_close_option_mode = True
    cfg.coupled_power_align_close_option_mode = True
    cfg.observation_space = OBSERVATION_DIM
    cfg.state_space = OBSERVATION_DIM
    cfg.episode_length_s = 3.0
    cfg.curriculum_dataset = str(curriculum_path)
    cfg.curriculum_boundary = "close_start"
    cfg.curriculum_reset_probability = 1.0
    cfg.curriculum_joint_noise = 0.0

    env = None
    try:
        env = gym.make("Pick-Tool-Token-Direct-v0", cfg=cfg)
        u = env.unwrapped
        observations, _ = env.reset()
        observation = _policy_observation(observations, args_cli.num_envs)
        device = torch.device(u.device)
        count = args_cli.num_envs

        if cfg.action_space != ACTION_DIM:
            raise RuntimeError(f"expected action_space={ACTION_DIM}, got {cfg.action_space}")
        if u._n_arm != ARM_DIM or u._n_tokens + u._n_distal_residuals != HAND_DIM:
            raise RuntimeError("runtime action partition is not arm7|hand14")
        if int(cfg.coupled_power_align_steps) != ALIGN_STEPS:
            raise RuntimeError("runtime coupled ALIGN length differs from teacher contract")
        if float(cfg.coupled_power_arm_action_multiplier) != 0.2:
            raise RuntimeError("runtime coupled arm multiplier differs from teacher contract")
        if float(cfg.coupled_power_arm_target_limit) != 0.12:
            raise RuntimeError("runtime coupled arm envelope differs from teacher contract")
        if not bool(u._curriculum_reset_mask.all()):
            raise RuntimeError("one or more environments missed the mandatory curriculum reset")
        if bool((u.episode_length_buf != 0).any()):
            raise RuntimeError("native reset did not clear the episode clock")
        if bool(u._power_is_grasped.any()):
            raise RuntimeError("native reset did not clear the power-grasp latch")

        anchor = u._coupled_arm_anchor.clone()
        current_arm_target = u.dof_targets[:, u._arm_ids_t]
        if float((anchor - current_arm_target).abs().max()) != 0.0:
            raise RuntimeError("coupled arm anchor differs from the reset controller target")
        boundary_joint = boundary["joint_pos"].to(device=device)
        expected_anchor = boundary_joint[:, u._arm_ids_t].expand(count, -1)
        if float((anchor - expected_anchor).abs().max()) > 1.0e-6:
            raise RuntimeError("runtime coupled arm anchor disagrees with curriculum joint_pos")

        hand_latent = torch.tensor(
            latent_values, dtype=torch.float32, device=device
        ).unsqueeze(0).expand(count, -1)
        arm_delta = torch.tensor(
            arm_delta_values, dtype=torch.float32, device=device
        ).unsqueeze(0).expand(count, -1)
        desired_final_target = anchor + arm_delta
        arm_lower = u.dof_lower[:, u._arm_ids_t]
        arm_upper = u.dof_upper[:, u._arm_ids_t]
        if bool(
            (
                (desired_final_target < arm_lower - 1.0e-6)
                | (desired_final_target > arm_upper + 1.0e-6)
            ).any()
        ):
            raise ValueError("teacher arm target violates a physical joint limit")
        if float((desired_final_target - anchor).abs().max()) > 0.12 + 1.0e-6:
            raise ValueError("teacher arm target violates the coupled anchor envelope")
        artifact_arm_target = torch.tensor(
            _require_finite_list(
                artifact["teacher_result"],
                "arm_target",
                ARM_DIM,
                "artifact.teacher_result",
            ),
            dtype=torch.float32,
            device=device,
        )
        if float((desired_final_target[0] - artifact_arm_target).abs().max()) > 1.0e-5:
            raise ValueError("runtime anchor plus teacher delta disagrees with artifact arm_target")

        max_steps = int(u.max_episode_length)
        if max_steps < POWER_STABLE_FRAMES + ALIGN_STEPS:
            raise RuntimeError("native episode horizon is too short for the coupled contract")
        obs_history = torch.empty(
            (max_steps, count, OBSERVATION_DIM),
            dtype=torch.float32,
            device=device,
        )
        action_history = torch.empty(
            (max_steps, count, ACTION_DIM),
            dtype=torch.float32,
            device=device,
        )
        phase_history = torch.full(
            (max_steps, count), -1, dtype=torch.int64, device=device
        )

        active = torch.ones(count, dtype=torch.bool, device=device)
        terminal_seen = torch.zeros_like(active)
        native_success = torch.zeros_like(active)
        native_failure = torch.zeros_like(active)
        native_timeout = torch.zeros_like(active)
        terminal_step = torch.full((count,), -1, dtype=torch.int64, device=device)
        terminal_stable_steps = torch.zeros_like(terminal_step)
        terminal_power_is_grasped = torch.zeros_like(active)
        terminal_thumb_contact = torch.zeros_like(active)
        terminal_legal_other = torch.zeros_like(terminal_step)
        terminal_grasp_quality = torch.zeros(count, dtype=torch.float32, device=device)
        terminal_hold_quality = torch.zeros_like(terminal_grasp_quality)
        terminal_max_force = torch.zeros_like(terminal_grasp_quality)
        terminal_align_active = torch.zeros_like(active)
        episode_length = torch.zeros_like(terminal_step)

        initial_signals = u._compute_grasp_signals()
        trajectory_max_force = initial_signals["force_magnitude"].max(dim=-1).values
        trajectory_max_xy_drift = (
            u._object_com_position_w()[:, :2] - u._close_option_start_xy
        ).norm(dim=-1)
        trajectory_max_rotation_drift = u._close_option_rotation_vector().norm(dim=-1)
        trajectory_max_clearance = u._object_true_min_z() - u._table_surface_z
        arm_target_saturated = u._coupled_arm_target_saturated_ever.clone()
        for name, value in {
            "initial force": trajectory_max_force,
            "initial xy drift": trajectory_max_xy_drift,
            "initial rotation drift": trajectory_max_rotation_drift,
            "initial clearance": trajectory_max_clearance,
        }.items():
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"{name} contains NaN or infinity")

        print(
            f"COUPLED TEACHER COLLECT envs={count} horizon={max_steps} "
            f"seed={args_cli.seed} teacher={teacher_sha256[:12]} "
            f"curriculum={curriculum_sha256[:12]}",
            flush=True,
        )

        for step in range(max_steps):
            if not bool(active.any()):
                break
            step_active = active.clone()
            obs_history[step, step_active] = observation[step_active]

            align_active, _ = coupled_align_phase_state(
                u.episode_length_buf,
                u._power_is_grasped,
                align_steps=ALIGN_STEPS,
            )
            phase = torch.where(
                align_active,
                torch.full_like(terminal_step, PHASE_ALIGN),
                torch.where(
                    u._power_is_grasped,
                    torch.full_like(terminal_step, PHASE_HOLD_LATCHED),
                    torch.full_like(terminal_step, PHASE_CLOSE_UNLATCHED),
                ),
            )
            phase_history[step, step_active] = phase[step_active]

            action = torch.zeros(
                (count, ACTION_DIM), dtype=torch.float32, device=device
            )
            active_align = step_active & align_active
            active_close = step_active & (~align_active)
            if bool(active_align.any()):
                progress = torch.clamp(
                    (u.episode_length_buf.float() + 1.0) / float(ALIGN_STEPS),
                    0.0,
                    1.0,
                )
                blend = progress.square() * (3.0 - 2.0 * progress)
                desired = anchor + blend.unsqueeze(-1) * arm_delta
                denominator = (
                    float(cfg.action_scale)
                    * float(cfg.act_moving_average)
                    * float(cfg.coupled_power_arm_action_multiplier)
                )
                if not math.isfinite(denominator) or denominator <= 0.0:
                    raise RuntimeError("invalid coupled public-controller action scale")
                raw_arm = torch.clamp(
                    (desired - u.dof_targets[:, u._arm_ids_t]) / denominator,
                    -1.0,
                    1.0,
                )
                action[active_align, :ARM_DIM] = raw_arm[active_align]
            if bool(active_close.any()):
                action[active_close, ARM_DIM:] = hand_latent[active_close]
            action_history[step, step_active] = action[step_active]
            if not bool(torch.isfinite(action[step_active]).all()):
                raise FloatingPointError("teacher action contains NaN or infinity")

            next_observations, _, terminated, truncated, info = env.step(action)
            if (
                not isinstance(terminated, torch.Tensor)
                or terminated.dtype != torch.bool
                or terminated.shape != (count,)
                or not isinstance(truncated, torch.Tensor)
                or truncated.dtype != torch.bool
                or truncated.shape != (count,)
            ):
                raise RuntimeError("environment returned invalid done tensors")
            if not isinstance(info, Mapping):
                raise RuntimeError("environment info must be a mapping")
            terminal = info.get("pick_tool_terminal")
            if not isinstance(terminal, Mapping):
                raise RuntimeError("native coupled step omitted pick_tool_terminal")

            success = _require_vector(
                terminal,
                "power_close_option_success",
                count=count,
                dtype=torch.bool,
                device=device,
            )
            failure = _require_vector(
                terminal,
                "power_close_option_failure",
                count=count,
                dtype=torch.bool,
                device=device,
            )
            timeout = _require_vector(
                terminal,
                "power_close_option_timeout",
                count=count,
                dtype=torch.bool,
                device=device,
            )
            if not torch.equal(terminated, success | failure):
                raise RuntimeError("native terminated mask disagrees with option truth")
            if not torch.equal(truncated, timeout):
                raise RuntimeError("native truncated mask disagrees with option truth")
            if bool(((success & failure) | (success & timeout) | (failure & timeout)).any()):
                raise RuntimeError("native terminal classes are not mutually exclusive")
            for generic, specific in (
                ("success", success),
                ("failure", failure),
                ("time_out", timeout),
            ):
                if not torch.equal(
                    _require_vector(
                        terminal,
                        generic,
                        count=count,
                        dtype=torch.bool,
                        device=device,
                    ),
                    specific,
                ):
                    raise RuntimeError(f"terminal alias {generic!r} is inconsistent")

            current_max_force = _require_vector(
                terminal, "max_force", count=count, dtype=torch.float32, device=device
            )
            current_xy_drift = _require_vector(
                terminal,
                "coupled_power_xy_drift",
                count=count,
                dtype=torch.float32,
                device=device,
            )
            current_rotation_drift = _require_vector(
                terminal,
                "coupled_power_rotation_drift",
                count=count,
                dtype=torch.float32,
                device=device,
            )
            current_clearance = _require_vector(
                terminal,
                "coupled_power_true_clearance",
                count=count,
                dtype=torch.float32,
                device=device,
            )
            current_saturated = _require_vector(
                terminal,
                "coupled_power_arm_target_saturated",
                count=count,
                dtype=torch.bool,
                device=device,
            )
            for name, value in (
                ("max_force", current_max_force),
                ("xy_drift", current_xy_drift),
                ("rotation_drift", current_rotation_drift),
                ("clearance", current_clearance),
            ):
                if not bool(torch.isfinite(value[step_active]).all()):
                    raise FloatingPointError(f"terminal {name} contains NaN or infinity")
            trajectory_max_force = torch.where(
                step_active,
                torch.maximum(trajectory_max_force, current_max_force),
                trajectory_max_force,
            )
            trajectory_max_xy_drift = torch.where(
                step_active,
                torch.maximum(trajectory_max_xy_drift, current_xy_drift),
                trajectory_max_xy_drift,
            )
            trajectory_max_rotation_drift = torch.where(
                step_active,
                torch.maximum(trajectory_max_rotation_drift, current_rotation_drift),
                trajectory_max_rotation_drift,
            )
            trajectory_max_clearance = torch.where(
                step_active,
                torch.maximum(trajectory_max_clearance, current_clearance),
                trajectory_max_clearance,
            )
            arm_target_saturated.logical_or_(step_active & current_saturated)

            done = success | failure | timeout
            newly_done = step_active & done & (~terminal_seen)
            if bool(newly_done.any()):
                native_success[newly_done] = success[newly_done]
                native_failure[newly_done] = failure[newly_done]
                native_timeout[newly_done] = timeout[newly_done]
                terminal_step[newly_done] = step + 1
                episode_length[newly_done] = step + 1
                terminal_stable_steps[newly_done] = _require_vector(
                    terminal,
                    "power_close_option_stable_steps",
                    count=count,
                    dtype=torch.int64,
                    device=device,
                )[newly_done]
                terminal_power_is_grasped[newly_done] = _require_vector(
                    terminal,
                    "power_is_grasped",
                    count=count,
                    dtype=torch.bool,
                    device=device,
                )[newly_done]
                terminal_thumb_contact[newly_done] = _require_vector(
                    terminal,
                    "power_thumb_contact",
                    count=count,
                    dtype=torch.bool,
                    device=device,
                )[newly_done]
                terminal_legal_other[newly_done] = _require_vector(
                    terminal,
                    "power_legal_other_contact_count",
                    count=count,
                    dtype=torch.int64,
                    device=device,
                )[newly_done]
                terminal_grasp_quality[newly_done] = _require_vector(
                    terminal,
                    "power_grasp_quality",
                    count=count,
                    dtype=torch.float32,
                    device=device,
                )[newly_done]
                terminal_hold_quality[newly_done] = _require_vector(
                    terminal,
                    "hold_quality",
                    count=count,
                    dtype=torch.float32,
                    device=device,
                )[newly_done]
                terminal_max_force[newly_done] = current_max_force[newly_done]
                terminal_align_active[newly_done] = _require_vector(
                    terminal,
                    "coupled_power_align_active",
                    count=count,
                    dtype=torch.bool,
                    device=device,
                )[newly_done]
                terminal_seen.logical_or_(newly_done)
                active.logical_and_(~newly_done)

            observation = _policy_observation(next_observations, count)

        if not bool(terminal_seen.all()):
            missing = int((~terminal_seen).sum().item())
            raise RuntimeError(f"{missing} environments reached no native terminal class")
        if bool((episode_length <= 0).any()):
            raise RuntimeError("one or more terminal episodes have an invalid length")

        accepted = conservative_coupled_teacher_pass(
            native_success=native_success,
            native_failure=native_failure,
            native_timeout=native_timeout,
            terminal_stable_steps=terminal_stable_steps,
            terminal_power_is_grasped=terminal_power_is_grasped,
            terminal_thumb_contact=terminal_thumb_contact,
            terminal_legal_other_contacts=terminal_legal_other,
            terminal_power_grasp_quality=terminal_grasp_quality,
            terminal_hold_quality=terminal_hold_quality,
            terminal_max_force=terminal_max_force,
            terminal_align_active=terminal_align_active,
            trajectory_force_peak=trajectory_max_force,
            trajectory_xy_drift_peak=trajectory_max_xy_drift,
            trajectory_rotation_drift_peak=trajectory_max_rotation_drift,
            trajectory_clearance_peak=trajectory_max_clearance,
            arm_target_saturated=arm_target_saturated,
        )
        if bool(
            (native_success & (terminal_stable_steps != POWER_STABLE_FRAMES)).any()
        ):
            raise RuntimeError(
                "native success did not terminate at exactly the 15-frame stable boundary"
            )
        accepted_ids = accepted.nonzero(as_tuple=False).squeeze(-1)
        accepted_count = int(accepted_ids.numel())
        if accepted_count < args_cli.minimum_successes:
            raise RuntimeError(
                f"strict teacher yielded {accepted_count}/{count} accepted episodes; "
                f"minimum is {args_cli.minimum_successes}"
            )

        lengths = episode_length[accepted_ids]
        observation_chunks = []
        action_chunks = []
        phase_chunks = []
        offsets = [0]
        for env_id, length in zip(
            accepted_ids.detach().cpu().tolist(),
            lengths.detach().cpu().tolist(),
            strict=True,
        ):
            observation_chunks.append(obs_history[:length, env_id])
            action_chunks.append(action_history[:length, env_id])
            phase_chunks.append(phase_history[:length, env_id])
            offsets.append(offsets[-1] + length)
        saved_observation = torch.cat(observation_chunks, dim=0).detach().cpu().contiguous()
        saved_action = torch.cat(action_chunks, dim=0).detach().cpu().contiguous()
        saved_phase = torch.cat(phase_chunks, dim=0).detach().cpu().contiguous()
        saved_offsets = torch.tensor(offsets, dtype=torch.int64)
        saved_episode_id = torch.repeat_interleave(
            torch.arange(accepted_count, dtype=torch.int64),
            lengths.detach().cpu(),
        )
        if saved_observation.shape != (offsets[-1], OBSERVATION_DIM):
            raise RuntimeError("flattened observation shape is inconsistent")
        if saved_action.shape != (offsets[-1], ACTION_DIM):
            raise RuntimeError("flattened action shape is inconsistent")
        if saved_phase.shape != (offsets[-1],):
            raise RuntimeError("flattened phase shape is inconsistent")
        phase_values = sorted(int(value) for value in torch.unique(saved_phase).tolist())
        actual_phases = set(phase_values)
        if not {PHASE_ALIGN, PHASE_HOLD_LATCHED}.issubset(
            actual_phases
        ) or not actual_phases.issubset(
            {PHASE_ALIGN, PHASE_CLOSE_UNLATCHED, PHASE_HOLD_LATCHED}
        ):
            raise RuntimeError(f"accepted trajectories lack required phases: {phase_values}")
        for episode, (start, stop) in enumerate(
            zip(saved_offsets[:-1].tolist(), saved_offsets[1:].tolist(), strict=True)
        ):
            episode_phase = saved_phase[start:stop]
            if bool((episode_phase[1:] < episode_phase[:-1]).any()):
                raise RuntimeError(f"episode {episode} phase labels are not monotonic")
            if int(episode_phase[0]) != PHASE_ALIGN:
                raise RuntimeError(f"episode {episode} does not begin in ALIGN")
            if int(episode_phase[-1]) != PHASE_HOLD_LATCHED:
                raise RuntimeError(f"episode {episode} does not terminate in latched HOLD")
            # The action that creates the four-frame latch is labeled from its pre-step
            # unlatched observation, but contributes stable_count=1 post-step.  Therefore a
            # fastest valid 15-frame terminal trajectory has fourteen pre-step HOLD labels.
            minimum_hold_rows = POWER_STABLE_FRAMES - 1
            if int((episode_phase == PHASE_HOLD_LATCHED).sum()) < minimum_hold_rows:
                raise RuntimeError(
                    f"episode {episode} has fewer than {minimum_hold_rows} HOLD rows"
                )

        observation_align = saved_observation[:, 129]
        observation_latch = saved_observation[:, 106]
        expected_align = (saved_phase == PHASE_ALIGN).to(dtype=saved_observation.dtype)
        expected_latch = (saved_phase == PHASE_HOLD_LATCHED).to(
            dtype=saved_observation.dtype
        )
        if not torch.equal(observation_align, expected_align):
            raise RuntimeError("phase labels disagree with observation ALIGN-active bit 129")
        if not torch.equal(observation_latch, expected_latch):
            raise RuntimeError("phase labels disagree with observation power-latch bit 106")

        align_hand_max = float(
            saved_action[saved_phase == PHASE_ALIGN, ARM_DIM:].abs().max()
        )
        close_arm_max = float(
            saved_action[saved_phase != PHASE_ALIGN, :ARM_DIM].abs().max()
        )
        if align_hand_max != 0.0 or close_arm_max != 0.0:
            raise RuntimeError("canonical phase-inactive action dimensions are not exact zero")
        first_rows = saved_offsets[:-1]
        first_old_action = torch.cat(
            (
                saved_observation[first_rows, 70:86],
                saved_observation[first_rows, 87:92],
            ),
            dim=-1,
        )
        first_action_error = float(first_old_action.abs().max())
        if first_action_error != 0.0:
            raise RuntimeError(
                "curriculum reset observation carries a non-zero previous action: "
                f"max={first_action_error}"
            )

        selected = accepted_ids.detach().cpu()
        payload = {
            "obs": saved_observation,
            "action": saved_action,
            "phase": saved_phase,
            "episode_offsets": saved_offsets,
            "episode_id": saved_episode_id,
            "episode_success": torch.ones(accepted_count, dtype=torch.bool),
            "episode_native_success": native_success[accepted_ids].detach().cpu(),
            "episode_native_failure": native_failure[accepted_ids].detach().cpu(),
            "episode_native_timeout": native_timeout[accepted_ids].detach().cpu(),
            "episode_conservative_teacher_pass": accepted[accepted_ids].detach().cpu(),
            "episode_terminal_stable_steps": terminal_stable_steps[accepted_ids]
            .detach()
            .cpu(),
            "episode_terminal_power_is_grasped": terminal_power_is_grasped[
                accepted_ids
            ]
            .detach()
            .cpu(),
            "episode_terminal_thumb_contact": terminal_thumb_contact[accepted_ids]
            .detach()
            .cpu(),
            "episode_terminal_legal_other_contact_count": terminal_legal_other[
                accepted_ids
            ]
            .detach()
            .cpu(),
            "episode_terminal_power_grasp_quality": terminal_grasp_quality[
                accepted_ids
            ]
            .detach()
            .cpu(),
            "episode_terminal_hold_quality": terminal_hold_quality[accepted_ids]
            .detach()
            .cpu(),
            "episode_terminal_max_force": terminal_max_force[accepted_ids]
            .detach()
            .cpu(),
            "episode_trajectory_max_force": trajectory_max_force[accepted_ids]
            .detach()
            .cpu(),
            "episode_trajectory_max_xy_drift": trajectory_max_xy_drift[
                accepted_ids
            ]
            .detach()
            .cpu(),
            "episode_trajectory_max_rotation_drift": trajectory_max_rotation_drift[
                accepted_ids
            ]
            .detach()
            .cpu(),
            "episode_terminal_align_active": terminal_align_active[accepted_ids]
            .detach()
            .cpu(),
            "episode_trajectory_max_true_clearance": trajectory_max_clearance[
                accepted_ids
            ]
            .detach()
            .cpu(),
            "episode_arm_target_saturated": arm_target_saturated[accepted_ids]
            .detach()
            .cpu(),
            "episode_source_env_id": selected.to(dtype=torch.int64),
            "episode_terminal_step": terminal_step[accepted_ids].detach().cpu(),
            "meta": {
                "format_version": 1,
                "task_mode": TASK_MODE,
                "observation_dim": OBSERVATION_DIM,
                "observation_contract": OBSERVATION_CONTRACT,
                "observation_layout": OBSERVATION_LAYOUT,
                "action_dim": ACTION_DIM,
                "action_layout": ACTION_LAYOUT,
                "action_projection": "identity_v1",
                "action_semantics": (
                    "canonical_policy_action_before_phase_shield_v1"
                ),
                "collector": "successful_coupled_power_align_close_teacher",
                "dataset_phase": "align_close_hold",
                "phase_names": PHASE_NAMES,
                "teacher_probability": 1.0,
                "executed_teacher_fraction": 1.0,
                "align_hand_action_abs_max": align_hand_max,
                "close_arm_action_abs_max": close_arm_max,
                "first_observation_last_action_max_error": first_action_error,
                "terminal_observation": "not_saved_auto_reset_excluded_v1",
                "trajectory_acceptance": (
                    "native_success_and_conservative_teacher_audit_v1"
                ),
                "clearance_authority": (
                    "true_mesh_convex_hull_min_z_minus_table_v1"
                ),
                "teacher_artifact_sha256": teacher_sha256,
                "curriculum_dataset_sha256": curriculum_sha256,
                "teacher_artifact": str(teacher_path),
                "curriculum_dataset": str(curriculum_path),
                "seed": int(args_cli.seed),
                "attempted_episodes": count,
                "accepted_episodes": accepted_count,
                "transitions": int(saved_observation.shape[0]),
            },
        }
        _atomic_torch_save(payload, output_path, overwrite=args_cli.overwrite)
        output_sha256 = _sha256(output_path)
        summary = {
            "output": str(output_path),
            "output_sha256": output_sha256,
            "attempted_episodes": count,
            "native_successes": int(native_success.sum().item()),
            "native_failures": int(native_failure.sum().item()),
            "native_timeouts": int(native_timeout.sum().item()),
            "accepted_episodes": accepted_count,
            "acceptance_fraction": accepted_count / count,
            "transitions": int(saved_observation.shape[0]),
            "episode_length_min": int(lengths.min().item()),
            "episode_length_max": int(lengths.max().item()),
            "trajectory_force_peak_max": float(
                trajectory_max_force[accepted_ids].max().item()
            ),
            "trajectory_xy_drift_peak_max": float(
                trajectory_max_xy_drift[accepted_ids].max().item()
            ),
            "trajectory_rotation_drift_peak_max": float(
                trajectory_max_rotation_drift[accepted_ids].max().item()
            ),
            "trajectory_clearance_peak_max": float(
                trajectory_max_clearance[accepted_ids].max().item()
            ),
            "phase_counts": {
                PHASE_NAMES[value]: int((saved_phase == value).sum().item())
                for value in range(len(PHASE_NAMES))
            },
        }
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
