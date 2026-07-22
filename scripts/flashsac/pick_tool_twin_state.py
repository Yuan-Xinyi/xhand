#!/usr/bin/env python3
"""Simulation-independent contracts for a PickTool cross-slot A/A diagnostic.

The diagnostic clones a fresh, finger-object-contact-free full-task reset from
each even slot into its spatially adjacent odd slot, then runs both slots with
identical SEARCH actions.  The hammer still contacts the table, whose PhysX
manifold is not publicly copyable.  Therefore a clone is causal evidence only
if its complete A/A rollout independently passes; a failed run instead
falsifies cross-slot forking.  Sensor buffers are read-only evidence and this
module never attempts to manufacture a contact manifold.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch


OBSERVATION_DIM = 115
ACTION_DIM = 21

PROXIMITY_SLICE = slice(92, 97)
NONTHUMB_PROXIMITY_SLICE = slice(92, 96)
THUMB_PROXIMITY_INDEX = 96
FORCE_SLICE = slice(97, 102)
CLOSE_QUALITY_INDEX = 102
GRASP_LATCH_INDEX = 106
READINESS_MIN_SCORE = 0.10
READINESS_HOLD_STEPS = 4
READINESS_G_STRATUM_THRESHOLD = 0.35
READINESS_CLOSE_STRATUM_THRESHOLD = 0.20
READINESS_STRATUM_NAMES = (
    "low_g_low_close",
    "low_g_high_close",
    "high_g_low_close",
    "high_g_high_close",
)

# Per-environment buffers from the reset-state audit.  Derived geometry and
# observations are deliberately absent: they are rebuilt after the PhysX write.
TWIN_CACHE_FIELDS = (
    # DirectRLEnv episode/reset/reward state.
    "episode_length_buf",
    "reset_buf",
    "reset_terminated",
    "reset_time_outs",
    # Public action/goal history and the relative-position controller state.
    "actions",
    "prev_actions",
    "target_quat",
    "dof_targets",
    "_last_token_hand_target",
    "_last_distal_delta",
    "_last_raw_hand_target",
    # Full-task grasp/success and one-shot reward latches.
    "_contact_steps",
    "_lost_contact_steps",
    "_is_grasped",
    "_grasp_bonus_given",
    "_safe_grasp_steps",
    "_grasp_age",
    "_success_steps",
    "_is_success",
    "_success_paid",
    "_lift_bonus_given",
    # Tactile termination and anti-fling state.
    "_hard_force_steps",
    "_overforce_steps",
    "_unlatched_lift_failure",
    # Gamma-correct reward-potential history.
    "_prev_close_quality",
    "_prev_wrap_quality",
    "_prev_lift_potential",
    "_prev_close_option_stable_potential",
    "_potential_initialized",
)


@dataclass(frozen=True)
class TwinPairLayout:
    """Even-slot sources paired one-to-one with adjacent odd destinations."""

    source: torch.Tensor
    destination: torch.Tensor
    num_envs: int

    @property
    def pair_count(self) -> int:
        return int(self.source.numel())


@dataclass(frozen=True)
class PublicReadinessUpdate:
    """One update of the public-observation-only fork-candidate state."""

    score: torch.Tensor
    eligible: torch.Tensor
    ready_count: torch.Tensor
    trigger: torch.Tensor
    fork_used: torch.Tensor
    sticky: torch.Tensor
    stratum: torch.Tensor


def make_twin_pair_layout(
    num_envs: int, device: torch.device | str = "cpu"
) -> TwinPairLayout:
    """Build spatially adjacent even-source/odd-destination twin pairs."""

    if isinstance(num_envs, bool) or not isinstance(num_envs, int):
        raise TypeError("num_envs must be an integer")
    if num_envs < 2 or num_envs % 2 != 0:
        raise ValueError("num_envs must be positive, at least two, and even")
    source = torch.arange(0, num_envs, 2, dtype=torch.long, device=device)
    return TwinPairLayout(
        source=source,
        destination=source + 1,
        num_envs=num_envs,
    )


def update_public_readiness(
    observation: torch.Tensor,
    ready_count: torch.Tensor,
    fork_used: torch.Tensor,
) -> PublicReadinessUpdate:
    """Update the public 115-D readiness rule without private/time features.

    ``g`` is the minimum of thumb proximity and the second-best non-thumb
    proximity.  Four consecutive unlatched samples with ``g >= 0.10`` trigger
    exactly once; ``fork_used`` is then sticky for the remainder of the episode.
    """

    if not isinstance(observation, torch.Tensor) or observation.ndim != 2:
        raise TypeError("observation must be a rank-two torch.Tensor")
    if observation.shape[1] != OBSERVATION_DIM:
        raise ValueError(
            f"observation must have shape [N,{OBSERVATION_DIM}], got {tuple(observation.shape)}"
        )
    batch = observation.shape[0]
    for name, value, dtype in (
        ("ready_count", ready_count, torch.long),
        ("fork_used", fork_used, torch.bool),
    ):
        if not isinstance(value, torch.Tensor) or value.shape != (batch,):
            raise ValueError(f"{name} must have shape ({batch},)")
        if value.device != observation.device:
            raise ValueError(f"{name} must be on {observation.device}")
        if value.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}")
    if not observation.dtype.is_floating_point:
        raise TypeError("observation must be floating point")
    if not bool(torch.isfinite(observation).all()):
        raise FloatingPointError("observation contains NaN or infinity")
    if bool((ready_count < 0).any()) or bool((ready_count > READINESS_HOLD_STEPS).any()):
        raise ValueError("ready_count is outside [0, 4]")

    nonthumb = observation[:, NONTHUMB_PROXIMITY_SLICE]
    second_nonthumb = torch.topk(nonthumb, k=2, dim=-1, largest=True).values[:, 1]
    thumb = observation[:, THUMB_PROXIMITY_INDEX]
    score = torch.minimum(thumb, second_nonthumb)
    latch = observation[:, GRASP_LATCH_INDEX]
    if not bool(((latch == 0.0) | (latch == 1.0)).all()):
        raise RuntimeError("public grasp latch observation must be strictly binary")
    eligible = (latch == 0.0) & (score >= READINESS_MIN_SCORE)
    incremented = torch.clamp(ready_count + 1, max=READINESS_HOLD_STEPS)
    next_ready = torch.where(eligible, incremented, torch.zeros_like(ready_count))
    trigger = (next_ready == READINESS_HOLD_STEPS) & (~fork_used)
    next_fork_used = fork_used | trigger

    high_g = score >= READINESS_G_STRATUM_THRESHOLD
    high_close = (
        observation[:, CLOSE_QUALITY_INDEX]
        >= READINESS_CLOSE_STRATUM_THRESHOLD
    )
    stratum = high_g.long() * 2 + high_close.long()
    return PublicReadinessUpdate(
        score=score,
        eligible=eligible,
        ready_count=next_ready,
        trigger=trigger,
        fork_used=next_fork_used,
        sticky=next_fork_used,
        stratum=stratum,
    )


def replicate_source_actions(
    source_actions: torch.Tensor, layout: TwinPairLayout
) -> torch.Tensor:
    """Place one actor evaluation per source into both halves."""

    if not isinstance(source_actions, torch.Tensor) or source_actions.ndim != 2:
        raise TypeError("source_actions must be a rank-two torch.Tensor")
    if source_actions.shape != (layout.pair_count, ACTION_DIM):
        raise ValueError(
            f"source_actions must have shape ({layout.pair_count},{ACTION_DIM})"
        )
    if source_actions.device != layout.source.device:
        raise ValueError("source_actions and pair indices must share a device")
    action = source_actions.new_empty((layout.num_envs, ACTION_DIM))
    action.index_copy_(0, layout.source, source_actions)
    action.index_copy_(0, layout.destination, source_actions)
    return action


def paired_max_abs(value: torch.Tensor, layout: TwinPairLayout) -> float:
    """Return the maximum absolute source/destination difference."""

    if not isinstance(value, torch.Tensor) or value.ndim < 1:
        raise TypeError("paired value must be a non-scalar torch.Tensor")
    if value.shape[0] != layout.num_envs or value.device != layout.source.device:
        raise ValueError("paired value has an incompatible leading dimension or device")
    source = value.index_select(0, layout.source)
    destination = value.index_select(0, layout.destination)
    if value.dtype is torch.bool:
        return float((source != destination).any())
    if not (value.dtype.is_floating_point or value.dtype.is_complex):
        return float((source != destination).any())
    return float((source - destination).abs().max()) if source.numel() else 0.0


def _declared_dimension(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple) and len(shape) == 1:
        return int(shape[0])
    return None


def validate_full_task_contract(task: Any) -> None:
    """Fail closed unless this is the ordinary 115-observation/21-action task."""

    cfg = getattr(task, "cfg", None)
    if cfg is None:
        raise TypeError("PickTool task has no cfg")
    observation_dim = _declared_dimension(getattr(cfg, "observation_space", None))
    action_dim = _declared_dimension(getattr(cfg, "action_space", None))
    if observation_dim != OBSERVATION_DIM or action_dim != ACTION_DIM:
        raise RuntimeError(
            f"twin reset requires the full-task {OBSERVATION_DIM}/{ACTION_DIM} contract, "
            f"got {observation_dim}/{action_dim}"
        )
    option_flags = (
        "close_option_mode",
        "power_close_option_mode",
        "coupled_power_align_close_option_mode",
        "hold_arm_until_stable_grasp",
    )
    enabled = [name for name in option_flags if bool(getattr(cfg, name, False))]
    if enabled:
        raise RuntimeError(f"twin reset requires full_task; option flags are enabled: {enabled}")


def _require_env_tensor(task: Any, name: str, num_envs: int) -> torch.Tensor:
    value = getattr(task, name, None)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"required state {name!r} is not a torch.Tensor")
    if value.ndim < 1 or value.shape[0] != num_envs:
        raise ValueError(
            f"required state {name!r} must lead with num_envs={num_envs}, got {tuple(value.shape)}"
        )
    return value


def _require_zero(name: str, value: torch.Tensor, atol: float) -> None:
    if not value.numel():
        maximum = 0.0
    elif value.dtype is torch.bool:
        maximum = float(value.detach().any())
    else:
        maximum = float(value.detach().abs().max())
    if not math.isfinite(maximum) or maximum > atol:
        raise RuntimeError(f"fresh reset requires {name} <= {atol:g}, got {maximum:g}")


def _contact_reset_audit(
    task: Any,
    layout: TwinPairLayout,
    *,
    force_atol: float,
    timestamp_atol: float,
) -> dict[str, Any]:
    sensors = getattr(task, "_object_contact_sensors", None)
    if not isinstance(sensors, Mapping) or not sensors:
        raise TypeError("PickTool object contact sensor mapping is unavailable")
    per_sensor: dict[str, Any] = {}
    matrix_max = 0.0
    net_max = 0.0
    timestamp_max_error = 0.0
    last_update_max_error = 0.0
    for name in sorted(sensors):
        sensor = sensors[name]
        data = getattr(sensor, "data", None)
        matrix = getattr(data, "force_matrix_w", None)
        net = getattr(data, "net_forces_w", None)
        if not isinstance(matrix, torch.Tensor) or not isinstance(net, torch.Tensor):
            raise TypeError(f"contact sensor {name!r} lacks force_matrix_w/net_forces_w")
        for label, value in (("force_matrix_w", matrix), ("net_forces_w", net)):
            if value.ndim < 2 or value.shape[0] != layout.num_envs:
                raise ValueError(f"contact sensor {name}.{label} has invalid shape {tuple(value.shape)}")
            if value.device != layout.source.device:
                raise ValueError(f"contact sensor {name}.{label} is on the wrong device")
        current_matrix_max = float(matrix.abs().max())
        current_net_max = float(net.abs().max())
        matrix_max = max(matrix_max, current_matrix_max)
        net_max = max(net_max, current_net_max)

        timestamp = getattr(sensor, "_timestamp", None)
        last_update = getattr(sensor, "_timestamp_last_update", None)
        if not isinstance(timestamp, torch.Tensor) or not isinstance(last_update, torch.Tensor):
            raise TypeError(f"contact sensor {name!r} lacks timestamp tensors")
        timestamp_error = paired_max_abs(timestamp, layout)
        last_update_error = paired_max_abs(last_update, layout)
        timestamp_max_error = max(timestamp_max_error, timestamp_error)
        last_update_max_error = max(last_update_max_error, last_update_error)
        per_sensor[name] = {
            "force_matrix_abs_max": current_matrix_max,
            "net_force_abs_max": current_net_max,
            "timestamp_pair_abs_error": timestamp_error,
            "last_update_timestamp_pair_abs_error": last_update_error,
        }
    if matrix_max > force_atol or net_max > force_atol:
        raise RuntimeError(
            "fresh reset is not finger-object-contact-free: "
            f"force_matrix={matrix_max:g}, net={net_max:g}, tolerance={force_atol:g}"
        )
    if timestamp_max_error > timestamp_atol or last_update_max_error > timestamp_atol:
        raise RuntimeError(
            "fresh-reset contact sensor timestamps disagree across twin pairs: "
            f"timestamp={timestamp_max_error:g}, last_update={last_update_max_error:g}"
        )
    return {
        "force_matrix_abs_max": matrix_max,
        "net_force_abs_max": net_max,
        "timestamp_pair_abs_error": timestamp_max_error,
        "last_update_timestamp_pair_abs_error": last_update_max_error,
        "per_sensor": per_sensor,
    }


def copy_fresh_reset_to_twins(
    task: Any,
    layout: TwinPairLayout,
    *,
    reset_zero_atol: float = 1.0e-6,
    contact_force_atol: float = 1.0e-6,
    timestamp_atol: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Clone reset state for a fail-closed cross-slot diagnostic and rebuild obs.

    No simulation step and no ``scene.update(dt)`` occur here.  Finger-object
    sensor data is read only before and after the write; a nonzero/stale reset
    is rejected.  This cannot copy the existing hammer-table manifold.
    """

    validate_full_task_contract(task)
    if int(getattr(task, "num_envs", -1)) != layout.num_envs:
        raise ValueError("pair layout does not match task.num_envs")
    device = torch.device(str(getattr(task, "device")))
    if device != layout.source.device:
        raise ValueError("pair layout and task must share a device")
    if reset_zero_atol < 0.0 or contact_force_atol < 0.0 or timestamp_atol < 0.0:
        raise ValueError("fresh-reset tolerances must be non-negative")

    cache = {
        name: _require_env_tensor(task, name, layout.num_envs)
        for name in TWIN_CACHE_FIELDS
    }
    robot = getattr(task, "robot", None)
    object_asset = getattr(task, "object", None)
    scene = getattr(task, "scene", None)
    sim = getattr(task, "sim", None)
    if any(value is None for value in (robot, object_asset, scene, sim)):
        raise TypeError("PickTool robot/object/scene/sim handles are required")

    robot_data = getattr(robot, "data", None)
    object_data = getattr(object_asset, "data", None)
    joint_pos = getattr(robot_data, "joint_pos", None)
    joint_vel = getattr(robot_data, "joint_vel", None)
    joint_pos_target = getattr(robot_data, "joint_pos_target", None)
    joint_vel_target = getattr(robot_data, "joint_vel_target", None)
    joint_effort_target = getattr(robot_data, "joint_effort_target", None)
    root_link_pos = getattr(object_data, "root_link_pos_w", None)
    root_link_quat = getattr(object_data, "root_link_quat_w", None)
    root_com_lin_vel = getattr(object_data, "root_com_lin_vel_w", None)
    root_com_ang_vel = getattr(object_data, "root_com_ang_vel_w", None)
    physical = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "joint_pos_target": joint_pos_target,
        "joint_vel_target": joint_vel_target,
        "joint_effort_target": joint_effort_target,
        "object_root_link_pos_w": root_link_pos,
        "object_root_link_quat_w": root_link_quat,
        "object_root_com_lin_vel_w": root_com_lin_vel,
        "object_root_com_ang_vel_w": root_com_ang_vel,
    }
    for name, value in physical.items():
        if not isinstance(value, torch.Tensor) or value.shape[0] != layout.num_envs:
            shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
            raise TypeError(f"required physical state {name!r} is invalid: {shape}")
        if value.device != device:
            raise ValueError(f"physical state {name!r} is on {value.device}, expected {device}")

    # This helper is not a general mid-episode fork.  Enforce the audited fresh
    # reset boundary before touching simulator state.
    _require_zero("actions", cache["actions"], reset_zero_atol)
    _require_zero("prev_actions", cache["prev_actions"], reset_zero_atol)
    _require_zero("episode_length_buf", cache["episode_length_buf"], reset_zero_atol)
    _require_zero("reset_buf", cache["reset_buf"], reset_zero_atol)
    _require_zero("reset_terminated", cache["reset_terminated"], reset_zero_atol)
    _require_zero("reset_time_outs", cache["reset_time_outs"], reset_zero_atol)
    _require_zero("joint_vel", joint_vel, reset_zero_atol)
    _require_zero("joint_vel_target", joint_vel_target, reset_zero_atol)
    _require_zero("joint_effort_target", joint_effort_target, reset_zero_atol)
    _require_zero("object COM velocity", root_com_lin_vel, reset_zero_atol)
    _require_zero("object COM angular velocity", root_com_ang_vel, reset_zero_atol)
    for name in (
        "_contact_steps",
        "_lost_contact_steps",
        "_is_grasped",
        "_success_steps",
        "_is_success",
        "_hard_force_steps",
        "_overforce_steps",
        "_unlatched_lift_failure",
        "_potential_initialized",
    ):
        _require_zero(name, cache[name], reset_zero_atol)
    dof_to_joint_error = float((cache["dof_targets"] - joint_pos).abs().max())
    articulation_target_error = float((joint_pos_target - joint_pos).abs().max())
    if dof_to_joint_error > reset_zero_atol or articulation_target_error > reset_zero_atol:
        raise RuntimeError(
            "fresh reset requires dof_targets == articulation joint_pos_target == joint_pos; "
            f"errors={dof_to_joint_error:g}/{articulation_target_error:g}"
        )

    contact_before = _contact_reset_audit(
        task,
        layout,
        force_atol=contact_force_atol,
        timestamp_atol=timestamp_atol,
    )

    source = layout.source
    destination = layout.destination
    origins = getattr(scene, "env_origins", None)
    if not isinstance(origins, torch.Tensor) or origins.shape != (layout.num_envs, 3):
        raise TypeError("scene.env_origins must have shape [num_envs,3]")
    all_env_ids = torch.arange(layout.num_envs, dtype=torch.long, device=device)
    local_object_pos = root_link_pos - origins
    local_object_pos = local_object_pos.clone()
    local_object_pos.index_copy_(
        0, destination, local_object_pos.index_select(0, source)
    )
    paired_object_quat = root_link_quat.clone()
    paired_object_quat.index_copy_(
        0, destination, paired_object_quat.index_select(0, source)
    )
    object_pose = torch.cat(
        (local_object_pos + origins, paired_object_quat), dim=-1
    )
    object_velocity = torch.cat(
        (root_com_lin_vel, root_com_ang_vel), dim=-1
    ).clone()
    object_velocity.index_copy_(
        0, destination, object_velocity.index_select(0, source)
    )
    source_joint_pos = joint_pos.index_select(0, source).clone()
    source_joint_vel = joint_vel.index_select(0, source).clone()
    source_pos_target = joint_pos_target.index_select(0, source).clone()
    source_vel_target = joint_vel_target.index_select(0, source).clone()
    source_effort_target = joint_effort_target.index_select(0, source).clone()
    paired_joint_pos = joint_pos.clone()
    paired_joint_vel = joint_vel.clone()
    paired_pos_target = joint_pos_target.clone()
    paired_vel_target = joint_vel_target.clone()
    paired_effort_target = joint_effort_target.clone()
    for paired, snapshot in (
        (paired_joint_pos, source_joint_pos),
        (paired_joint_vel, source_joint_vel),
        (paired_pos_target, source_pos_target),
        (paired_vel_target, source_vel_target),
        (paired_effort_target, source_effort_target),
    ):
        paired.index_copy_(0, destination, snapshot)
    cache_snapshots = {
        name: value.index_select(0, source).clone() for name, value in cache.items()
    }

    # Rewrite both members, not only the destination.  This gives the two
    # PhysX actors the same number and ordering of reset writes before their
    # first contact solve; otherwise only the destination's table manifold is
    # invalidated by the post-reset teleport.
    robot.write_joint_state_to_sim(
        paired_joint_pos, paired_joint_vel, env_ids=all_env_ids
    )
    robot.set_joint_position_target(paired_pos_target, env_ids=all_env_ids)
    robot.set_joint_velocity_target(paired_vel_target, env_ids=all_env_ids)
    robot.set_joint_effort_target(paired_effort_target, env_ids=all_env_ids)
    object_asset.write_root_pose_to_sim(object_pose, env_ids=all_env_ids)
    object_asset.write_root_velocity_to_sim(object_velocity, env_ids=all_env_ids)
    for name, snapshot in cache_snapshots.items():
        cache[name].index_copy_(0, destination, snapshot)

    # Audited order: write buffered controls, kinematic forward, then rebuild
    # observations.  Advancing scene/sensor clocks here would no longer be a
    # zero-time reset-state diagnostic and is intentionally forbidden.
    scene.write_data_to_sim()
    sim.forward()
    observations = task._get_observations()
    if not isinstance(observations, Mapping) or not isinstance(
        observations.get("policy"), torch.Tensor
    ):
        raise TypeError("PickTool _get_observations() did not return a policy tensor")
    policy = observations["policy"]
    if policy.shape != (layout.num_envs, OBSERVATION_DIM) or policy.device != device:
        raise RuntimeError(
            f"rebuilt policy observation has invalid shape/device: {tuple(policy.shape)} {policy.device}"
        )
    task.obs_buf = observations

    contact_after = _contact_reset_audit(
        task,
        layout,
        force_atol=contact_force_atol,
        timestamp_atol=timestamp_atol,
    )
    local_pose = torch.cat(
        (
            object_data.root_link_pos_w - origins,
            object_data.root_link_quat_w,
        ),
        dim=-1,
    )
    com_velocity = torch.cat(
        (object_data.root_com_lin_vel_w, object_data.root_com_ang_vel_w), dim=-1
    )
    cache_errors = {name: paired_max_abs(value, layout) for name, value in cache.items()}
    physical_errors = {
        "joint_pos": paired_max_abs(robot_data.joint_pos, layout),
        "joint_vel": paired_max_abs(robot_data.joint_vel, layout),
        "joint_pos_target": paired_max_abs(robot_data.joint_pos_target, layout),
        "joint_vel_target": paired_max_abs(robot_data.joint_vel_target, layout),
        "joint_effort_target": paired_max_abs(robot_data.joint_effort_target, layout),
        "object_root_link_local_pose": paired_max_abs(local_pose, layout),
        "object_com_velocity": paired_max_abs(com_velocity, layout),
        "cache_max": max(cache_errors.values(), default=0.0),
    }
    post_copy_max = max(physical_errors.values(), default=0.0)
    if not math.isfinite(post_copy_max) or post_copy_max > reset_zero_atol:
        raise RuntimeError(
            "zero-time twin copy failed its physical/cache audit: "
            f"max_error={post_copy_max:g}, tolerance={reset_zero_atol:g}, "
            f"components={physical_errors}"
        )
    report = {
        "pair_count": layout.pair_count,
        "pairing": "adjacent_even_source_to_odd_destination_v1",
        "copied_cache_fields": list(TWIN_CACHE_FIELDS),
        "contact_before": contact_before,
        "contact_after": contact_after,
        "post_copy_pair_abs_error": physical_errors,
        "cache_pair_abs_error": cache_errors,
        "sync_sequence": [
            "scene.write_data_to_sim",
            "sim.forward",
            "task._get_observations",
        ],
        "contact_manifold_fabricated": False,
        "physical_write_scope": "all_twins_symmetrically_v1",
        "scene_update_called": False,
    }
    return policy, report


def publish_json_no_clobber(payload: Mapping[str, Any], output: Path) -> None:
    """Publish strict JSON atomically without replacing an existing path."""

    # ``Path.resolve`` follows a dangling destination symlink and would publish
    # through it.  Make the spelling absolute without dereferencing the final
    # path so the hard-link operation remains genuinely no-clobber.
    output = Path(os.path.abspath(os.fspath(output)))
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        dict(payload), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", dir=output.parent
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staging, output)
    finally:
        staging.unlink(missing_ok=True)


__all__ = [
    "ACTION_DIM",
    "CLOSE_QUALITY_INDEX",
    "FORCE_SLICE",
    "GRASP_LATCH_INDEX",
    "OBSERVATION_DIM",
    "PROXIMITY_SLICE",
    "PublicReadinessUpdate",
    "READINESS_HOLD_STEPS",
    "READINESS_STRATUM_NAMES",
    "TWIN_CACHE_FIELDS",
    "TwinPairLayout",
    "copy_fresh_reset_to_twins",
    "make_twin_pair_layout",
    "paired_max_abs",
    "publish_json_no_clobber",
    "replicate_source_actions",
    "update_public_readiness",
    "validate_full_task_contract",
]
