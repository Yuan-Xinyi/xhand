# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared, simulator-free helpers for the pick-tool option pipeline.

These utilities were previously copy-pasted across the dataset collectors, the scripted
oracle and the strict evaluator, and had already drifted apart (the curriculum boundary
schema is the documented example).  Keeping them here gives collection, evaluation and the
oracle a single source of truth: editing the pregrasp-score formula, the boundary schema or
the force-safety constants in one place can no longer silently desynchronize the others.

The module imports only ``torch`` and the standard library; every function takes the already
constructed environment ``u`` (an unwrapped ``PickToolTokenEnv``) as an argument, so importing
it never pulls in Isaac Sim.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


# Curriculum boundary schema.  The env's ``_load_curriculum_boundary`` requires exactly these
# keys; the three latch vectors are Markov-critical (restoring a latched lift boundary without
# ``is_grasped=True`` makes the grasp shield block upward arm motion).
BOUNDARY_POSE_KEYS = (
    "joint_pos",
    "joint_vel",
    "dof_targets",
    "object_local_pos",
    "object_quat",
    "object_velocity",
    "last_action",
)
BOUNDARY_LATCH_KEYS = ("contact_steps", "lost_contact_steps", "is_grasped")

# Documented pregrasp handoff operating point (see docs/pick_tool_token_journey.md section 9).
# Collectors and the evaluator should default to this so "close contract entry" measures the
# same event everywhere.
PREGRASP_GATE = {
    "score": 0.30,
    "hold_steps": 4,
    "min_step": 400,
    "min_proximity": 0.02,
}


def sha256(path: Path | str) -> str:
    """Stream a file's SHA-256 so large checkpoints/datasets are not read into memory at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def limit_norm(value: torch.Tensor, limit: float) -> torch.Tensor:
    """Scale a batched vector down to at most ``limit`` in L2 norm, leaving direction intact."""

    norm = value.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
    return value * torch.clamp(limit / norm, max=1.0)


def pregrasp_score(u) -> torch.Tensor:
    """Geometry-only readiness score used by the reach-to-grasp option transition.

    Combines thumb+two-nearest-opposing-finger proximity, finger alignment and palm facing, and
    is gated to zero unless the tool's true mesh minimum is still resting on the table.
    """

    distances = u._curr_fingertip_distances
    other_distances = distances[:, u._other_ee_idx]
    nearest_distances, nearest_indices = torch.topk(other_distances, k=2, dim=1, largest=False)
    grasp_distance = (distances[:, u._thumb_ee_idx] + nearest_distances.sum(dim=-1)) / 3.0

    other_alignment = u._finger_align[:, u._other_ee_idx]
    alignment = (
        u._finger_align[:, u._thumb_ee_idx]
        + torch.gather(other_alignment, 1, nearest_indices).sum(dim=-1)
    ) / 3.0
    to_handle = u.handle_center_w - u.palm_center_w
    to_handle = to_handle / to_handle.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    palm_facing = 0.5 * (1.0 + (u.palm_normal_w * to_handle).sum(dim=-1))
    clearance = u._object_true_min_z() - u._table_surface_z
    score = torch.exp(-grasp_distance / 0.025) * alignment * palm_facing
    return torch.where(clearance.abs() <= 0.005, score, torch.zeros_like(score))


def capture_boundary(u) -> dict[str, torch.Tensor]:
    """Snapshot the full curriculum boundary schema (pose fields + Markov latch vectors)."""

    return {
        "joint_pos": u.robot.data.joint_pos.detach().clone(),
        "joint_vel": u.robot.data.joint_vel.detach().clone(),
        "dof_targets": u.dof_targets.detach().clone(),
        "object_local_pos": (u.object.data.root_pos_w - u.scene.env_origins).detach().clone(),
        "object_quat": u.object.data.root_quat_w.detach().clone(),
        "object_velocity": torch.cat(
            (u.object.data.root_com_lin_vel_w, u.object.data.root_com_ang_vel_w), dim=-1
        ).detach().clone(),
        "last_action": u.actions.detach().clone(),
        "contact_steps": u._contact_steps.detach().clone(),
        "lost_contact_steps": u._lost_contact_steps.detach().clone(),
        "is_grasped": u._is_grasped.detach().clone(),
    }


def sustained_force_unsafe(
    force_max: torch.Tensor,
    hard_force_steps: torch.Tensor,
    overforce_steps: torch.Tensor,
    cfg,
) -> torch.Tensor:
    """Advance the env's two sustained-force counters in place and return the unsafe mask.

    Mirrors ``PickToolTokenEnv``'s cutoff exactly (30 N for ``tactile_hard_terminate_steps`` frames
    or 60 N for ``tactile_terminate_steps`` frames), so collectors that disable the terminations
    still reject trajectories the deployment/eval env would kill.  ``hard_force_steps`` and
    ``overforce_steps`` are updated in place.
    """

    hard_force_steps.copy_(
        torch.where(
            force_max > cfg.tactile_hard_force_limit,
            hard_force_steps + 1,
            torch.zeros_like(hard_force_steps),
        )
    )
    overforce_steps.copy_(
        torch.where(
            force_max > cfg.tactile_terminate_force_limit,
            overforce_steps + 1,
            torch.zeros_like(overforce_steps),
        )
    )
    return (hard_force_steps >= cfg.tactile_hard_terminate_steps) | (
        overforce_steps >= cfg.tactile_terminate_steps
    )
