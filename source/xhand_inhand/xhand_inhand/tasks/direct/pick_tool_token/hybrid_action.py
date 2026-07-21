# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-torch helpers for the token-plus-distal-residual hand action."""

from __future__ import annotations

import torch


def zero_action_prefix(actions: torch.Tensor, prefix_width: int) -> torch.Tensor:
    """Return actions with a leading control group replaced by exact zeros."""

    if actions.ndim != 2:
        raise ValueError("actions must have shape (N, A)")
    if not 0 <= prefix_width <= actions.shape[1]:
        raise ValueError("prefix_width must be within the action dimension")
    constrained = actions.clone()
    constrained[:, :prefix_width] = 0.0
    return constrained


def hold_arm_reset_state(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    dof_targets: torch.Tensor,
    last_action: torch.Tensor,
    arm_joint_indices: torch.Tensor,
    arm_action_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove captured arm momentum, target preload and action history at an option reset.

    The hand portions remain physically captured.  This makes a close-only option begin from the
    recorded pregrasp geometry while the arm holds that pose instead of continuing a parent
    policy's reach/lift command.
    """

    if joint_pos.ndim != 2 or joint_vel.shape != joint_pos.shape or dof_targets.shape != joint_pos.shape:
        raise ValueError("joint_pos, joint_vel and dof_targets must share shape (N, J)")
    if last_action.ndim != 2 or last_action.shape[0] != joint_pos.shape[0]:
        raise ValueError("last_action must have shape (N, A)")
    if arm_joint_indices.ndim != 1:
        raise ValueError("arm_joint_indices must be one-dimensional")
    if arm_joint_indices.numel() != arm_action_width:
        raise ValueError("arm action width must equal the number of arm joints")
    if not 0 <= arm_action_width <= last_action.shape[1]:
        raise ValueError("arm_action_width must be within the action dimension")
    if arm_joint_indices.numel() and (
        int(arm_joint_indices.min()) < 0 or int(arm_joint_indices.max()) >= joint_pos.shape[1]
    ):
        raise ValueError("arm_joint_indices contains an out-of-range joint")
    if arm_joint_indices.unique().numel() != arm_joint_indices.numel():
        raise ValueError("arm_joint_indices must not contain duplicates")

    held_velocity = joint_vel.clone()
    held_targets = dof_targets.clone()
    held_action = zero_action_prefix(last_action, arm_action_width)
    held_velocity.index_fill_(1, arm_joint_indices, 0.0)
    held_targets.index_copy_(1, arm_joint_indices, joint_pos.index_select(1, arm_joint_indices))
    return held_velocity, held_targets, held_action


def update_arm_hold_release(
    stable_grasp: torch.Tensor,
    stable_steps: torch.Tensor,
    released: torch.Tensor,
    *,
    confirm_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance a one-way stable-grasp arm-hold supervisor.

    Only environments whose arm is still held accumulate consecutive stable-grasp frames.  Once
    the confirmation window is reached, release is permanent and the counter is frozen at its
    release value.  Freezing makes the terminal payload an auditable record even if the grasp is
    later lost during lift.
    """

    if stable_grasp.ndim != 1:
        raise ValueError("stable_grasp must be one-dimensional")
    if stable_steps.shape != stable_grasp.shape or stable_steps.dtype != torch.long:
        raise ValueError("stable_steps must be a long vector matching stable_grasp")
    if released.shape != stable_grasp.shape or released.dtype != torch.bool:
        raise ValueError("released must be a bool vector matching stable_grasp")
    if stable_grasp.dtype != torch.bool:
        raise ValueError("stable_grasp must be boolean")
    if stable_steps.device != stable_grasp.device or released.device != stable_grasp.device:
        raise ValueError("arm-hold state tensors must share a device")
    if confirm_steps < 1:
        raise ValueError("confirm_steps must be positive")

    pending = ~released
    candidate_steps = torch.where(
        stable_grasp,
        stable_steps + 1,
        torch.zeros_like(stable_steps),
    )
    next_steps = torch.where(pending, candidate_steps, stable_steps)
    next_released = released | (pending & (next_steps >= confirm_steps))
    return next_steps, next_released


def apply_asymmetric_joint_residual(
    base_target: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    residual: torch.Tensor,
    joint_indices: torch.Tensor,
    *,
    validate_indices: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add normalized residuals without accumulating them across control steps.

    For every selected joint, residual values ``-1, 0, +1`` map exactly to the runtime lower
    limit, the token-decoded target, and the runtime upper limit.  The asymmetric mapping retains
    the full feasible set even when the token target is not centered in the joint range.
    """

    if base_target.ndim != 2 or lower.shape != base_target.shape or upper.shape != base_target.shape:
        raise ValueError("base_target, lower and upper must have the same (N, J) shape")
    if residual.ndim != 2 or residual.shape[0] != base_target.shape[0]:
        raise ValueError("residual must have shape (N, R)")
    if joint_indices.ndim != 1 or residual.shape[1] != joint_indices.numel():
        raise ValueError("residual width must equal the number of selected joints")
    if validate_indices:
        # Runtime control passes prevalidated static indices with this disabled, avoiding a GPU/CPU
        # synchronization on every physics step. Standalone callers retain defensive validation.
        if joint_indices.numel() and (
            int(joint_indices.min()) < 0 or int(joint_indices.max()) >= base_target.shape[1]
        ):
            raise ValueError("joint_indices contains an out-of-range joint")
        if joint_indices.unique().numel() != joint_indices.numel():
            raise ValueError("joint_indices must not contain duplicates")

    base = torch.maximum(torch.minimum(base_target, upper), lower)
    selected_base = base.index_select(1, joint_indices)
    selected_lower = lower.index_select(1, joint_indices)
    selected_upper = upper.index_select(1, joint_indices)
    residual = residual.clamp(-1.0, 1.0)
    available_span = torch.where(
        residual >= 0.0,
        selected_upper - selected_base,
        selected_base - selected_lower,
    )
    delta = residual * available_span
    target = base.clone()
    target.index_copy_(1, joint_indices, selected_base + delta)
    return target, delta


def invert_asymmetric_joint_residual(
    target: torch.Tensor,
    base_target: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    joint_indices: torch.Tensor,
) -> torch.Tensor:
    """Encode selected absolute joint targets back into normalized residual actions."""

    if target.shape != base_target.shape or lower.shape != base_target.shape or upper.shape != base_target.shape:
        raise ValueError("target, base_target, lower and upper must have the same (N, J) shape")
    selected_target = target.index_select(1, joint_indices)
    selected_base = torch.maximum(
        torch.minimum(base_target.index_select(1, joint_indices), upper.index_select(1, joint_indices)),
        lower.index_select(1, joint_indices),
    )
    delta = selected_target - selected_base
    span = torch.where(
        delta >= 0.0,
        upper.index_select(1, joint_indices) - selected_base,
        selected_base - lower.index_select(1, joint_indices),
    ).clamp_min(1.0e-8)
    return (delta / span).clamp(-1.0, 1.0)
