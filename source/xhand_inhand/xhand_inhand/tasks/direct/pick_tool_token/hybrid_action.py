# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-torch helpers for the token-plus-distal-residual hand action."""

from __future__ import annotations

import torch


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
