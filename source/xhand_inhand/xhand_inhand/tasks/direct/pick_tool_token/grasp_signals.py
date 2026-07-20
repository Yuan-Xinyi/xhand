# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-torch grasp signals shared by the environment and regression tests.

Keeping these calculations independent of Isaac Sim makes the two most important invariants
cheap to test: rigid palm/object motion must have zero slip, and every grasp phase transition
must use the same Schmitt-trigger quality signal.
"""

from __future__ import annotations

import torch


def rigid_hold_quality(
    palm_com_pos_w: torch.Tensor,
    palm_com_lin_vel_w: torch.Tensor,
    palm_com_ang_vel_w: torch.Tensor,
    object_com_pos_w: torch.Tensor,
    object_com_lin_vel_w: torch.Tensor,
    object_com_ang_vel_w: torch.Tensor,
    linear_scale: float,
    angular_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return rigid-body hold quality and relative linear/angular slip.

    All positions and velocities deliberately use COM frames. Mixing link-frame positions with
    COM velocities adds a fictitious ``omega x offset`` term whenever the hand rotates.
    """

    palm_to_object = object_com_pos_w - palm_com_pos_w
    expected_object_lin_vel = palm_com_lin_vel_w + torch.cross(
        palm_com_ang_vel_w, palm_to_object, dim=-1
    )
    slip_lin = (object_com_lin_vel_w - expected_object_lin_vel).norm(dim=-1)
    slip_ang = (object_com_ang_vel_w - palm_com_ang_vel_w).norm(dim=-1)
    quality = torch.exp(-slip_lin / linear_scale) * torch.exp(-slip_ang / angular_scale)
    return quality, slip_lin, slip_ang


def wrap_quality(
    force_magnitude: torch.Tensor,
    handle_surface_distance: torch.Tensor,
    handle_surface_normal_w: torch.Tensor,
    finger_alignment: torch.Tensor,
    palm_facing: torch.Tensor,
    thumb_index: int,
    other_indices: torch.Tensor,
    *,
    force_threshold: float,
    force_saturation: float,
    surface_margin: float,
    palm_facing_min: float,
    alignment_min: float,
    opposition_min: float,
) -> dict[str, torch.Tensor]:
    """Compute a bounded power-grasp quality from contact topology and geometry.

    The second-strongest non-thumb contact is used, so a thumb plus one grazing finger cannot
    form a grasp.  The minimum of force coverage, palm orientation, pad alignment and contact-side
    opposition keeps every component necessary: a large value cannot hide a missing component.
    """

    near_handle = handle_surface_distance < surface_margin
    force_present = force_magnitude > force_threshold
    contact = near_handle & force_present
    contact_strength = torch.tanh(force_magnitude / force_saturation) * contact.float()

    other_strength = contact_strength[:, other_indices]
    thumb_strength = contact_strength[:, thumb_index]
    thumb_align = finger_alignment[:, thumb_index]
    thumb_normal = handle_surface_normal_w[:, thumb_index].unsqueeze(1)
    other_normals = handle_surface_normal_w[:, other_indices]
    opposition_each = 0.5 * (1.0 - (thumb_normal * other_normals).sum(dim=-1))
    opposition_each_score = torch.clamp(
        (opposition_each - opposition_min) / max(1.0 - opposition_min, 1.0e-6), 0.0, 1.0
    )
    other_align = finger_alignment[:, other_indices]
    other_align_score = torch.clamp(
        (other_align - alignment_min) / max(1.0 - alignment_min, 1.0e-6), 0.0, 1.0
    )
    # Rank non-thumb fingers by whether each one is simultaneously in contact, pad-aligned and
    # opposite the thumb. A strong wrong-side collision must not hide two weaker legal contacts.
    other_candidate = other_strength * other_align_score * opposition_each_score
    _, selected_local_idx = torch.topk(other_candidate, k=2, dim=1)
    selected_finger_idx = other_indices[selected_local_idx]
    # The product above is only a ranking key.  Coverage itself is the weaker selected contact
    # strength; alignment and opposition are separate minimum gates below.  Reusing the product as
    # coverage applies both geometric factors twice and makes a perfectly valid moderate grasp fail.
    selected_strength = torch.gather(other_strength, 1, selected_local_idx)
    other_coverage = selected_strength.min(dim=1).values

    selected_align = torch.gather(finger_alignment, 1, selected_finger_idx)
    alignment_raw = torch.minimum(thumb_align, selected_align.min(dim=1).values)
    alignment_score = torch.clamp(
        (alignment_raw - alignment_min) / max(1.0 - alignment_min, 1.0e-6), 0.0, 1.0
    )
    selected_opposition = torch.gather(opposition_each, 1, selected_local_idx)
    opposition_raw = selected_opposition.min(dim=1).values
    opposition_score = torch.clamp(
        (opposition_raw - opposition_min) / max(1.0 - opposition_min, 1.0e-6), 0.0, 1.0
    )

    palm_score = torch.clamp(
        (palm_facing - palm_facing_min) / max(1.0 - palm_facing_min, 1.0e-6), 0.0, 1.0
    )

    quality = torch.stack(
        (thumb_strength, other_coverage, alignment_score, opposition_score, palm_score), dim=1
    ).min(dim=1).values

    thumb_contact = contact[:, thumb_index]
    other_count = contact[:, other_indices].sum(dim=1)
    return {
        "quality": quality,
        "contact": contact,
        "contact_strength": contact_strength,
        "thumb_strength": thumb_strength,
        "other_coverage": other_coverage,
        "thumb_contact": thumb_contact,
        "other_contact_count": other_count,
        "palm_score": palm_score,
        "alignment_score": alignment_score,
        "alignment_raw": alignment_raw,
        "opposition_score": opposition_score,
        "opposition_raw": opposition_raw,
    }


def update_grasp_latch(
    quality: torch.Tensor,
    is_grasped: torch.Tensor,
    confirm_count: torch.Tensor,
    release_count: torch.Tensor,
    *,
    high_threshold: float,
    low_threshold: float,
    confirm_steps: int,
    release_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a Schmitt trigger to one shared grasp-quality signal.

    Quality in the dead band preserves an existing latch but cannot confirm a new one.  Importantly,
    poor geometry or high slip increments the release counter even if two raw contacts remain.
    """

    above_high = quality >= high_threshold
    below_low = quality < low_threshold
    confirm_count = torch.where(above_high, confirm_count + 1, torch.zeros_like(confirm_count))
    release_count = torch.where(below_low, release_count + 1, torch.zeros_like(release_count))
    confirmed = confirm_count >= confirm_steps
    released = release_count >= release_steps
    next_is_grasped = (is_grasped | confirmed) & ~released
    newly_confirmed = next_is_grasped & ~is_grasped
    return next_is_grasped, confirm_count, release_count, newly_confirmed, released
