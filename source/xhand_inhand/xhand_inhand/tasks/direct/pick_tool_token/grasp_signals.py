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


def staged_close_quality(
    force_magnitude: torch.Tensor,
    handle_surface_distance: torch.Tensor,
    handle_contact_region: torch.Tensor,
    handle_surface_normal_w: torch.Tensor,
    finger_alignment: torch.Tensor,
    palm_score: torch.Tensor,
    thumb_index: int,
    other_indices: torch.Tensor,
    *,
    alignment_min: float,
    opposition_min: float,
    proximity_scale_far: float,
    proximity_scale_near: float,
    force_saturation: float,
) -> dict[str, torch.Tensor]:
    """Return a staged near -> thumb -> thumb+one -> thumb+two closure potential.

    Unlike strict :func:`wrap_quality`, this signal must remain informative before force closure.
    Its four additive tiers are 0.15/0.20/0.25/0.40, so reaching the handle, touching with the
    thumb, adding one opposed pad and finally adding a second opposed pad each create a distinct
    improvement.  It is a potential only: latch, lift and success still use strict wrap quality.
    """

    alignment_score = torch.clamp(
        (finger_alignment - alignment_min) / max(1.0 - alignment_min, 1.0e-6), 0.0, 1.0
    )
    thumb_normal = handle_surface_normal_w[:, thumb_index].unsqueeze(1)
    other_normals = handle_surface_normal_w[:, other_indices]
    opposition_raw = 0.5 * (1.0 - (thumb_normal * other_normals).sum(dim=-1))
    opposition_score = torch.clamp(
        (opposition_raw - opposition_min) / max(1.0 - opposition_min, 1.0e-6), 0.0, 1.0
    )

    proximity = 0.5 * torch.exp(-handle_surface_distance / proximity_scale_far)
    proximity = proximity + 0.5 * torch.exp(-handle_surface_distance / proximity_scale_near)
    proximity = proximity * handle_contact_region.float()
    force_strength = torch.tanh(force_magnitude / force_saturation)

    legal_near = proximity * alignment_score
    legal_other_near = legal_near[:, other_indices] * opposition_score
    legal_finger_proximity = legal_near.clone()
    legal_finger_proximity.index_copy_(1, other_indices, legal_other_near)
    best_other_near = torch.topk(legal_other_near, k=2, dim=1).values
    thumb_near = legal_near[:, thumb_index]
    proximity_quality = (thumb_near + best_other_near.sum(dim=-1)) / 3.0

    thumb_contact = thumb_near * force_strength[:, thumb_index]
    other_contact = legal_other_near * force_strength[:, other_indices]
    best_other_contact = torch.topk(other_contact, k=2, dim=1).values
    one_opposed = torch.minimum(thumb_contact, best_other_contact[:, 0])
    two_opposed = torch.minimum(one_opposed, best_other_contact[:, 1])
    quality = palm_score * (
        0.15 * proximity_quality
        + 0.20 * thumb_contact
        + 0.25 * one_opposed
        + 0.40 * two_opposed
    )
    contact_quality = palm_score * (
        0.20 * thumb_contact + 0.25 * one_opposed + 0.40 * two_opposed
    )
    return {
        "close_quality": quality,
        "contact_quality": contact_quality,
        "proximity_quality": proximity_quality,
        "finger_proximity": proximity,
        "legal_finger_proximity": legal_finger_proximity,
        "finger_force_strength": force_strength,
        "finger_alignment_score": alignment_score,
        "finger_opposition_score": opposition_score,
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


def update_close_option_state(
    grasp_quality: torch.Tensor,
    hold_quality: torch.Tensor,
    max_force: torch.Tensor,
    is_grasped: torch.Tensor,
    stable_count: torch.Tensor,
    clearance: torch.Tensor,
    horizontal_drift: torch.Tensor,
    proximity_quality: torch.Tensor,
    lost_window_count: torch.Tensor,
    unsafe_force: torch.Tensor,
    *,
    grasp_quality_threshold: float,
    hold_quality_threshold: float,
    safe_force_limit: float,
    confirm_steps: int,
    unlatched_lift_limit: float,
    horizontal_drift_limit: float,
    min_proximity: float,
    lost_window_steps: int,
) -> dict[str, torch.Tensor]:
    """Update the terminal state for the isolated close option.

    This state machine is intentionally stricter than the four-frame phase latch.  A close policy
    only succeeds after continuously carrying a safe, low-slip latch for ``confirm_steps``.  Before
    that point it may neither lift the object nor shove it across the table.  Losing the pregrasp
    window for several frames also ends the short-horizon option instead of letting it farm a
    state-only potential until timeout.
    """

    stable_now = (
        is_grasped
        & (grasp_quality >= grasp_quality_threshold)
        & (hold_quality >= hold_quality_threshold)
        & (max_force <= safe_force_limit)
    )
    next_stable_count = torch.where(
        stable_now, stable_count + 1, torch.zeros_like(stable_count)
    )
    success = next_stable_count >= confirm_steps

    outside_window = (~is_grasped) & (proximity_quality < min_proximity)
    next_lost_window_count = torch.where(
        outside_window, lost_window_count + 1, torch.zeros_like(lost_window_count)
    )
    lost_window = next_lost_window_count >= lost_window_steps
    # Once the ordinary Schmitt latch has confirmed, rigid palm/object motion is legitimate.  The
    # independent 15-frame streak below verifies that it remains a safe, low-slip hold.
    unlatched_lift = (~is_grasped) & (clearance > unlatched_lift_limit)
    horizontal_escape = (~is_grasped) & (horizontal_drift > horizontal_drift_limit)
    failure = (unsafe_force | unlatched_lift | horizontal_escape | lost_window) & ~success

    return {
        "stable_now": stable_now,
        "stable_count": next_stable_count,
        "success": success,
        "lost_window_count": next_lost_window_count,
        "lost_window": lost_window,
        "unlatched_lift": unlatched_lift,
        "horizontal_escape": horizontal_escape,
        "failure": failure,
    }
