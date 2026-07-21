"""Pure-torch contract shared by fixed-pregrasp power-close search tools.

Search scores are deliberately only a ranking aid.  ``strict_power_close_pass`` remains the
authority: a four-frame power-grasp latch followed by thumb plus three legal opposed pads, rigid
hold, and a 15-frame stable window.  The search additionally imposes conservative trajectory-wide
force, horizontal-drift and launch limits before an oracle may be promoted to public-action replay.
"""

from __future__ import annotations

import torch


POWER_GRASP_QUALITY_MIN = 0.35
POWER_GRASP_QUALITY_LOW = 0.20
POWER_HOLD_QUALITY_MIN = 0.50
POWER_FORCE_LIMIT = 30.0
POWER_REQUIRED_OTHER_CONTACTS = 3
POWER_LATCH_CONFIRM_FRAMES = 4
POWER_LATCH_RELEASE_FRAMES = 6
POWER_STABLE_FRAMES = 15
POWER_XY_DRIFT_LIMIT = 0.03
POWER_CLEARANCE_LIMIT = 0.015
POWER_ROTATION_DRIFT_LIMIT = 0.35
POWER_MIN_PROXIMITY = 0.01
POWER_LOST_WINDOW_FRAMES = 12


def aggregate_replicated_candidates(
    env_score: torch.Tensor,
    strict_env_pass: torch.Tensor,
    finite: torch.Tensor,
    replicates: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate repeated physical clones without promoting a lucky singleton.

    Ranking is an equal blend of the replica mean and worst replica.  The lexicographic pass bonus
    is granted only when every replica independently satisfies the strict trajectory contract.
    """

    if env_score.ndim != 1 or strict_env_pass.shape != env_score.shape or finite.shape != env_score.shape:
        raise ValueError("replica score/pass/finite tensors must share one-dimensional shape")
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if env_score.numel() % replicates != 0:
        raise ValueError("environment count must be divisible by replicates")
    candidate_count = env_score.numel() // replicates
    score_matrix = env_score.reshape(candidate_count, replicates)
    finite_group = finite.reshape(candidate_count, replicates).all(dim=1)
    strict_group_pass = (
        strict_env_pass.reshape(candidate_count, replicates).all(dim=1) & finite_group
    )
    candidate_score = (
        0.5 * score_matrix.mean(dim=1)
        + 0.5 * score_matrix.amin(dim=1)
        + 50.0 * strict_group_pass.float()
    )
    candidate_score = torch.where(
        finite_group,
        candidate_score,
        torch.full_like(candidate_score, -1.0e9),
    )
    return candidate_score, strict_group_pass, finite_group


def update_power_grasp_latch(
    power_grasp_quality: torch.Tensor,
    is_grasped: torch.Tensor,
    confirm_count: torch.Tensor,
    release_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror the option's four-frame Schmitt latch for search rollouts.

    The returned latch is the state for the *next* done evaluation.  A caller must evaluate the
    current option-stability frame with the incoming ``is_grasped`` value before installing this
    result.  That one-step ordering is why the formal option's first possible success is action 19.
    """

    above_high = power_grasp_quality >= POWER_GRASP_QUALITY_MIN
    below_low = power_grasp_quality < POWER_GRASP_QUALITY_LOW
    next_confirm = torch.where(
        above_high, confirm_count + 1, torch.zeros_like(confirm_count)
    )
    next_release = torch.where(
        below_low, release_count + 1, torch.zeros_like(release_count)
    )
    confirmed = next_confirm >= POWER_LATCH_CONFIRM_FRAMES
    released = next_release >= POWER_LATCH_RELEASE_FRAMES
    next_is_grasped = (is_grasped | confirmed) & (~released)
    return next_is_grasped, next_confirm, next_release


def power_close_stable_frame(
    power_is_grasped: torch.Tensor,
    thumb_contact: torch.Tensor,
    legal_other_contacts: torch.Tensor,
    power_grasp_quality: torch.Tensor,
    hold_quality: torch.Tensor,
    max_force: torch.Tensor,
) -> torch.Tensor:
    """Return the exact per-frame strict power-close predicate."""

    return (
        power_is_grasped.bool()
        & thumb_contact.bool()
        & (legal_other_contacts >= POWER_REQUIRED_OTHER_CONTACTS)
        & (power_grasp_quality >= POWER_GRASP_QUALITY_MIN)
        & (hold_quality >= POWER_HOLD_QUALITY_MIN)
        & (max_force <= POWER_FORCE_LIMIT)
    )


def simultaneous_power_contact_stages(
    thumb_contact: torch.Tensor, legal_other_contacts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return simultaneous thumb-plus-three and thumb-plus-four contact stages."""

    if thumb_contact.shape != legal_other_contacts.shape:
        raise ValueError("thumb and legal-other contact tensors must share shape")
    thumb = thumb_contact.bool()
    return (
        thumb & (legal_other_contacts >= POWER_REQUIRED_OTHER_CONTACTS),
        thumb & (legal_other_contacts >= 4),
    )


def update_stable_streak(
    streak: torch.Tensor, peak: torch.Tensor, stable: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance a consecutive-frame counter without turning occupancy into a pass."""

    next_streak = torch.where(stable, streak + 1, torch.zeros_like(streak))
    return next_streak, torch.maximum(peak, next_streak)


def strict_power_close_pass(
    stable_streak_at_end: torch.Tensor,
    force_peak: torch.Tensor,
    xy_drift_peak: torch.Tensor,
    rotation_drift_peak: torch.Tensor,
    clearance_peak: torch.Tensor,
    unexpected_done: torch.Tensor,
) -> torch.Tensor:
    """Require a handoff-ready terminal hold plus conservative trajectory-wide safety."""

    return (
        (stable_streak_at_end >= POWER_STABLE_FRAMES)
        & (force_peak <= POWER_FORCE_LIMIT)
        & (xy_drift_peak <= POWER_XY_DRIFT_LIMIT)
        & (rotation_drift_peak <= POWER_ROTATION_DRIFT_LIMIT)
        & (clearance_peak <= POWER_CLEARANCE_LIMIT)
        & (~unexpected_done.bool())
    )


def conservative_coupled_teacher_pass(
    *,
    native_success: torch.Tensor,
    native_failure: torch.Tensor,
    native_timeout: torch.Tensor,
    terminal_stable_steps: torch.Tensor,
    terminal_power_is_grasped: torch.Tensor,
    terminal_thumb_contact: torch.Tensor,
    terminal_legal_other_contacts: torch.Tensor,
    terminal_power_grasp_quality: torch.Tensor,
    terminal_hold_quality: torch.Tensor,
    terminal_max_force: torch.Tensor,
    terminal_align_active: torch.Tensor,
    trajectory_force_peak: torch.Tensor,
    trajectory_xy_drift_peak: torch.Tensor,
    trajectory_rotation_drift_peak: torch.Tensor,
    trajectory_clearance_peak: torch.Tensor,
    arm_target_saturated: torch.Tensor,
) -> torch.Tensor:
    """Audit native option success before a rollout may supervise the coupled actor.

    Native success remains the task-state authority.  This stricter promotion gate additionally
    rejects trajectories that ever relied on transient >30 N contact or the arm-target envelope;
    those behaviors may be recoverable online, but are unsuitable behavioral-cloning labels.
    """

    return (
        native_success.bool()
        & (~native_failure.bool())
        & (~native_timeout.bool())
        & (terminal_stable_steps >= POWER_STABLE_FRAMES)
        & terminal_power_is_grasped.bool()
        & terminal_thumb_contact.bool()
        & (terminal_legal_other_contacts >= POWER_REQUIRED_OTHER_CONTACTS)
        & (terminal_power_grasp_quality >= POWER_GRASP_QUALITY_MIN)
        & (terminal_hold_quality >= POWER_HOLD_QUALITY_MIN)
        & (terminal_max_force <= POWER_FORCE_LIMIT)
        & (~terminal_align_active.bool())
        & (trajectory_force_peak <= POWER_FORCE_LIMIT)
        & (trajectory_xy_drift_peak <= POWER_XY_DRIFT_LIMIT)
        & (trajectory_rotation_drift_peak <= POWER_ROTATION_DRIFT_LIMIT)
        & (trajectory_clearance_peak <= POWER_CLEARANCE_LIMIT)
        & (~arm_target_saturated.bool())
    )


def power_close_candidate_score(
    *,
    power_close_mean: torch.Tensor,
    thumb_fraction: torch.Tensor,
    thumb_and_third_fraction: torch.Tensor,
    thumb_and_fourth_fraction: torch.Tensor,
    power_wrap_mean: torch.Tensor,
    power_grasp_mean: torch.Tensor,
    hold_mean: torch.Tensor,
    stable_fraction: torch.Tensor,
    stable_streak_at_end: torch.Tensor,
    stable_streak_peak: torch.Tensor,
    force_peak: torch.Tensor,
    xy_drift_peak: torch.Tensor,
    rotation_drift_peak: torch.Tensor,
    clearance_peak: torch.Tensor,
    strict_pass: torch.Tensor,
    unexpected_done: torch.Tensor,
) -> torch.Tensor:
    """Dense CEM ranking with simultaneous thumb-plus-third/fourth-pad credit.

    The large strict-pass offset makes an already feasible sample lexicographically preferable,
    but cannot manufacture feasibility: the saved artifact records the independent boolean verdict.
    Hold is credited only after the third pad arrives, because a static object on the table has
    near-perfect hold quality even when the hand is not grasping it.
    """

    end_streak_progress = torch.clamp(
        stable_streak_at_end.float() / POWER_STABLE_FRAMES, 0.0, 1.0
    )
    peak_streak_progress = torch.clamp(
        stable_streak_peak.float() / POWER_STABLE_FRAMES, 0.0, 1.0
    )
    overforce = torch.relu(force_peak - POWER_FORCE_LIMIT) / POWER_FORCE_LIMIT
    horizontal_escape = torch.relu(xy_drift_peak - POWER_XY_DRIFT_LIMIT) / POWER_XY_DRIFT_LIMIT
    rotation_escape = (
        torch.relu(rotation_drift_peak - POWER_ROTATION_DRIFT_LIMIT)
        / POWER_ROTATION_DRIFT_LIMIT
    )
    launch = torch.relu(clearance_peak - POWER_CLEARANCE_LIMIT) / POWER_CLEARANCE_LIMIT
    unsafe_indicator = (
        (force_peak > POWER_FORCE_LIMIT)
        | (xy_drift_peak > POWER_XY_DRIFT_LIMIT)
        | (rotation_drift_peak > POWER_ROTATION_DRIFT_LIMIT)
        | (clearance_peak > POWER_CLEARANCE_LIMIT)
        | unexpected_done.bool()
    ).float()

    return (
        2.0 * power_close_mean
        + 1.0 * thumb_fraction
        + 3.0 * thumb_and_third_fraction
        + 1.5 * thumb_and_fourth_fraction
        + 2.0 * power_wrap_mean
        + 3.0 * power_grasp_mean
        + 1.0 * hold_mean * thumb_and_third_fraction
        + 2.0 * stable_fraction
        + 6.0 * end_streak_progress
        + 2.0 * peak_streak_progress
        + 50.0 * strict_pass.float()
        - 8.0 * overforce
        - 8.0 * horizontal_escape
        - 4.0 * rotation_escape
        - 8.0 * launch
        - 10.0 * unsafe_indicator
    )
