"""Public, gate-only safety state for PickTool option selection.

The full-task PPO observation remains the frozen 115-D contract.  A policy
that decides whether to leave SEARCH nevertheless needs the progress of the
two consecutive-force termination counters to make that *gate* decision
Markov.  This module publishes only those normalized counters; it deliberately
does not expose true clearance, raw contact force, or evaluator-only geometry.
Progress is normalized by the active environment termination limits; the
deployment trial separately requires the authored full-task values 10 and 2.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


PUBLIC_GATE_STATE_EXTRAS_KEY = "pick_tool_public_gate_state_v1"
PUBLIC_GATE_STATE_CONTRACT = "pick_tool_public_gate_state_v1"
PUBLIC_GATE_STATE_VERSION = 1
PUBLIC_GATE_FEATURE_NAMES = (
    "hard_force_count_progress",
    "overforce_count_progress",
)


def _positive_step_limit(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_public_gate_state(
    hard_force_steps: torch.Tensor,
    overforce_steps: torch.Tensor,
    *,
    hard_terminate_steps: int,
    overforce_terminate_steps: int,
) -> dict[str, torch.Tensor]:
    """Return detached float32 progress without mutating counter tensors."""

    if not isinstance(hard_force_steps, torch.Tensor) or not isinstance(
        overforce_steps, torch.Tensor
    ):
        raise TypeError("force counters must be torch.Tensor instances")
    if hard_force_steps.ndim != 1 or overforce_steps.shape != hard_force_steps.shape:
        raise ValueError("force counters must be equally shaped one-dimensional tensors")
    if hard_force_steps.device != overforce_steps.device:
        raise ValueError("force counters must share a device")
    for name, value in (
        ("hard_force_steps", hard_force_steps),
        ("overforce_steps", overforce_steps),
    ):
        if value.dtype == torch.bool or value.dtype.is_floating_point:
            raise TypeError(f"{name} must use an integer dtype")
        # These are environment-owned monotone/reset counters.  Do not place a
        # CUDA reduction/host synchronization in the per-step publication path.

    hard_limit = _positive_step_limit("hard_terminate_steps", hard_terminate_steps)
    overforce_limit = _positive_step_limit(
        "overforce_terminate_steps", overforce_terminate_steps
    )
    return {
        "hard_force_count_progress": torch.clamp(
            hard_force_steps.detach().to(dtype=torch.float32) / float(hard_limit),
            0.0,
            1.0,
        ),
        "overforce_count_progress": torch.clamp(
            overforce_steps.detach().to(dtype=torch.float32) / float(overforce_limit),
            0.0,
            1.0,
        ),
    }


def public_gate_feature_tensor(
    state: Mapping[str, torch.Tensor],
    *,
    num_envs: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Check metadata and stack state without synchronizing a CUDA rollout."""

    if not isinstance(state, Mapping):
        raise TypeError("public gate state must be a mapping")
    if set(state) != set(PUBLIC_GATE_FEATURE_NAMES):
        missing = sorted(set(PUBLIC_GATE_FEATURE_NAMES) - set(state))
        extra = sorted(set(state) - set(PUBLIC_GATE_FEATURE_NAMES))
        raise KeyError(f"public gate state schema mismatch: missing={missing}, extra={extra}")

    expected_device = torch.device(device) if device is not None else None
    columns: list[torch.Tensor] = []
    inferred_rows: int | None = None
    for name in PUBLIC_GATE_FEATURE_NAMES:
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise ValueError(f"public gate state {name} must be a one-dimensional tensor")
        if value.dtype != torch.float32:
            raise TypeError(f"public gate state {name} must use torch.float32")
        if expected_device is not None and value.device != expected_device:
            raise ValueError(
                f"public gate state {name} is on {value.device}, expected {expected_device}"
            )
        if inferred_rows is None:
            inferred_rows = int(value.numel())
        elif value.numel() != inferred_rows:
            raise ValueError("public gate state feature lengths disagree")
        columns.append(value)

    assert inferred_rows is not None
    if num_envs is not None and inferred_rows != num_envs:
        raise ValueError(
            f"public gate state has {inferred_rows} rows, expected {num_envs}"
        )
    return torch.stack(columns, dim=-1)


def validate_public_gate_feature_values(features: torch.Tensor) -> None:
    """Synchronizing value audit for setup, artifacts, tests, and diagnostics only."""

    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError("public gate features must be a two-dimensional tensor")
    if features.shape[1] != len(PUBLIC_GATE_FEATURE_NAMES):
        raise ValueError(
            f"public gate features must have {len(PUBLIC_GATE_FEATURE_NAMES)} columns"
        )
    if features.dtype != torch.float32:
        raise TypeError("public gate features must use torch.float32")
    if not bool(torch.isfinite(features).all()) or bool(
        ((features < 0.0) | (features > 1.0)).any()
    ):
        raise ValueError("public gate features must be finite and in [0, 1]")


__all__ = [
    "PUBLIC_GATE_FEATURE_NAMES",
    "PUBLIC_GATE_STATE_CONTRACT",
    "PUBLIC_GATE_STATE_EXTRAS_KEY",
    "PUBLIC_GATE_STATE_VERSION",
    "build_public_gate_state",
    "public_gate_feature_tensor",
    "validate_public_gate_feature_values",
]
