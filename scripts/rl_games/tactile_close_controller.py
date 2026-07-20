#!/usr/bin/env python3
"""Stateful, bounded distal close servo for frozen pick-tool actors."""

from __future__ import annotations

import torch


class TactileCloseController:
    """Close low-force fingertips while preserving the frozen actor outside a hard gate.

    Action layout is arm7 | token9 | distal5.  The controller enters only near the handle and
    before grasp latch.  Its default form changes distal5 exclusively; optional arm-zero and
    token-hold switches are separate structural ablations.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        *,
        entry_proximity: float = 0.02,
        exit_proximity: float = 0.01,
        low_force: float = 3.0,
        high_force: float = 20.0,
        step: float = 0.005,
        max_travel: float = 0.10,
        timeout_steps: int = 120,
        arm_zero: bool = False,
        token_hold: bool = False,
    ) -> None:
        if num_envs <= 0 or timeout_steps <= 0:
            raise ValueError("num_envs and timeout_steps must be positive")
        if not 0.0 <= exit_proximity <= entry_proximity <= 1.0:
            raise ValueError("require 0 <= exit_proximity <= entry_proximity <= 1")
        if not 0.0 <= low_force < high_force:
            raise ValueError("require 0 <= low_force < high_force")
        if step <= 0.0 or max_travel < 0.0:
            raise ValueError("step must be positive and max_travel non-negative")
        self.entry_proximity = float(entry_proximity)
        self.exit_proximity = float(exit_proximity)
        self.low_force = float(low_force)
        self.high_force = float(high_force)
        self.step = float(step)
        self.max_travel = float(max_travel)
        self.timeout_steps = int(timeout_steps)
        self.arm_zero = bool(arm_zero)
        self.token_hold = bool(token_hold)

        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.blocked = torch.zeros_like(self.active)
        self.age = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.entry_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.servo_offset = torch.zeros((num_envs, 5), device=device)
        self.held_token = torch.zeros((num_envs, 9), device=device)

    def reset(self, mask: torch.Tensor) -> None:
        if mask.shape != self.active.shape:
            raise ValueError("reset mask must have shape [num_envs]")
        self.active[mask] = False
        self.blocked[mask] = False
        self.age[mask] = 0
        self.servo_offset[mask] = 0.0

    def apply(
        self,
        base_action: torch.Tensor,
        observation: torch.Tensor,
        force_by_distal: torch.Tensor,
        *,
        external_gate: torch.Tensor | None = None,
        entry_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if base_action.shape != (self.active.numel(), 21):
            raise ValueError(f"expected base action [N,21], got {tuple(base_action.shape)}")
        if observation.shape != (self.active.numel(), 115):
            raise ValueError(f"expected observation [N,115], got {tuple(observation.shape)}")
        if force_by_distal.shape != (self.active.numel(), 5):
            raise ValueError("force_by_distal must have shape [N,5]")

        proximity = observation[:, 92:97].mean(dim=-1)
        latched = observation[:, 106] >= 0.5
        allowed = torch.ones_like(self.active)
        if external_gate is not None:
            if external_gate.shape != self.active.shape:
                raise ValueError("external_gate must have shape [N]")
            allowed &= external_gate.bool()
        may_enter = allowed
        if entry_gate is not None:
            if entry_gate.shape != self.active.shape:
                raise ValueError("entry_gate must have shape [N]")
            may_enter = may_enter & entry_gate.bool()

        # A timed-out attempt cannot immediately re-enter.  Moving away from the handle, latch,
        # or an explicit reset clears the block and permits a genuinely new close attempt.
        self.blocked &= (proximity >= self.exit_proximity) & ~latched & allowed
        remain = (
            self.active
            & (proximity >= self.exit_proximity)
            & ~latched
            & allowed
            & (self.age < self.timeout_steps)
        )
        timed_out = self.active & (self.age >= self.timeout_steps)
        self.blocked |= timed_out
        self.active.copy_(remain)
        enter = (
            ~self.active
            & ~self.blocked
            & (proximity >= self.entry_proximity)
            & ~latched
            & may_enter
        )
        if bool(enter.any()):
            self.servo_offset[enter] = 0.0
            self.held_token[enter] = base_action[enter, 7:16]
            self.age[enter] = 0
            self.entry_count[enter] += 1
        self.active |= enter

        close = force_by_distal < self.low_force
        release = force_by_distal > self.high_force
        increment = torch.where(
            close,
            torch.full_like(force_by_distal, self.step),
            torch.where(
                release, torch.full_like(force_by_distal, -self.step), torch.zeros_like(force_by_distal)
            ),
        )
        proposed = self.servo_offset + increment
        self.servo_offset.copy_(proposed.clamp(0.0, self.max_travel))
        self.age[self.active] += 1

        result = base_action.clone()
        if self.arm_zero:
            result[self.active, :7] = 0.0
        if self.token_hold:
            result[self.active, 7:16] = self.held_token[self.active]
        result[self.active, 16:21] = (
            base_action[self.active, 16:21] + self.servo_offset[self.active]
        )
        result.clamp_(-1.0, 1.0)
        delta = result - base_action
        return result, self.active.clone(), delta
