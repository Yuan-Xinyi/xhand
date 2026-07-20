#!/usr/bin/env python3
"""CPU invariants for the stateful tactile close controller."""

from __future__ import annotations

import torch

from tactile_close_controller import TactileCloseController


def main() -> None:
    base = torch.linspace(-0.4, 0.4, 21).repeat(3, 1)
    obs = torch.zeros((3, 115))
    obs[1:, 92:97] = 0.10
    force = torch.zeros((3, 5))
    controller = TactileCloseController(3, "cpu", step=0.01, max_travel=0.02)
    action, gate, delta = controller.apply(
        base, obs, force, entry_gate=torch.tensor([True, False, True])
    )
    assert gate.tolist() == [False, False, True]
    assert torch.equal(action[0], base[0])
    assert torch.equal(action[:, :16], base[:, :16])
    assert torch.allclose(delta[gate, 16:], torch.full((1, 5), 0.01))

    force[1] = 25.0
    action, gate, _ = controller.apply(
        base, obs, force, entry_gate=torch.tensor([True, True, False])
    )
    assert torch.allclose(action[1, 16:], base[1, 16:])
    assert torch.allclose(action[2, 16:], base[2, 16:] + 0.02)
    shifted_base = base.clone()
    shifted_base[:, 16:21] += 0.07
    shifted, _, _ = controller.apply(
        shifted_base, obs, force, entry_gate=torch.tensor([False, False, False])
    )
    assert torch.allclose(shifted[2, 16:], shifted_base[2, 16:] + 0.02)
    obs[2, 106] = 1.0
    action, gate, _ = controller.apply(base, obs, force)
    assert gate.tolist() == [False, True, False]
    assert torch.equal(action[2], base[2])

    structured = TactileCloseController(
        3, "cpu", step=0.01, arm_zero=True, token_hold=True
    )
    first, gate, _ = structured.apply(base, obs, force)
    changed_base = base.clone()
    changed_base[:, :16] += 0.1
    second, _, _ = structured.apply(changed_base, obs, force)
    assert torch.equal(second[1, :7], torch.zeros(7))
    assert torch.equal(second[1, 7:16], first[1, 7:16])
    assert torch.equal(second[~gate], changed_base[~gate])
    print("PASS: default servo changes distal5 only inside the pre-latch gate")
    print("PASS: the servo is additive to the live base action and latch restores the base actor")
    print("PASS: arm-zero/token-hold remain explicit independent ablations")


if __name__ == "__main__":
    main()
