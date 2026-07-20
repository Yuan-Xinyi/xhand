#!/usr/bin/env python3
"""Fast tensor-only regression for pick_tool_token grasp/transport invariants.

This intentionally does not launch Isaac Sim, so it can run in seconds and still catches the
link/COM rigid-motion error, weak contact topology, back-of-hand presses and latch-release bugs.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT
    / "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/grasp_signals.py"
)
SPEC = importlib.util.spec_from_file_location("pick_tool_grasp_signals", MODULE_PATH)
signals = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(signals)

ACTION_MODULE_PATH = (
    REPO_ROOT
    / "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/hybrid_action.py"
)
ACTION_SPEC = importlib.util.spec_from_file_location("pick_tool_hybrid_action", ACTION_MODULE_PATH)
hybrid_action = importlib.util.module_from_spec(ACTION_SPEC)
assert ACTION_SPEC.loader is not None
ACTION_SPEC.loader.exec_module(hybrid_action)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_rigid_hold_quality() -> None:
    n = 5
    palm_pos = torch.zeros(n, 3)
    object_pos = torch.tensor((0.08, 0.0, 0.0)).repeat(n, 1)
    palm_lin = torch.zeros(n, 3)
    palm_ang = torch.zeros(n, 3)
    object_lin = torch.zeros(n, 3)
    object_ang = torch.zeros(n, 3)

    # 0 stationary; 1 common translation; 2 rigid rotation; 3 linear fling; 4 angular slip.
    palm_lin[1] = torch.tensor((0.7, -0.2, 0.1))
    object_lin[1] = palm_lin[1]
    palm_ang[2] = torch.tensor((0.0, 0.0, 4.0))
    object_ang[2] = palm_ang[2]
    object_lin[2] = torch.cross(palm_ang[2], object_pos[2] - palm_pos[2], dim=-1)
    object_lin[3] = torch.tensor((3.0, 0.0, 0.0))
    object_ang[4] = torch.tensor((0.0, 0.0, 3.0))

    quality, slip_lin, slip_ang = signals.rigid_hold_quality(
        palm_pos, palm_lin, palm_ang, object_pos, object_lin, object_ang, 0.3, 3.0
    )
    check(torch.all(slip_lin[:3] < 1.0e-7).item(), f"rigid linear slip is nonzero: {slip_lin[:3]}")
    check(torch.all(slip_ang[:3] < 1.0e-7).item(), f"rigid angular slip is nonzero: {slip_ang[:3]}")
    check(torch.all(quality[:3] > 0.999999).item(), f"rigid hold quality is below one: {quality[:3]}")
    check(quality[3].item() < 1.0e-4, f"3m/s fling was not rejected: q={quality[3].item()}")
    check(abs(quality[4].item() - math.exp(-1.0)) < 1.0e-6, "angular slip scale changed")
    print("PASS rigid hold: stationary/translation/rotation=1, fling≈0, angular-slip=e^-1")


def make_wrap_case() -> dict[str, torch.Tensor]:
    force = torch.tensor([[10.0, 10.0, 10.0, 0.0, 0.0]])
    distance = torch.zeros(1, 5)
    # thumb on +x side; index/middle on the opposite side.
    normal = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]])
    alignment = torch.full((1, 5), 0.9)
    palm_facing = torch.tensor([0.9])
    return {"force": force, "distance": distance, "normal": normal,
            "alignment": alignment, "palm_facing": palm_facing}


def evaluate_wrap(case: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return signals.wrap_quality(
        case["force"], case["distance"], case["normal"], case["alignment"], case["palm_facing"],
        0, torch.tensor((1, 2, 3, 4)), force_threshold=0.2, force_saturation=5.0,
        surface_margin=0.008, palm_facing_min=0.5, alignment_min=0.3, opposition_min=0.5,
    )


def test_wrap_quality() -> None:
    legal = evaluate_wrap(make_wrap_case())
    check(legal["quality"].item() > 0.79, f"legal wrap too low: {legal['quality'].item()}")
    check(legal["other_contact_count"].item() == 2, "legal wrap contact count changed")
    extra_wrong_side = make_wrap_case()
    extra_wrong_side["force"][:, 3] = 100.0
    check(
        evaluate_wrap(extra_wrong_side)["quality"].item() > 0.79,
        "a strong wrong-side collision hid two legal opposed contacts",
    )

    # Force, alignment and opposition are independent min-gates.  The aligned/opposed product may
    # rank candidate fingers, but must not be multiplied into force coverage and then gated twice.
    moderate = make_wrap_case()
    moderate["force"][:, :3] = 5.0
    moderate["alignment"][:, :3] = 0.65  # normalized alignment score = 0.5
    moderate["palm_facing"][:] = 0.75    # normalized palm score = 0.5
    moderate["normal"][:, 1:3] = torch.tensor((-0.5, 0.8660254, 0.0))  # opposition score = 0.5
    moderate_quality = evaluate_wrap(moderate)["quality"].item()
    check(
        abs(moderate_quality - 0.5) < 1.0e-5,
        f"independent wrap gates were multiplied/double-counted: q={moderate_quality}",
    )

    cases = {}
    cases["thumb-only"] = make_wrap_case()
    cases["thumb-only"]["force"][:, 1:] = 0.0
    cases["thumb+one"] = make_wrap_case()
    cases["thumb+one"]["force"][:, 2:] = 0.0
    cases["back-of-hand"] = make_wrap_case()
    cases["back-of-hand"]["palm_facing"][:] = 0.2
    cases["same-side"] = make_wrap_case()
    cases["same-side"]["normal"][:, 1:3] = torch.tensor((1.0, 0.0, 0.0))
    cases["misaligned-pad"] = make_wrap_case()
    cases["misaligned-pad"]["alignment"][:, 2] = 0.1
    cases["off-handle"] = make_wrap_case()
    cases["off-handle"]["distance"][:, 2] = 0.02

    for name, case in cases.items():
        quality = evaluate_wrap(case)["quality"].item()
        check(quality == 0.0, f"{name} incorrectly passed with q={quality}")
    print("PASS wrap truth table: thumb+2 opposed pads required; back/misaligned/off-handle rejected")


def test_staged_close_quality() -> None:
    def close(case: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        palm_score = torch.clamp((case["palm_facing"] - 0.5) / 0.5, 0.0, 1.0)
        return signals.staged_close_quality(
            case["force"], case["distance"], case["distance"] < 0.008,
            case["normal"], case["alignment"], palm_score, 0, torch.tensor((1, 2, 3, 4)),
            alignment_min=0.3, opposition_min=0.5, proximity_scale_far=0.08,
            proximity_scale_near=0.02, force_saturation=5.0,
        )

    values = []
    for contacts in (0, 1, 2, 3):
        case = make_wrap_case()
        case["force"].zero_()
        if contacts:
            case["force"][:, :contacts] = 10.0
        values.append(close(case)["close_quality"].item())
    check(values[0] > 0.0, "near-handle shaping disappeared before contact")
    check(values[0] < values[1] < values[2] < values[3], f"close stages are not monotonic: {values}")

    back = make_wrap_case()
    back["palm_facing"][:] = 0.2
    check(close(back)["close_quality"].item() == 0.0, "back-of-hand close shaping is nonzero")
    wrong = make_wrap_case()
    wrong["normal"][:, 1:3] = torch.tensor((1.0, 0.0, 0.0))
    check(
        close(wrong)["close_quality"].item() < close(make_wrap_case())["close_quality"].item() / 2.0,
        "same-side contacts receive nearly as much close shaping as opposed contacts",
    )
    print("PASS staged close: near -> thumb -> thumb+1 -> thumb+2 is monotonic and direction-gated")


def test_schmitt_latch() -> None:
    grasped = torch.tensor([False])
    confirm = torch.zeros(1, dtype=torch.long)
    release = torch.zeros(1, dtype=torch.long)
    new_events = 0
    for _ in range(4):
        grasped, confirm, release, newly, _ = signals.update_grasp_latch(
            torch.tensor([0.8]), grasped, confirm, release,
            high_threshold=0.45, low_threshold=0.2, confirm_steps=4, release_steps=6,
        )
        new_events += int(newly.item())
    check(grasped.item() and new_events == 1, "grasp did not confirm exactly once")

    # Dead-band quality preserves the latch.
    for _ in range(10):
        grasped, confirm, release, _, _ = signals.update_grasp_latch(
            torch.tensor([0.3]), grasped, confirm, release,
            high_threshold=0.45, low_threshold=0.2, confirm_steps=4, release_steps=6,
        )
    check(grasped.item(), "Schmitt dead band released a valid grasp")

    # Geometry/transport quality is now authoritative: the latch releases even if a caller still
    # observes raw forces elsewhere.
    for _ in range(6):
        grasped, confirm, release, _, _ = signals.update_grasp_latch(
            torch.tensor([0.0]), grasped, confirm, release,
            high_threshold=0.45, low_threshold=0.2, confirm_steps=4, release_steps=6,
        )
    check(not grasped.item(), "poor grasp quality did not release the latch")
    print("PASS Schmitt latch: one confirmation, dead-band hold, quality-driven release")


def test_asymmetric_joint_residual() -> None:
    base = torch.tensor(
        [[0.1, 0.4, -0.2, 0.7, 0.6, 0.0], [0.2, 0.5, -0.1, 0.8, 0.7, 0.1],
         [0.3, 0.6, 0.0, 0.9, 0.8, 0.2]]
    )
    lower = base - torch.tensor((0.5, 0.4, 0.3, 0.2, 0.6, 0.1))
    upper = base + torch.tensor((0.4, 0.8, 0.2, 0.5, 0.3, 0.7))
    joint_ids = torch.tensor((4, 1))  # deliberately not articulation order
    residual = torch.tensor(((-1.0, 1.0), (0.0, 0.0), (0.5, -0.25)))
    target, delta = hybrid_action.apply_asymmetric_joint_residual(
        base, lower, upper, residual, joint_ids
    )
    check(torch.equal(target[0, joint_ids], torch.tensor((0.0, 1.2))), "-1/+1 missed limits")
    check(torch.equal(target[1], base[1]), "zero residual changed the token target")
    expected_row2 = torch.tensor((0.95, 0.5))
    check(torch.allclose(target[2, joint_ids], expected_row2), "asymmetric residual scale changed")
    untouched = torch.tensor((0, 2, 3, 5))
    check(torch.equal(target[:, untouched], base[:, untouched]), "advanced indexing changed other joints")
    check(torch.all(target >= lower).item() and torch.all(target <= upper).item(), "target escaped limits")
    check(torch.allclose(delta[2], expected_row2 - base[2, joint_ids]), "reported delta is wrong")
    encoded = hybrid_action.invert_asymmetric_joint_residual(target, base, lower, upper, joint_ids)
    roundtrip, _ = hybrid_action.apply_asymmetric_joint_residual(base, lower, upper, encoded, joint_ids)
    check(torch.allclose(roundtrip, target), "absolute distal target failed residual round trip")
    print("PASS hybrid action: full asymmetric range, zero identity, shuffled joints and rows")


if __name__ == "__main__":
    test_rigid_hold_quality()
    test_wrap_quality()
    test_staged_close_quality()
    test_schmitt_latch()
    test_asymmetric_joint_residual()
    print("ALL GRASP SIGNAL TESTS PASSED")
