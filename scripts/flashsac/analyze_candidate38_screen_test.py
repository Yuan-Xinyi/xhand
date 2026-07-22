#!/usr/bin/env python3
"""CPU-only tests for the sealed Candidate 38 screen."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

import analyze_candidate35_phase1_screen as historical_analysis
from analyze_candidate38_screen import (
    EVENTS,
    NUM_ENVS,
    REPLICATES,
    SEEDS,
    compute_screen,
    validate_manifest_payload,
)
from online_close_ab import assignment_candidate_mask


def test_historical_analyzer_globals_are_isolated() -> None:
    assert historical_analysis.MANIFEST_KIND == (
        "pick_tool_candidate35_phase1_screen_manifest_v1"
    )
    assert historical_analysis.SEEDS == (320, 321, 322)
    assert historical_analysis.RUN_ORDER == (
        "320a",
        "320b",
        "321b",
        "321a",
        "322a",
        "322b",
    )
    historical_manifest = json.loads(
        Path(__file__)
        .with_name("candidate35_phase1_screen_manifest.json")
        .read_text(encoding="utf-8")
    )
    historical_analysis.validate_manifest_payload(historical_manifest)


def _expect(error_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _cube(*, passing: bool) -> dict[str, torch.Tensor]:
    shape = (len(SEEDS), len(REPLICATES), NUM_ENVS)
    assignment = torch.stack(
        [
            torch.stack(
                [
                    assignment_candidate_mask(
                        seed=seed, num_envs=NUM_ENVS, replicate=replicate
                    )
                    for replicate in REPLICATES
                ]
            )
            for seed in SEEDS
        ]
    )
    result = {
        "assignment_candidate": assignment,
        "triggered": torch.ones(shape, dtype=torch.bool),
        **{event: torch.zeros(shape, dtype=torch.bool) for event in EVENTS},
    }
    for seed_index in range(len(SEEDS)):
        candidate = assignment[seed_index].flatten().nonzero().flatten()
        baseline = (~assignment[seed_index]).flatten().nonzero().flatten()
        for event, candidate_count, baseline_count in (
            ("success", 24 if passing else 19, 20),
            ("ever_grasped", 34 if passing else 29, 32),
            ("unlatched_clearance_ge_5cm", 8, 9),
        ):
            flat = result[event][seed_index].flatten()
            flat[candidate[:candidate_count]] = True
            flat[baseline[:baseline_count]] = True
        if not passing and seed_index == 0:
            result["unsafe_force"][seed_index].flatten()[candidate[0]] = True
    return result


def test_screen_passes_only_when_every_gate_passes() -> None:
    passed = compute_screen(_cube(passing=True))
    assert passed["all_gates_pass"] is True
    assert abs(passed["aggregate"]["conditional_success_delta"] - 0.0625) < 1.0e-12
    assert passed["aggregate"]["positive_conditional_success_seeds"] == 3

    failed = compute_screen(_cube(passing=False))
    assert failed["all_gates_pass"] is False
    assert failed["gates"]["conditional_success_delta"]["pass"] is False
    assert failed["gates"]["candidate_triggered_unsafe_force"]["pass"] is False


def test_cube_tampering_is_rejected() -> None:
    malformed = _cube(passing=True)
    malformed["assignment_candidate"][0, 1, 0] ^= True
    _expect(ValueError, compute_screen, malformed)
    malformed = _cube(passing=True)
    malformed["triggered"] = malformed["triggered"].float()
    _expect(ValueError, compute_screen, malformed)


def test_manifest_identity_and_gate_tampering_is_rejected() -> None:
    path = Path(__file__).with_name("candidate38_screen_manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest_payload(manifest)

    malformed = copy.deepcopy(manifest)
    malformed["gates"]["conditional_success_delta_min"] = -1.0
    _expect(ValueError, validate_manifest_payload, malformed)
    malformed = copy.deepcopy(manifest)
    malformed["candidate"]["actor_sha256"] = malformed["baseline"][
        "actor_sha256"
    ]
    _expect(ValueError, validate_manifest_payload, malformed)
    malformed = copy.deepcopy(manifest)
    malformed["collector"]["source_path_count"] -= 1
    _expect(ValueError, validate_manifest_payload, malformed)
    malformed = copy.deepcopy(manifest)
    malformed["run_order"][0] = "325b"
    _expect(ValueError, validate_manifest_payload, malformed)


if __name__ == "__main__":
    test_historical_analyzer_globals_are_isolated()
    test_screen_passes_only_when_every_gate_passes()
    test_cube_tampering_is_rejected()
    test_manifest_identity_and_gate_tampering_is_rejected()
    print("analyze_candidate38_screen_test: PASS")
