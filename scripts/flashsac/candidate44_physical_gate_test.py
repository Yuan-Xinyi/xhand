#!/usr/bin/env python3
"""Simulation-free tests for Candidate44's paired physical gate."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from candidate44_physical_gate import (  # noqa: E402
    compare_physical_gates,
    validate_offline_gate_binding,
)


def _evaluation(
    *,
    checkpoint: str,
    seed: int,
    probability: float,
    success: int,
    grasped: int,
    unlatched: int,
    unsafe: int,
) -> dict[str, Any]:
    return {
        "status": "complete",
        "seed": seed,
        "requested_episodes": 512,
        "completed_episodes": 512,
        "num_envs": 512,
        "task_mode": "full_task",
        "checkpoint_task_mode": "full_task",
        "policy": "deterministic_tanh_actor_mean",
        "max_episode_steps": 1000,
        "checkpoint": checkpoint,
        "curriculum": {
            "probability": probability,
            "boundary": "close_start",
            "joint_noise": 0.0,
            "dataset": None if probability == 0.0 else "/fixed/holdout.pt",
            "dataset_sha256": None if probability == 0.0 else "a" * 64,
        },
        "events": {
            "success": success,
            "unsafe_force": unsafe,
            "unlatched_clearance_ge_5cm": unlatched,
        },
        "funnel": {"ever_grasped": grasped},
        "strict_success_rate": success / 512,
    }


def _passing_inputs() -> tuple[dict[str, Any], ...]:
    return (
        _evaluation(
            checkpoint="c2",
            seed=360,
            probability=0.0,
            success=1,
            grasped=2,
            unlatched=0,
            unsafe=0,
        ),
        _evaluation(
            checkpoint="c44",
            seed=360,
            probability=0.0,
            success=1,
            grasped=28,
            unlatched=5,
            unsafe=5,
        ),
        _evaluation(
            checkpoint="c2",
            seed=361,
            probability=1.0,
            success=135,
            grasped=224,
            unlatched=153,
            unsafe=4,
        ),
        _evaluation(
            checkpoint="c44",
            seed=361,
            probability=1.0,
            success=110,
            grasped=199,
            unlatched=168,
            unsafe=9,
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checkpoint(path: Path, *, actor: bytes, task: bytes) -> None:
    path.mkdir(parents=True)
    (path / "actor.pt").write_bytes(actor)
    (path / "task_contract.json").write_bytes(task)


def _offline_report(baseline: Path, candidate: Path) -> dict[str, Any]:
    return {
        "kind": "pick_tool_candidate44_phase0_correction_offline_gate_v1",
        "status": "complete",
        "baseline_checkpoint": str(baseline),
        "candidate_checkpoint": str(candidate),
        "integrity": {
            "all_pass": True,
            "checks": {"checkpoint_evidence": True},
            "baseline_actor_sha256": _sha256(baseline / "actor.pt"),
            "candidate_actor_sha256": _sha256(candidate / "actor.pt"),
            "baseline_task_contract_sha256": _sha256(
                baseline / "task_contract.json"
            ),
            "candidate_task_contract_sha256": _sha256(
                candidate / "task_contract.json"
            ),
        },
        "registered_gate": {
            "all_pass": True,
            "decision": "release_to_paired_physical_gate",
            "checks": {"integrity": True, "phase0": True},
        },
    }


def _bound_inputs(root: Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    baseline = root / "baseline"
    candidate = root / "candidate"
    _write_checkpoint(baseline, actor=b"baseline actor", task=b'{"task":"full"}')
    _write_checkpoint(candidate, actor=b"candidate actor", task=b'{"task":"full"}')
    evaluations = list(_passing_inputs())
    for index in (0, 2):
        evaluations[index]["checkpoint"] = str(baseline)
    for index in (1, 3):
        evaluations[index]["checkpoint"] = str(candidate)
    return _offline_report(baseline, candidate), tuple(evaluations)


def test_registered_boundaries_pass_inclusively() -> None:
    result = compare_physical_gates(*_passing_inputs(), offline_gate_passed=True)
    assert result["all_pass"]
    assert result["ordinary_delta"]["ever_grasped_count"] == 26
    assert result["decision"] == (
        "release_to_new_preregistered_reverse_curriculum_candidate"
    )


def test_every_gate_is_required() -> None:
    inputs = _passing_inputs()
    assert not compare_physical_gates(*inputs, offline_gate_passed=False)["all_pass"]
    mutations = (
        (1, "funnel", "ever_grasped", 27),
        (1, "events", "success", 0),
        (1, "events", "unlatched_clearance_ge_5cm", 6),
        (1, "events", "unsafe_force", 6),
        (3, "events", "success", 109),
        (3, "funnel", "ever_grasped", 198),
        (3, "events", "unlatched_clearance_ge_5cm", 169),
        (3, "events", "unsafe_force", 10),
    )
    for index, container, key, value in mutations:
        changed = list(_passing_inputs())
        changed[index] = dict(changed[index])
        changed[index][container] = dict(changed[index][container])
        changed[index][container][key] = value
        if container == "events" and key == "success":
            changed[index]["strict_success_rate"] = value / 512
        result = compare_physical_gates(*changed, offline_gate_passed=True)
        assert not result["all_pass"], (index, container, key)


def test_offline_gate_is_bound_to_exact_checkpoint_bytes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        offline, evaluations = _bound_inputs(Path(directory))
        audit = validate_offline_gate_binding(offline, *evaluations)
        assert audit["all_pass"]
        assert audit["reported_hashes"] == audit["computed_hashes"]
        assert audit["baseline_checkpoint"] == str(
            Path(offline["baseline_checkpoint"]).resolve()
        )


def test_old_or_mismatched_offline_evidence_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        offline, evaluations = _bound_inputs(root)

        old_report = {"registered_gate": {"all_pass": True}}
        try:
            validate_offline_gate_binding(old_report, *evaluations)
        except ValueError:
            pass
        else:
            raise AssertionError("old unbound offline report was accepted")

        wrong_checkpoint_evaluations = list(evaluations)
        wrong_checkpoint_evaluations[3] = dict(wrong_checkpoint_evaluations[3])
        wrong_checkpoint_evaluations[3]["checkpoint"] = str(root / "wrong_candidate")
        try:
            validate_offline_gate_binding(
                offline, *tuple(wrong_checkpoint_evaluations)
            )
        except ValueError:
            pass
        else:
            raise AssertionError("evaluation from the wrong checkpoint was accepted")

        wrong_hash = dict(offline)
        wrong_hash["integrity"] = dict(offline["integrity"])
        wrong_hash["integrity"]["candidate_actor_sha256"] = "0" * 64
        try:
            validate_offline_gate_binding(wrong_hash, *evaluations)
        except ValueError:
            pass
        else:
            raise AssertionError("offline report with a wrong actor hash was accepted")

        missing_task_hash = dict(offline)
        missing_task_hash["integrity"] = dict(offline["integrity"])
        del missing_task_hash["integrity"]["baseline_task_contract_sha256"]
        try:
            validate_offline_gate_binding(missing_task_hash, *evaluations)
        except ValueError:
            pass
        else:
            raise AssertionError("offline report without baseline task hash was accepted")


def test_symlink_checkpoint_evidence_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        offline, evaluations = _bound_inputs(root)
        candidate_actor = root / "candidate" / "actor.pt"
        actor_target = root / "candidate-actor-target.pt"
        actor_target.write_bytes(candidate_actor.read_bytes())
        candidate_actor.unlink()
        candidate_actor.symlink_to(actor_target)
        try:
            validate_offline_gate_binding(offline, *evaluations)
        except ValueError:
            pass
        else:
            raise AssertionError("symlink actor evidence was accepted")


def main() -> None:
    test_registered_boundaries_pass_inclusively()
    print("[PASS] paired physical thresholds are inclusive")
    test_every_gate_is_required()
    print("[PASS] every offline, ordinary and close-start gate is mandatory")
    test_offline_gate_is_bound_to_exact_checkpoint_bytes()
    print("[PASS] offline report is bound to exact checkpoints and evidence hashes")
    test_old_or_mismatched_offline_evidence_is_rejected()
    print("[PASS] old and mismatched offline evidence is rejected")
    test_symlink_checkpoint_evidence_is_rejected()
    print("[PASS] symlink checkpoint evidence is rejected")


if __name__ == "__main__":
    main()
