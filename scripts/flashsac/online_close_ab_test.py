#!/usr/bin/env python3
"""Simulation-free tests for the online CLOSE A/B evidence contract."""

from __future__ import annotations

import copy
import argparse
import json
from pathlib import Path
import tempfile

import torch
import online_close_ab as ab_contract

from online_close_ab import (
    ACTION_DIM,
    ARTIFACT_KIND,
    ASSIGNMENT_CONTRACT,
    ASSIGNMENT_SALT,
    COLLECTION_CONTRACT,
    FORMAT_VERSION,
    HANDOFF_HOLD_STEPS,
    HANDOFF_MIN_SCORE,
    MAX_EPISODE_ACTIONS,
    OBSERVATION_DIM,
    OUTCOME_NAMES,
    REPORT_KIND,
    RUNTIME_ASSET_NAMES,
    assignment_candidate_mask,
    build_artifact,
    publish_artifact_and_report_no_clobber,
    publish_json_no_clobber,
    select_executed_action,
    summarize_artifact,
    validate_artifact,
    validate_report,
)
from collect_online_close_ab import (
    CHECKPOINT_FILES,
    build_spec,
    publish_failure_attempt,
    source_fingerprints,
)


SHA = "a" * 64


def _expect(error_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _metadata(num_envs: int = 8, replicate: str = "a") -> dict[str, object]:
    return {
        "kind": ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "collection_contract": COLLECTION_CONTRACT,
        "assignment_contract": ASSIGNMENT_CONTRACT,
        "assignment_salt": ASSIGNMENT_SALT,
        "handoff_min_score": HANDOFF_MIN_SCORE,
        "handoff_hold_steps": HANDOFF_HOLD_STEPS,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "max_episode_actions": MAX_EPISODE_ACTIONS,
        "seed": 290,
        "replicate": replicate,
        "num_envs": num_envs,
        "baseline_checkpoint": "/checkpoint/base",
        "candidate_checkpoint": "/checkpoint/candidate",
        "search_checkpoint": "/checkpoint/search.pth",
        "baseline_actor_sha256": "1" * 64,
        "candidate_actor_sha256": "2" * 64,
        "baseline_task_contract_sha256": "3" * 64,
        "candidate_task_contract_sha256": "3" * 64,
        "common_frozen_lift_actor_sha256": "4" * 64,
        "common_frozen_lift_semantic_sha256": "5" * 64,
        "common_frozen_lift_source_actor_sha256": "7" * 64,
        "search_checkpoint_sha256": "6" * 64,
        "baseline_bridge_state_sha256": "8" * 64,
        "candidate_bridge_state_sha256": "9" * 64,
        "flashsac_upstream_commit": "a" * 40,
        "flashsac_fork_commit": "c" * 40,
        "source_sha256": source_fingerprints(Path(__file__).resolve().parents[2]),
        "runtime_asset_sha256": {name: SHA for name in RUNTIME_ASSET_NAMES},
        "git": {
            "commit": "b" * 40,
            "branch": "test-branch",
            "source_files_dirty": False,
            "flashsac_commit": "c" * 40,
            "flashsac_dirty": False,
        },
        "runtime": {
            "python": "test",
            "torch": "test",
            "cuda": "test",
            "cudnn": 1,
            "cuda_device_index": 0,
            "cuda_device_name": "test",
            "cuda_device_capability": [9, 0],
            "isaac_sim": "test",
            "packages": {
                "isaaclab": "test",
                "isaaclab_tasks": "test",
                "isaaclab_assets": "test",
                "numpy": "test",
                "gymnasium": "test",
            },
            "nvidia_smi_inventory": "test",
            "platform": "test",
            "seed": 290,
        },
    }


def _tensors(num_envs: int = 8, replicate: str = "a") -> dict[str, torch.Tensor]:
    assignment = assignment_candidate_mask(
        seed=290, num_envs=num_envs, replicate=replicate
    )
    triggered = torch.arange(num_envs) < (num_envs - 2)
    success = torch.tensor([True, False, False, True, False, False, False, False])
    timeout = torch.tensor([False, False, True, False, False, True, False, False])
    if num_envs != 8:
        raise ValueError("test fixture currently requires eight environments")
    failure = ~(success | timeout)
    maximum = torch.tensor([0.25, 0.06, 0.01, 0.22, 0.07, 0.00, 0.08, 0.09])
    episode_length = torch.full((num_envs,), 100, dtype=torch.long)
    episode_length[timeout] = MAX_EPISODE_ACTIONS
    result = {
        "env_slot": torch.arange(num_envs, dtype=torch.long),
        "assignment_candidate": assignment,
        "triggered": triggered,
        "trigger_step": torch.where(
            triggered,
            torch.arange(num_envs, dtype=torch.long) + 10,
            torch.full((num_envs,), -1, dtype=torch.long),
        ),
        "trigger_score": torch.where(
            triggered,
            torch.full((num_envs,), 0.35, dtype=torch.float32),
            torch.zeros(num_envs, dtype=torch.float32),
        ),
        "episode_length": episode_length,
        "max_true_clearance_m": maximum,
        "success": success,
        "failure": failure,
        "time_out": timeout,
        "dropped": torch.zeros(num_envs, dtype=torch.bool),
        "unsafe_force": torch.zeros(num_envs, dtype=torch.bool),
        "unlatched_clearance_ge_5cm": failure.clone(),
        "ever_grasped": success.clone(),
        "ever_clearance_ge_20cm": maximum >= 0.20,
    }
    assert set(result) == {
        "env_slot",
        "assignment_candidate",
        "triggered",
        "trigger_step",
        "trigger_score",
        "episode_length",
        "max_true_clearance_m",
        *OUTCOME_NAMES,
    }
    return result


def _report(artifact: dict[str, object]) -> dict[str, object]:
    metadata = artifact["metadata"]
    tensors = artifact["tensors"]
    assert isinstance(metadata, dict) and isinstance(tensors, dict)
    return {
        "kind": REPORT_KIND,
        "status": "complete",
        "seed": metadata["seed"],
        "replicate": metadata["replicate"],
        "num_envs": metadata["num_envs"],
        "vector_steps": int(tensors["episode_length"].max().item()),
        "collection_contract": COLLECTION_CONTRACT,
        "baseline_actor_sha256": metadata["baseline_actor_sha256"],
        "candidate_actor_sha256": metadata["candidate_actor_sha256"],
        "common_frozen_lift_actor_sha256": metadata[
            "common_frozen_lift_actor_sha256"
        ],
        "search_checkpoint_sha256": metadata["search_checkpoint_sha256"],
        "summary": summarize_artifact(artifact),
    }


def test_assignment_is_balanced_stable_and_complementary() -> None:
    a = assignment_candidate_mask(seed=290, num_envs=64, replicate="a")
    b = assignment_candidate_mask(seed=290, num_envs=64, replicate="b")
    assert int(a.sum()) == 32
    assert torch.equal(b, ~a)
    assert torch.equal(
        a, assignment_candidate_mask(seed=290, num_envs=64, replicate="a")
    )
    assert not torch.equal(
        a, assignment_candidate_mask(seed=291, num_envs=64, replicate="a")
    )
    _expect(ValueError, assignment_candidate_mask, seed=290, num_envs=63, replicate="a")


def test_action_selection_has_no_pretreatment_leak_and_common_lift() -> None:
    search = torch.full((4, ACTION_DIM), -0.5)
    baseline = torch.full((4, ACTION_DIM), 0.1)
    candidate = torch.full((4, ACTION_DIM), 0.9)
    active = torch.tensor([False, True, True, False])
    assignment = torch.tensor([True, False, True, False])
    latch = torch.tensor([False, False, False, True])
    candidate[3] = baseline[3]
    selected = select_executed_action(
        search_action=search,
        baseline_action=baseline,
        candidate_action=candidate,
        option_active=active,
        assignment_candidate=assignment,
        public_latch=latch,
    )
    torch.testing.assert_close(selected[0], search[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(selected[1], baseline[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(selected[2], candidate[2], rtol=0.0, atol=0.0)
    torch.testing.assert_close(selected[3], search[3], rtol=0.0, atol=0.0)

    candidate[3, 0] += 1.0e-7
    _expect(
        RuntimeError,
        select_executed_action,
        search_action=search,
        baseline_action=baseline,
        candidate_action=candidate,
        option_active=active,
        assignment_candidate=assignment,
        public_latch=latch,
    )


def test_artifact_validation_summary_and_corruption_rejection() -> None:
    artifact = build_artifact(metadata=_metadata(), tensors=_tensors())
    validate_artifact(artifact)
    summary = summarize_artifact(artifact)
    assert summary["all_assigned"]["baseline"]["episodes"] == 4
    assert summary["all_assigned"]["candidate"]["episodes"] == 4
    assert (
        summary["triggered"]["baseline"]["episodes"]
        + summary["triggered"]["candidate"]["episodes"]
        == 6
    )

    corrupt = copy.deepcopy(artifact)
    corrupt["tensors"]["assignment_candidate"][0] ^= True
    _expect(ValueError, validate_artifact, corrupt)
    corrupt = copy.deepcopy(artifact)
    corrupt["tensors"]["trigger_score"][0] = 0.29
    _expect(ValueError, validate_artifact, corrupt)
    corrupt = copy.deepcopy(artifact)
    corrupt["tensors"]["success"][1] = True
    _expect(ValueError, validate_artifact, corrupt)
    corrupt = copy.deepcopy(artifact)
    corrupt["metadata"]["candidate_task_contract_sha256"] = "7" * 64
    _expect(ValueError, validate_artifact, corrupt)
    corrupt = copy.deepcopy(artifact)
    corrupt["tensors"]["trigger_step"][0] = 2
    _expect(ValueError, validate_artifact, corrupt)
    corrupt = copy.deepcopy(artifact)
    corrupt["tensors"]["trigger_step"][0] = corrupt["tensors"]["episode_length"][0]
    _expect(ValueError, validate_artifact, corrupt)
    corrupt = copy.deepcopy(artifact)
    corrupt["tensors"]["episode_length"][2] = MAX_EPISODE_ACTIONS - 1
    _expect(ValueError, validate_artifact, corrupt)
    corrupt = copy.deepcopy(artifact)
    corrupt["tensors"]["ever_clearance_ge_20cm"][0] = False
    _expect(ValueError, validate_artifact, corrupt)
    corrupt = copy.deepcopy(artifact)
    corrupt["tensors"]["ever_grasped"][0] = False
    _expect(ValueError, validate_artifact, corrupt)
    corrupt = copy.deepcopy(artifact)
    corrupt["tensors"]["max_true_clearance_m"][1] = 0.049
    _expect(ValueError, validate_artifact, corrupt)

    corrupt = copy.deepcopy(artifact)
    corrupt["metadata"]["extra"] = 1
    _expect(ValueError, validate_artifact, corrupt)
    corrupt = copy.deepcopy(artifact)
    del corrupt["metadata"]["source_sha256"][next(iter(corrupt["metadata"]["source_sha256"]))]
    _expect(ValueError, validate_artifact, corrupt)
    corrupt = copy.deepcopy(artifact)
    del corrupt["metadata"]["runtime_asset_sha256"][RUNTIME_ASSET_NAMES[0]]
    _expect(ValueError, validate_artifact, corrupt)
    corrupt = copy.deepcopy(artifact)
    del corrupt["metadata"]["runtime"]
    _expect(ValueError, validate_artifact, corrupt)
    corrupt = copy.deepcopy(artifact)
    corrupt["metadata"]["runtime"]["cudnn"] = float("nan")
    _expect(ValueError, validate_artifact, corrupt)
    corrupt = copy.deepcopy(artifact)
    corrupt["metadata"]["runtime"]["platform"] = object()
    _expect(TypeError, validate_artifact, corrupt)


def test_report_is_exactly_bound_to_artifact() -> None:
    artifact = build_artifact(metadata=_metadata(), tensors=_tensors())
    report = _report(artifact)
    validate_report(report, artifact)
    for key, value in (
        ("seed", 291),
        ("vector_steps", MAX_EPISODE_ACTIONS - 1),
        ("baseline_actor_sha256", "f" * 64),
        ("summary", {}),
    ):
        corrupt = copy.deepcopy(report)
        corrupt[key] = value
        _expect(ValueError, validate_report, corrupt, artifact)
    corrupt = copy.deepcopy(report)
    corrupt["extra"] = True
    _expect(ValueError, validate_report, corrupt, artifact)
    corrupt = copy.deepcopy(report)
    del corrupt["summary"]
    _expect(ValueError, validate_report, corrupt, artifact)


def test_publication_is_weights_only_and_no_clobber() -> None:
    artifact = build_artifact(metadata=_metadata(), tensors=_tensors())
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact_path = root / "trial.pt"
        report_path = root / "trial.json"
        digest = publish_artifact_and_report_no_clobber(
            artifact,
            _report(artifact),
            artifact_output=artifact_path,
            report_output=report_path,
        )
        assert len(digest) == 64
        loaded = torch.load(artifact_path, map_location="cpu", weights_only=True)
        validate_artifact(loaded)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["artifact_sha256"] == digest
        _expect(
            FileExistsError,
            publish_artifact_and_report_no_clobber,
            artifact,
            _report(artifact),
            artifact_output=artifact_path,
            report_output=report_path,
        )


def test_publication_rolls_back_on_base_exception() -> None:
    class InjectedAbort(BaseException):
        pass

    artifact = build_artifact(metadata=_metadata(), tensors=_tensors())
    report = _report(artifact)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact_path = root / "trial.pt"
        report_path = root / "trial.json"
        real_link = ab_contract.os.link
        calls = 0

        def abort_after_link(source, destination):
            nonlocal calls
            calls += 1
            real_link(source, destination)
            if calls == 2:
                raise InjectedAbort()

        ab_contract.os.link = abort_after_link
        try:
            _expect(
                InjectedAbort,
                publish_artifact_and_report_no_clobber,
                artifact,
                report,
                artifact_output=artifact_path,
                report_output=report_path,
            )
        finally:
            ab_contract.os.link = real_link
        assert not artifact_path.exists()
        assert not report_path.exists()
        assert list(root.iterdir()) == []


def test_json_publication_is_transactional() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "failure.json"
        _expect(TypeError, publish_json_no_clobber, {"bad": object()}, output)
        assert not output.exists()

        class InjectedAbort(BaseException):
            pass

        real_link = ab_contract.os.link

        def abort_after_link(source, destination):
            real_link(source, destination)
            raise InjectedAbort()

        ab_contract.os.link = abort_after_link
        try:
            _expect(InjectedAbort, publish_json_no_clobber, {"ok": True}, output)
        finally:
            ab_contract.os.link = real_link
        assert not output.exists()
        assert list(root.iterdir()) == []


def test_collection_spec_freezes_assignment_before_launcher() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        baseline = root / "baseline"
        candidate = root / "candidate"
        baseline.mkdir()
        candidate.mkdir()
        common = {
            "task_contract.json": b"same-task",
            "frozen_lift_actor.pt": b"same-lift",
        }
        for checkpoint, actor, bridge in (
            (baseline, b"base-actor", b"base-bridge"),
            (candidate, b"candidate-actor", b"candidate-bridge"),
        ):
            payloads = {"actor.pt": actor, "torch_bridge_state.pt": bridge, **common}
            assert set(payloads) == set(CHECKPOINT_FILES)
            for name, payload in payloads.items():
                (checkpoint / name).write_bytes(payload)
        search = root / "search.pth"
        search.write_bytes(b"search")
        args = argparse.Namespace(
            baseline_checkpoint=baseline,
            candidate_checkpoint=candidate,
            search_checkpoint=search,
            seed=290,
            replicate="a",
            num_envs=8,
            output_stem=root / "trial",
        )
        spec = build_spec(args)
        assert spec.artifact_output == root / "trial.pt"
        assert spec.report_output == root / "trial.json"
        assert torch.equal(
            spec.assignment_candidate,
            assignment_candidate_mask(seed=290, num_envs=8, replicate="a"),
        )
        (candidate / "frozen_lift_actor.pt").write_bytes(b"changed-lift")
        _expect(ValueError, build_spec, args)


def test_failure_attempts_leave_canonical_outputs_retryable() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        baseline = root / "baseline"
        candidate = root / "candidate"
        baseline.mkdir()
        candidate.mkdir()
        for checkpoint, actor in (
            (baseline, b"base-actor"),
            (candidate, b"candidate-actor"),
        ):
            for name, payload in {
                "actor.pt": actor,
                "task_contract.json": b"same-task",
                "frozen_lift_actor.pt": b"same-lift",
                "torch_bridge_state.pt": actor + b"-bridge",
            }.items():
                (checkpoint / name).write_bytes(payload)
        search = root / "search.pth"
        search.write_bytes(b"search")
        spec = build_spec(
            argparse.Namespace(
                baseline_checkpoint=baseline,
                candidate_checkpoint=candidate,
                search_checkpoint=search,
                seed=290,
                replicate="a",
                num_envs=8,
                output_stem=root / "trial",
            )
        )
        try:
            raise RuntimeError("first")
        except RuntimeError as error:
            first = publish_failure_attempt(spec, error)
        try:
            raise RuntimeError("second")
        except RuntimeError as error:
            second = publish_failure_attempt(spec, error)
        assert first.name == "trial.failed_attempt_001.json"
        assert second.name == "trial.failed_attempt_002.json"
        assert not spec.artifact_output.exists()
        assert not spec.report_output.exists()


if __name__ == "__main__":
    test_assignment_is_balanced_stable_and_complementary()
    test_action_selection_has_no_pretreatment_leak_and_common_lift()
    test_artifact_validation_summary_and_corruption_rejection()
    test_report_is_exactly_bound_to_artifact()
    test_publication_is_weights_only_and_no_clobber()
    test_publication_rolls_back_on_base_exception()
    test_json_publication_is_transactional()
    test_collection_spec_freezes_assignment_before_launcher()
    test_failure_attempts_leave_canonical_outputs_retryable()
    print("online_close_ab_test: PASS")
