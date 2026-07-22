#!/usr/bin/env python3
"""Focused CPU-only tests for the Candidate41 collector authority boundary."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
from unittest import mock

import torch

import candidate41_public_arm_ramp_episode as contract
import collect_candidate41_public_arm_ramp_ab as collector


def _raises(error: type[BaseException], fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _arguments(stem: Path, *, seed: int, replicate: str, num_envs: int) -> dict:
    return {
        "device": "cuda:0",
        "enable_cameras": False,
        "experience": "",
        "fixed_direction": str((_root() / contract.FIXED_DIRECTION_PATH).resolve()),
        "headless": True,
        "kit_args": contract.KIT_ARGS,
        "livestream": -1,
        "num_envs": num_envs,
        "output_stem": str(stem.resolve()),
        "pre_tanh_l2_cap": contract.PRE_TANH_L2_CAP,
        "raw_z_abs_cap": contract.RAW_Z_ABS_CAP,
        "replicate": replicate,
        "search_checkpoint": str(
            (_root() / contract.SEARCH_CHECKPOINT_PATH).resolve()
        ),
        "seed": seed,
        "token_component_cap": contract.TOKEN_COMPONENT_CAP,
        "token_scale": contract.TOKEN_SCALE,
        "distal_component_cap": contract.DISTAL_COMPONENT_CAP,
        "distal_scale": contract.DISTAL_SCALE,
        "v6_checkpoint": str((_root() / contract.V6_CHECKPOINT_PATH).resolve()),
        "verify_steps": 15,
        "window_steps": contract.WINDOW_STEPS,
    }


def _checkpoints(*, include_smoke: bool) -> dict:
    result = {
        "v6": {"actor.pt": "a" * 64},
        "search": "b" * 64,
        "fixed_direction": "c" * 64,
    }
    if include_smoke:
        result[contract.SMOKE_ARTIFACT_PATH] = "d" * 64
        result[contract.SMOKE_REPORT_PATH] = "e" * 64
    return result


def _receipt(
    stem: Path,
    *,
    seed: int,
    replicate: str,
    num_envs: int,
    attempt: int,
    first_step: bool,
    collection_commit: str,
    source_manifest_sha256: str,
    checkpoint_manifest_sha256: str,
    argument_receipt: dict,
) -> dict:
    assignment = collector.verifier.exact_balanced_treatment_mask(
        seed=seed, num_envs=num_envs, replicate=replicate
    )
    return {
        "kind": collector.ATTEMPT_RECEIPT_KIND,
        "status": "failed",
        "attempt": attempt,
        "first_env_step_invoked": first_step,
        "retry_permitted": not first_step,
        "seed": seed,
        "replicate": replicate,
        "num_envs": num_envs,
        "canonical_artifact_output": str(Path(f"{stem}.pt")),
        "canonical_report_output": str(Path(f"{stem}.json")),
        "collection_commit": collection_commit,
        "assignment_mask_sha256": collector.verifier.assignment_mask_sha256(
            assignment
        ),
        "source_manifest_sha256": source_manifest_sha256,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "argument_receipt": dict(argument_receipt),
        "error_type": "RuntimeError",
        "error": "synthetic CPU-only failure",
        "traceback": "synthetic traceback",
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _validate_attempt(
    root: Path,
    *,
    seed: int,
    replicate: str,
    num_envs: int,
    checkpoints: dict,
    arguments: dict,
    collection_commit: str = "f" * 40,
    source_manifest_sha256: str = "9" * 64,
) -> None:
    treatment = collector.verifier.exact_balanced_treatment_mask(
        seed=seed, num_envs=num_envs, replicate=replicate
    )
    collector._validate_exact_invocation_attempt(
        root,
        seed=seed,
        replicate=replicate,
        num_envs=num_envs,
        collection_commit=collection_commit,
        assignment_mask_sha256=collector.verifier.assignment_mask_sha256(
            treatment
        ),
        source_manifest_sha256=source_manifest_sha256,
        checkpoint_sha256=checkpoints,
        argument_receipt=arguments,
    )


def _finalized_plan(pristine: dict) -> dict:
    result = copy.deepcopy(pristine)
    result["status"] = contract.SEALED_PLAN_STATUS
    result["preregistration_receipt"]["commit"] = "a" * 40
    result["preregistration_receipt"]["sha256"] = "b" * 64
    result["implementation_seal"] = {
        "status": "complete_without_simulator_evidence",
        "implementation_commit": "c" * 40,
        "source_sha256": {"source.py": "d" * 64},
        "simulator_evidence_before_source_seal": False,
        "simulation_free_tests": {
            "status": "passed",
            "passed": 1,
            "failed": 0,
            "required_contracts_covered": copy.deepcopy(
                pristine["implementation_seal_requirements"][
                    "required_simulation_free_tests"
                ]
            ),
        },
        "static_audit": {
            "status": "passed",
            "blockers": 0,
            "simulator_invocations": 0,
            "runtime_assets": 8,
            "runtime_source_files": 1,
        },
    }
    result["collection_seal"] = {
        "tag": collector.COLLECTION_SEAL_TAG,
        "status": "sealed_before_smoke_and_collection",
    }
    return result


def test_source_closure_and_exact_five_invocations_are_explicit() -> None:
    assert collector.CANDIDATE41_INVOCATION_SEQUENCE == (
        (340, "b", 8),
        (341, "a", 64),
        (341, "b", 64),
        (342, "b", 64),
        (342, "a", 64),
    )
    assert collector.CANDIDATE41_RUN_ORDER == (
        (341, "a"),
        (341, "b"),
        (342, "b"),
        (342, "a"),
    )
    required = {
        *contract.IMPLEMENTATION_SOURCE_FILES,
        "scripts/flashsac/collect_candidate39_episode_residual_ab.py",
        "scripts/flashsac/candidate39_episode_residual.py",
        "scripts/flashsac/candidate40_verified_arm_handoff.py",
        "scripts/flashsac/candidate40_verified_arm_episode.py",
        "scripts/flashsac/option_residual_screen.py",
        contract.VALIDATION_PLAN,
    }
    assert required <= set(collector.EXTRA_SOURCE_FILES)
    dynamic = set(collector.DYNAMIC_IMPORT_PACKAGE_SOURCES)
    critical_dynamic = {
        "source/xhand_inhand/xhand_inhand/robots/__init__.py",
        "source/xhand_inhand/xhand_inhand/robots/fr3.py",
        "source/xhand_inhand/xhand_inhand/robots/fr3_xhand.py",
        "source/xhand_inhand/xhand_inhand/robots/xhand.py",
        "source/xhand_inhand/xhand_inhand/tasks/__init__.py",
        "source/xhand_inhand/xhand_inhand/tasks/direct/functional_grasping/__init__.py",
        "source/xhand_inhand/xhand_inhand/tasks/direct/pick_cube/agents/__init__.py",
        "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/agents/__init__.py",
        "source/xhand_inhand/xhand_inhand/foundationpose_cube.py",
        "source/xhand_inhand/xhand_inhand/utils/friction.py",
        "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/agents/rl_games_ppo_cfg.yaml",
    }
    assert critical_dynamic <= dynamic <= set(collector.EXTRA_SOURCE_FILES)
    assert all("/.claude/" not in f"/{path}" for path in dynamic)
    assert all("/__pycache__/" not in f"/{path}" for path in dynamic)
    for relative in dynamic:
        path = _root() / relative
        assert path.is_file() and not path.is_symlink()


def test_independent_float32_arm_clock_reconstructs_treatment_and_control() -> None:
    common = torch.linspace(-0.8, 0.8, 14, dtype=torch.float32).reshape(2, 7)
    treatment = torch.tensor([True, False])
    active = torch.ones(2, dtype=torch.bool)
    denominator = torch.tensor(15.0, dtype=torch.float32)
    for count in range(16):
        before = torch.full((2,), count, dtype=torch.long)
        audit = collector.reconstruct_public_arm_ramp(
            common_arm=common,
            stable_current=active,
            stable_count_before=before,
            treatment=treatment,
            activated_this_action=active,
            option_active=active,
            episode_active=active,
        )
        expected = torch.tensor(count, dtype=torch.long).to(torch.float32) / denominator
        assert audit.ramp_scale.dtype == torch.float32
        assert torch.equal(audit.ramp_scale, expected.repeat(2))
        assert torch.equal(audit.authority_scale, torch.tensor([expected, 1.0]))
        assert torch.equal(audit.requested_arm[0], common[0] * expected)
        assert torch.equal(audit.requested_arm[1], common[1])
        assert torch.equal(
            audit.stable_count_after,
            torch.full((2,), min(count + 1, 15), dtype=torch.long),
        )

    unstable = collector.reconstruct_public_arm_ramp(
        common_arm=common,
        stable_current=torch.zeros(2, dtype=torch.bool),
        stable_count_before=torch.tensor([8, 8]),
        treatment=treatment,
        activated_this_action=active,
        option_active=active,
        episode_active=active,
    )
    assert torch.equal(unstable.authority_scale, torch.tensor([0.0, 1.0]))
    assert torch.equal(unstable.stable_count_after, torch.zeros(2, dtype=torch.long))
    search = collector.reconstruct_public_arm_ramp(
        common_arm=common,
        stable_current=active,
        stable_count_before=torch.tensor([8, 8]),
        treatment=treatment,
        activated_this_action=active,
        option_active=torch.zeros(2, dtype=torch.bool),
        episode_active=active,
    )
    assert torch.equal(search.authority_scale, torch.ones(2))
    assert torch.equal(search.requested_arm, common)


def test_argument_receipt_covers_app_launcher_arguments() -> None:
    with tempfile.TemporaryDirectory() as name:
        stem = Path(name) / "trial"
        values = _arguments(stem, seed=340, replicate="b", num_envs=8)
        namespace = argparse.Namespace(**values)
        receipt = collector._argument_receipt(namespace, stem)
        assert receipt == values
        changed = copy.copy(namespace)
        changed.device = "cuda:1"
        assert collector._argument_receipt(changed, stem) != receipt
        prior = collector._argument_receipt_for_identity(
            receipt,
            seed=341,
            replicate="a",
            num_envs=64,
            output_stem=Path(name) / "dev",
        )
        assert prior["device"] == "cuda:0"
        assert (prior["seed"], prior["replicate"], prior["num_envs"]) == (
            341,
            "a",
            64,
        )


def test_trigger_witness_boundary_separates_pre_and_first_post_row() -> None:
    # Row 0 triggers strictly before eligibility and must carry the explicit
    # pre marker.  Row 1 triggers on first eligibility and is witnessed by the
    # first post trace row, so it must not be duplicated in the pre table.
    selected = collector.pre_trace_record_mask(
        preeligible=torch.tensor([True, False, True]),
        option_active=torch.tensor([True, True, False]),
        trigger=torch.tensor([True, True, False]),
        newly_latched=torch.tensor([False, False, True]),
    )
    assert torch.equal(selected, torch.tensor([True, False, True]))
    assert "pre_trigger_witness" in contract.PRE_STEP_FIELDS


def test_pre_step_retry_is_bound_and_post_step_failure_is_authoritative() -> None:
    collection_commit = "f" * 40
    source_sha = "9" * 64
    checkpoints = _checkpoints(include_smoke=False)
    checkpoint_sha = collector._json_sha256(checkpoints)
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        stem = collector._canonical_invocation_stem(root, 340, "b", 8)
        arguments = _arguments(stem, seed=340, replicate="b", num_envs=8)
        _validate_attempt(
            root,
            seed=340,
            replicate="b",
            num_envs=8,
            checkpoints=checkpoints,
            arguments=arguments,
        )
        receipt = _receipt(
            stem,
            seed=340,
            replicate="b",
            num_envs=8,
            attempt=1,
            first_step=False,
            collection_commit=collection_commit,
            source_manifest_sha256=source_sha,
            checkpoint_manifest_sha256=checkpoint_sha,
            argument_receipt=arguments,
        )
        path = Path(f"{stem}.failed_attempt_001.json")
        _write_json(path, receipt)
        original = path.read_bytes()
        _validate_attempt(
            root,
            seed=340,
            replicate="b",
            num_envs=8,
            checkpoints=checkpoints,
            arguments=arguments,
        )
        assert path.read_bytes() == original
        changed = dict(arguments)
        changed["device"] = "cuda:1"
        _raises(
            ValueError,
            _validate_attempt,
            root,
            seed=340,
            replicate="b",
            num_envs=8,
            checkpoints=checkpoints,
            arguments=changed,
        )

    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        stem = collector._canonical_invocation_stem(root, 340, "b", 8)
        arguments = _arguments(stem, seed=340, replicate="b", num_envs=8)
        receipt = _receipt(
            stem,
            seed=340,
            replicate="b",
            num_envs=8,
            attempt=1,
            first_step=True,
            collection_commit=collection_commit,
            source_manifest_sha256=source_sha,
            checkpoint_manifest_sha256=checkpoint_sha,
            argument_receipt=arguments,
        )
        _write_json(Path(f"{stem}.failed_attempt_001.json"), receipt)
        _raises(
            ValueError,
            _validate_attempt,
            root,
            seed=340,
            replicate="b",
            num_envs=8,
            checkpoints=checkpoints,
            arguments=arguments,
        )


def test_order_rejects_later_namespace_and_authenticates_prior_receipts() -> None:
    collection_commit = "f" * 40
    source_sha = "9" * 64
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        current = collector._canonical_invocation_stem(root, 340, "b", 8)
        later = collector._canonical_invocation_stem(root, 341, "a", 64)
        arguments = _arguments(current, seed=340, replicate="b", num_envs=8)
        _write_json(Path(f"{later}.failed_attempt_001.json"), {})
        _raises(
            ValueError,
            _validate_attempt,
            root,
            seed=340,
            replicate="b",
            num_envs=8,
            checkpoints=_checkpoints(include_smoke=False),
            arguments=arguments,
        )

    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        smoke = collector._canonical_invocation_stem(root, 340, "b", 8)
        current = collector._canonical_invocation_stem(root, 341, "a", 64)
        Path(f"{smoke}.pt").parent.mkdir(parents=True, exist_ok=True)
        Path(f"{smoke}.pt").write_bytes(b"sealed smoke placeholder")
        Path(f"{smoke}.json").write_bytes(b"sealed smoke report placeholder")
        checkpoints = _checkpoints(include_smoke=True)
        smoke_checkpoints = dict(checkpoints)
        smoke_checkpoints.pop(contract.SMOKE_ARTIFACT_PATH)
        smoke_checkpoints.pop(contract.SMOKE_REPORT_PATH)
        current_arguments = _arguments(
            current, seed=341, replicate="a", num_envs=64
        )
        smoke_arguments = collector._argument_receipt_for_identity(
            current_arguments,
            seed=340,
            replicate="b",
            num_envs=8,
            output_stem=smoke,
        )
        receipt = _receipt(
            smoke,
            seed=340,
            replicate="b",
            num_envs=8,
            attempt=1,
            first_step=False,
            collection_commit=collection_commit,
            source_manifest_sha256=source_sha,
            checkpoint_manifest_sha256=collector._json_sha256(smoke_checkpoints),
            argument_receipt=smoke_arguments,
        )
        receipt_path = Path(f"{smoke}.failed_attempt_001.json")
        _write_json(receipt_path, receipt)
        _validate_attempt(
            root,
            seed=341,
            replicate="a",
            num_envs=64,
            checkpoints=checkpoints,
            arguments=current_arguments,
        )
        receipt["source_manifest_sha256"] = "8" * 64
        _write_json(receipt_path, receipt)
        _raises(
            ValueError,
            _validate_attempt,
            root,
            seed=341,
            replicate="a",
            num_envs=64,
            checkpoints=checkpoints,
            arguments=current_arguments,
        )


def test_preregistration_protected_fields_allow_only_seal_receipts() -> None:
    pristine = json.loads(
        (_root() / contract.VALIDATION_PLAN).read_text(encoding="utf-8")
    )
    final = _finalized_plan(pristine)
    collector.validate_preregistration_protected_fields(pristine, final)

    changed = copy.deepcopy(final)
    changed["execution"]["run_order"][0] = "342a"
    _raises(
        ValueError,
        collector.validate_preregistration_protected_fields,
        pristine,
        changed,
    )
    changed = copy.deepcopy(final)
    changed["preregistration_receipt"]["tag"] = "different-tag"
    _raises(
        ValueError,
        collector.validate_preregistration_protected_fields,
        pristine,
        changed,
    )


def test_static_audit_source_count_binds_prelaunch_closure() -> None:
    pristine = json.loads(
        (_root() / contract.VALIDATION_PLAN).read_text(encoding="utf-8")
    )
    sealed = _finalized_plan(pristine)
    collector.validate_runtime_source_count(sealed, {"source.py": "d" * 64})

    sealed["implementation_seal"]["static_audit"][
        "runtime_source_files"
    ] = 2
    _raises(
        ValueError,
        collector.validate_runtime_source_count,
        sealed,
        {"source.py": "d" * 64},
    )
    sealed["implementation_seal"]["static_audit"][
        "runtime_source_files"
    ] = True
    _raises(
        ValueError,
        collector.validate_runtime_source_count,
        sealed,
        {"source.py": "d" * 64},
    )


def test_exact_collection_head_checks_tag_and_protected_plan() -> None:
    pristine = json.loads(
        (_root() / contract.VALIDATION_PLAN).read_text(encoding="utf-8")
    )
    final = _finalized_plan(pristine)
    head = "c" * 40
    prereg = "a" * 40
    final["preregistration_receipt"]["commit"] = prereg
    raw = json.dumps(pristine, sort_keys=True).encode("utf-8")

    def resolve(_root_path: Path, revision: str) -> str:
        return {
            "HEAD": head,
            collector.COLLECTION_SEAL_TAG: head,
            collector.PREREGISTRATION_TAG: prereg,
        }[revision]

    with mock.patch.object(collector, "_git_resolve", side_effect=resolve), mock.patch.object(
        collector, "_git_blob_sha256", return_value="b" * 64
    ), mock.patch.object(
        collector, "_git_blob_bytes", return_value=raw
    ), mock.patch.object(
        collector, "require_git_ancestor"
    ) as ancestor:
        assert collector.require_exact_collection_head(_root(), final) == head
    ancestor.assert_called_once_with(_root(), prereg, head)

    changed = copy.deepcopy(final)
    changed["execution"]["run_order"] = list(reversed(changed["execution"]["run_order"]))
    with mock.patch.object(collector, "_git_resolve", side_effect=resolve), mock.patch.object(
        collector, "_git_blob_sha256", return_value="b" * 64
    ), mock.patch.object(
        collector, "_git_blob_bytes", return_value=raw
    ), mock.patch.object(collector, "require_git_ancestor"):
        _raises(
            ValueError,
            collector.require_exact_collection_head,
            _root(),
            changed,
        )


def test_collection_seal_diff_changes_only_final_plan() -> None:
    valid = mock.Mock(
        returncode=0,
        stdout=f"{contract.VALIDATION_PLAN}\n",
        stderr="",
    )
    with mock.patch.object(collector.subprocess, "run", return_value=valid):
        collector.validate_collection_seal_name_only_diff(
            _root(),
            implementation_commit="a" * 40,
            collection_commit="b" * 40,
        )
    changed = mock.Mock(
        returncode=0,
        stdout=(
            f"{contract.VALIDATION_PLAN}\n"
            "scripts/flashsac/collect_candidate41_public_arm_ramp_ab.py\n"
        ),
        stderr="",
    )
    with mock.patch.object(collector.subprocess, "run", return_value=changed):
        _raises(
            ValueError,
            collector.validate_collection_seal_name_only_diff,
            _root(),
            implementation_commit="a" * 40,
            collection_commit="b" * 40,
        )


def test_post_step_application_close_failure_cannot_publish_success() -> None:
    class FailingApp:
        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1
            raise RuntimeError("synthetic application close failure")

    app = FailingApp()

    class Launcher:
        def __init__(self, _args: argparse.Namespace) -> None:
            self.app = app

    spec = mock.Mock(
        seed=340,
        replicate="b",
        num_envs=8,
        artifact_output=Path("never-published.pt"),
        report_output=Path("never-published.json"),
    )

    def collected(_spec, *, device_string: str, progress) -> tuple[dict, dict]:
        assert device_string == "cuda:0"
        progress.first_env_step_invoked = True
        return {}, {"summary": {"trace_rows": 0}}

    with mock.patch.object(
        collector,
        "parse_args",
        return_value=(argparse.Namespace(device="cuda:0"), spec, Launcher),
    ), mock.patch.object(
        collector, "run_collection", side_effect=collected
    ), mock.patch.object(
        collector.artifact_contract, "publish_artifact_and_report_no_clobber"
    ) as publish_success, mock.patch.object(
        collector,
        "publish_failure_attempt",
        return_value=Path("failure-receipt.json"),
    ) as publish_failure:
        _raises(RuntimeError, collector.main)
    assert app.close_count == 1
    publish_success.assert_not_called()
    assert publish_failure.call_count == 1
    assert publish_failure.call_args.args[2].first_env_step_invoked is True


def test_stdout_failure_after_durable_pair_does_not_create_failure_receipt() -> None:
    events: list[str] = []

    class App:
        def close(self) -> None:
            events.append("close")

    class Launcher:
        def __init__(self, _args: argparse.Namespace) -> None:
            self.app = App()

    spec = mock.Mock(
        seed=340,
        replicate="b",
        num_envs=8,
        artifact_output=Path("durable.pt"),
        report_output=Path("durable.json"),
    )

    def collected(_spec, *, device_string: str, progress) -> tuple[dict, dict]:
        progress.first_env_step_invoked = True
        events.append("collect")
        return {}, {"summary": {"trace_rows": 0}}

    def post_authority(_spec) -> None:
        events.append("post-authority")

    def published(*_args, **_kwargs) -> str:
        events.append("publish")
        return "a" * 64

    with mock.patch.object(
        collector,
        "parse_args",
        return_value=(argparse.Namespace(device="cuda:0"), spec, Launcher),
    ), mock.patch.object(
        collector, "run_collection", side_effect=collected
    ), mock.patch.object(
        collector, "validate_post_application_authority", side_effect=post_authority
    ), mock.patch.object(
        collector.artifact_contract,
        "publish_artifact_and_report_no_clobber",
        side_effect=published,
    ), mock.patch.object(
        collector, "publish_failure_attempt"
    ) as publish_failure, mock.patch(
        "builtins.print", side_effect=BrokenPipeError("synthetic stdout failure")
    ):
        _raises(BrokenPipeError, collector.main)
    assert events == ["collect", "close", "post-authority", "publish"]
    publish_failure.assert_not_called()


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    test_source_closure_and_exact_five_invocations_are_explicit()
    test_independent_float32_arm_clock_reconstructs_treatment_and_control()
    test_argument_receipt_covers_app_launcher_arguments()
    test_trigger_witness_boundary_separates_pre_and_first_post_row()
    test_pre_step_retry_is_bound_and_post_step_failure_is_authoritative()
    test_order_rejects_later_namespace_and_authenticates_prior_receipts()
    test_preregistration_protected_fields_allow_only_seal_receipts()
    test_static_audit_source_count_binds_prelaunch_closure()
    test_exact_collection_head_checks_tag_and_protected_plan()
    test_collection_seal_diff_changes_only_final_plan()
    test_post_step_application_close_failure_cannot_publish_success()
    test_stdout_failure_after_durable_pair_does_not_create_failure_receipt()
    print("collect_candidate41_public_arm_ramp_ab tests passed")


if __name__ == "__main__":
    main()
