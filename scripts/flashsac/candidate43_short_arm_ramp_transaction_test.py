#!/usr/bin/env python3
"""CPU campaign-binding tests for Candidate43's inherited C42 transaction.

The transaction implementation and its fault-injection suite remain owned by
``candidate42_attempt_transaction``.  These tests prove that the Candidate43
parent, child, and collector are wired to that exact implementation while the
scientific campaign bindings (D8, seeds, assignment, and output stems) are
independent.  No test imports IsaacLab or starts a simulator.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import torch

import candidate42_attempt_transaction as transaction
import candidate42_public_arm_ramp_collection as c42_collection
import collect_candidate42_public_arm_ramp_ab as c42_parent
import collect_candidate42_public_arm_ramp_child as c42_child
import candidate43_short_arm_ramp as supervisor
import candidate43_short_arm_ramp_collection as collection
import candidate43_short_arm_ramp_episode as artifact_contract
import collect_candidate43_short_arm_ramp_ab as parent
import collect_candidate43_short_arm_ramp_child as child


ROOT = Path(__file__).resolve().parents[2]
C42_PLAN = ROOT / "scripts/flashsac/candidate42_transactional_arm_ramp_development_plan.json"
C43_PLAN = ROOT / "scripts/flashsac/candidate43_short_arm_ramp_development_plan.json"
TRANSACTION_PROTOCOL_SHA256 = (
    "8b6a3fca3ab44eb24e8e5ce8ad648ac5fe9742b585a6c5d2cbda97f6eddb54a4"
)


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    encoded = (encoded + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _function_node(function: object) -> ast.FunctionDef:
    module = ast.parse(inspect.getsource(function))
    node = module.body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


class _CampaignStringNormalizer(ast.NodeTransformer):
    """Normalize only registered campaign labels, never executable algebra."""

    _exact = {
        "collect_candidate42_public_arm_ramp_child.py": "<campaign-child>",
        "collect_candidate43_short_arm_ramp_child.py": "<campaign-child>",
    }

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if not isinstance(node.value, str):
            return node
        value = self._exact.get(node.value, node.value)
        value = value.replace("Candidate42", "CandidateXX")
        value = value.replace("Candidate43", "CandidateXX")
        value = value.replace("candidate42", "candidateXX")
        value = value.replace("candidate43", "candidateXX")
        return ast.copy_location(ast.Constant(value=value), node)


def _normalized_function(function: object) -> str:
    node = _CampaignStringNormalizer().visit(_function_node(function))
    return ast.dump(ast.fix_missing_locations(node), include_attributes=False)


def _normalized_module(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree = _CampaignStringNormalizer().visit(tree)
    assert isinstance(tree, ast.Module)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {
                    "candidate42_public_arm_ramp_collection",
                    "candidate43_short_arm_ramp_collection",
                }:
                    alias.name = "candidateXX_campaign_collection"
                elif alias.name in {
                    "candidate42_public_arm_ramp_episode",
                    "candidate43_short_arm_ramp_episode",
                }:
                    alias.name = "candidateXX_campaign_episode"
    return ast.dump(ast.fix_missing_locations(tree), include_attributes=False)


def test_exact_c42_transaction_module_is_reused_everywhere() -> None:
    assert collection.transaction is transaction
    assert parent.transaction is transaction
    assert child.transaction is transaction
    assert transaction.FINAL_REPORT_KIND == (
        "pick_tool_candidate42_transactional_public_arm_ramp_final_report_v1"
    )
    assert artifact_contract.FINAL_REPORT_KIND == transaction.FINAL_REPORT_KIND


def test_transaction_protocol_object_is_byte_semantically_identical() -> None:
    c42 = json.loads(C42_PLAN.read_text(encoding="utf-8"))
    c43 = json.loads(C43_PLAN.read_text(encoding="utf-8"))
    assert c43["transaction_protocol"] == c42["transaction_protocol"]
    assert _canonical_json_sha256(c43["transaction_protocol"]) == (
        TRANSACTION_PROTOCOL_SHA256
    )


def test_inherited_namespace_suffix_and_attempt_path_are_exact(tmp_path: Path) -> None:
    stem = (tmp_path / "campaign" / "trial").resolve()
    paths = transaction.TransactionPaths.from_output_stem(stem)
    assert paths.namespace == Path(f"{stem}.c42_txn")
    assert parent._attempt_directory(stem, 7) == (
        Path(f"{stem}.c42_txn") / "attempt_007"
    )


def test_parent_import_graph_is_non_isaac_and_child_owns_delayed_import() -> None:
    parent_source = Path(parent.__file__).read_text(encoding="utf-8")
    parent_tree = ast.parse(parent_source)
    imported = {
        alias.name
        for node in ast.walk(parent_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(parent_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(name == "isaaclab" or name.startswith("isaaclab.") for name in imported)
    assert "isaaclab.app" not in imported

    full_arguments = _function_node(child._full_arguments)
    delayed = [
        node for node in ast.walk(full_arguments)
        if isinstance(node, ast.ImportFrom) and node.module == "isaaclab.app"
    ]
    assert len(delayed) == 1


def test_child_and_parent_transaction_code_remain_normalized_c42_equivalent() -> None:
    c42_child_path = Path(c42_child.__file__)
    c43_child_path = Path(child.__file__)
    assert _normalized_module(c43_child_path) == _normalized_module(c42_child_path)

    # The parent differs only in its campaign imports, child basename, and
    # user-facing labels.  Its namespace, markers, recovery, and publication
    # code must remain identical.
    assert _normalized_module(Path(parent.__file__)) == _normalized_module(
        Path(c42_parent.__file__)
    )


def test_rollout_kernel_is_normalized_c42_equivalent() -> None:
    assert _normalized_function(collection.run_collection) == _normalized_function(
        c42_collection.run_collection
    )


def test_candidate43_registered_sequence_and_stems_are_independent() -> None:
    assert collection.CANDIDATE43_INVOCATION_SEQUENCE == (
        (352, "b", 8),
        (353, "a", 64),
        (353, "b", 64),
        (354, "b", 64),
        (354, "a", 64),
    )
    assert collection.CANDIDATE43_RUN_ORDER == (
        (353, "a"),
        (353, "b"),
        (354, "b"),
        (354, "a"),
    )
    assert collection._canonical_dev_stem(ROOT, 353, "a") == (
        ROOT / "logs/flashsac/pick_tool/55_c43_short_arm_ramp_dev_s353_a/trial"
    ).resolve()
    smoke = collection._canonical_invocation_stem(ROOT, 352, "b", 8)
    assert smoke == (ROOT / artifact_contract.SMOKE_ARTIFACT_PATH).with_suffix("").resolve()


def test_c42_causal_bytes_and_annotated_tag_are_live_authority() -> None:
    hashes = collection._validate_causal_receipts(ROOT)
    assert hashes[artifact_contract.CANDIDATE42_PLAN] == (
        artifact_contract.CANDIDATE42_PLAN_SHA256
    )
    assert hashes[artifact_contract.CANDIDATE42_RESULT] == (
        artifact_contract.CANDIDATE42_RESULT_SHA256
    )
    assert hashes[artifact_contract.CANDIDATE42_REPORT] == (
        artifact_contract.CANDIDATE42_REPORT_SHA256
    )


def test_child_arguments_bind_c43_campaign_but_keep_inherited_private_contract(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        v6_checkpoint=tmp_path / "v6",
        search_checkpoint=tmp_path / "search.pth",
        fixed_direction=tmp_path / "fixed.pt",
        seed=352,
        replicate="b",
        num_envs=8,
        window_steps=24,
        token_scale=0.1,
        distal_scale=0.2,
        raw_z_abs_cap=3.0,
        token_component_cap=0.2,
        distal_component_cap=0.3,
        pre_tanh_l2_cap=1.0,
        verify_steps=8,
        output_stem=tmp_path / "trial",
        device="cuda:0",
        kit_args="--/app/extensions/fsWatcherEnabled=false",
        headless=True,
        enable_cameras=False,
        livestream=-1,
        experience="",
    )
    identity = transaction.AttemptIdentity(
        run_id="a" * 64, attempt_id="b" * 64, attempt_number=3
    )
    argv = parent._child_arguments(args, identity, 17)
    assert Path(argv[1]).name == "collect_candidate43_short_arm_ramp_child.py"
    assert argv[argv.index("--seed") + 1] == "352"
    assert argv[argv.index("--verify_steps") + 1] == "8"
    assert argv[argv.index("--_candidate42_run_id") + 1] == identity.run_id
    assert argv[argv.index("--_candidate42_attempt_id") + 1] == identity.attempt_id
    assert argv[argv.index("--_candidate42_attempt_number") + 1] == "3"
    assert argv[argv.index("--_candidate42_lock_fd") + 1] == "17"


def test_independent_online_reconstruction_has_exact_d8_clock() -> None:
    common = torch.ones((6, 7), dtype=torch.float32)
    stable = torch.tensor([True, True, True, False, True, True])
    before = torch.tensor([0, 1, 7, 8, 8, 8], dtype=torch.long)
    treatment = torch.tensor([True, True, True, True, False, True])
    activated = torch.tensor([True, True, True, True, True, False])
    option = torch.ones(6, dtype=torch.bool)
    active = torch.ones(6, dtype=torch.bool)
    result = collection.reconstruct_public_arm_ramp(
        common_arm=common,
        stable_current=stable,
        stable_count_before=before,
        treatment=treatment,
        activated_this_action=activated,
        option_active=option,
        episode_active=active,
    )
    expected_ramp = torch.tensor([0.0, 0.125, 0.875, 0.0, 1.0, 1.0])
    expected_authority = torch.tensor([0.0, 0.125, 0.875, 0.0, 1.0, 1.0])
    expected_after = torch.tensor([1, 2, 8, 0, 8, 8], dtype=torch.long)
    assert result.ramp_scale.dtype == torch.float32
    assert torch.equal(result.ramp_scale, expected_ramp)
    assert torch.equal(result.authority_scale, expected_authority)
    assert torch.equal(result.stable_count_after, expected_after)
    assert torch.equal(result.requested_arm, expected_authority.unsqueeze(-1).expand(-1, 7))


def test_reconstruction_rejects_c42_clock_values_above_d8() -> None:
    with pytest.raises(ValueError, match="sealed public clock"):
        collection.reconstruct_public_arm_ramp(
            common_arm=torch.zeros((1, 7), dtype=torch.float32),
            stable_current=torch.ones(1, dtype=torch.bool),
            stable_count_before=torch.tensor([9], dtype=torch.long),
            treatment=torch.ones(1, dtype=torch.bool),
            activated_this_action=torch.ones(1, dtype=torch.bool),
            option_active=torch.ones(1, dtype=torch.bool),
            episode_active=torch.ones(1, dtype=torch.bool),
        )


def test_first_step_wrapper_arms_exactly_once_before_env_step() -> None:
    events: list[str] = []

    class Env:
        def step(self, action: torch.Tensor) -> tuple[object, ...]:
            events.append("step")
            return (None, None, None, None, None)

    progress = collection.AttemptProgress()
    action = torch.zeros((1, 21), dtype=torch.float32)

    def boundary() -> None:
        events.append("boundary")

    collection.authoritative_env_step(
        Env(), action, progress=progress, durable_boundary=boundary
    )
    collection.authoritative_env_step(
        Env(), action, progress=progress, durable_boundary=boundary
    )
    assert events == ["boundary", "step", "step"]
    assert progress.first_env_step_invoked is True


def test_assignment_is_exact_balanced_complement_for_all_registered_seeds() -> None:
    for seed, _, num_envs in collection.CANDIDATE43_INVOCATION_SEQUENCE:
        a = supervisor.exact_balanced_treatment_mask(
            seed=seed, num_envs=num_envs, replicate="a"
        )
        b = supervisor.exact_balanced_treatment_mask(
            seed=seed, num_envs=num_envs, replicate="b"
        )
        assert int(a.sum()) == num_envs // 2
        assert torch.equal(b, ~a)
        rank = supervisor.assignment_rank(seed=seed, num_envs=num_envs)
        assert torch.equal(torch.sort(rank).values, torch.arange(num_envs))
        assert supervisor.assignment_mask_sha256(a) != (
            supervisor.assignment_mask_sha256(b)
        )


def test_smoke_requires_both_partial_and_full_d8_authority() -> None:
    episodes = {
        "treatment": torch.tensor([True, True, False, False]),
        "eligible": torch.tensor([True, True, True, False]),
        "partial_authority_rows": torch.tensor([2, 0, 0, 0]),
        "full_authority_rows": torch.tensor([0, 1, 0, 0]),
    }
    collection._validate_smoke_authority_coverage(episodes)
    missing_full = {key: value.clone() for key, value in episodes.items()}
    missing_full["full_authority_rows"].zero_()
    with pytest.raises(ValueError, match="full arm authority"):
        collection._validate_smoke_authority_coverage(missing_full)
