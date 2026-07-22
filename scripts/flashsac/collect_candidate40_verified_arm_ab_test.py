#!/usr/bin/env python3
"""CPU-only pre-launch tests for the Candidate40 collector."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest import mock

import torch

import candidate40_verified_arm_episode as contract
import collect_candidate40_verified_arm_ab as collector


def _raises(error: type[BaseException], fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _head() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=_root(),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _sealed_plan() -> dict:
    return {
        "implementation_seal": {
            "implementation_commit": _head(),
        }
    }


def _clean_git() -> dict:
    return {
        "commit": _head(),
        "branch": contract.REQUIRED_BRANCH,
        "source_files_dirty": False,
        "flashsac_commit": contract.FLASHSAC_FORK_COMMIT,
        "flashsac_dirty": False,
    }


def _args(*, num_envs: int = 8, seed: int = 334, replicate: str = "b") -> argparse.Namespace:
    root = _root()
    output = (
        (root / contract.SMOKE_ARTIFACT_PATH).with_suffix("")
        if num_envs == 8
        else collector._canonical_dev_stem(root, seed, replicate)
    )
    return argparse.Namespace(
        v6_checkpoint=root / contract.V6_CHECKPOINT_PATH,
        search_checkpoint=root / contract.SEARCH_CHECKPOINT_PATH,
        fixed_direction=root / contract.FIXED_DIRECTION_PATH,
        seed=seed,
        replicate=replicate,
        num_envs=num_envs,
        window_steps=contract.WINDOW_STEPS,
        token_scale=contract.TOKEN_SCALE,
        distal_scale=contract.DISTAL_SCALE,
        raw_z_abs_cap=contract.RAW_Z_ABS_CAP,
        token_component_cap=contract.TOKEN_COMPONENT_CAP,
        distal_component_cap=contract.DISTAL_COMPONENT_CAP,
        pre_tanh_l2_cap=contract.PRE_TANH_L2_CAP,
        verify_steps=contract.verifier.VERIFY_STEPS,
        kit_args=contract.KIT_ARGS,
        output_stem=output,
    )


def _build(args: argparse.Namespace) -> collector.CollectionSpec:
    with mock.patch.object(
        contract, "validate_sealed_plan", return_value=_sealed_plan()
    ), mock.patch.object(
        collector, "git_provenance", return_value=_clean_git()
    ), mock.patch.object(
        collector, "validate_implementation_source_authority"
    ), mock.patch.object(collector, "_owned", return_value=False):
        return collector.build_spec(args)


def test_smoke_build_spec_freezes_canonical_authorities() -> None:
    spec = _build(_args())
    assert spec.num_envs == 8
    assert (spec.seed, spec.replicate) == (334, "b")
    assert spec.smoke_artifact_sha256 is None
    assert spec.smoke_report_sha256 is None
    expected_from_rank = ~(spec.assignment_rank < 4)
    assert torch.equal(spec.treatment, expected_from_rank)
    assert spec.source_sha256 == collector.source_fingerprints(_root())
    assert spec.collection_commit == _head()
    assert (
        "scripts/flashsac/collect_candidate39_episode_residual_ab.py"
        in spec.source_sha256
    )
    assert collector._current_hashes(spec) == spec.checkpoint_sha256


def test_build_spec_rejects_aliases_wrong_outputs_and_bool_size() -> None:
    args = _args()
    bad = copy.copy(args)
    bad.num_envs = True
    _raises(ValueError, _build, bad)
    with tempfile.TemporaryDirectory() as name:
        temp = Path(name)
        bad = copy.copy(args)
        bad.output_stem = temp / "trial"
        _raises(ValueError, _build, bad)
        alternate = temp / "same_bytes.pt"
        shutil.copy2(args.fixed_direction, alternate)
        bad = copy.copy(args)
        bad.fixed_direction = alternate
        _raises(ValueError, _build, bad)


def test_dev_requires_smoke_receipts_and_preserves_run_order() -> None:
    args = _args(num_envs=64, seed=335, replicate="a")
    with mock.patch.object(
        contract, "validate_sealed_plan", return_value=_sealed_plan()
    ), mock.patch.object(
        collector, "git_provenance", return_value=_clean_git()
    ), mock.patch.object(
        collector, "validate_implementation_source_authority"
    ), mock.patch.object(
        collector,
        "_validate_smoke_prerequisite",
        return_value=("d" * 64, "e" * 64),
    ) as smoke, mock.patch.object(
        collector, "_validate_run_order"
    ) as run_order, mock.patch.object(collector, "_owned", return_value=False):
        spec = collector.build_spec(args)
    smoke.assert_called_once_with(_root())
    run_order.assert_called_once()
    assert spec.smoke_artifact_sha256 == "d" * 64
    assert spec.smoke_report_sha256 == "e" * 64

    with tempfile.TemporaryDirectory() as name:
        temp = Path(name)
        first = collector._canonical_dev_stem(temp, 335, "a")
        collector._validate_run_order(
            temp,
            seed=335,
            replicate="a",
            output_stem=first,
            source_sha256={"source.py": "a" * 64},
            smoke_artifact_sha256="d" * 64,
            smoke_report_sha256="e" * 64,
            collection_commit=_head(),
        )
        later = collector._canonical_dev_stem(temp, 336, "a")
        later.parent.mkdir(parents=True)
        Path(f"{later}.pt").write_bytes(b"out-of-order")
        _raises(
            ValueError,
            collector._validate_run_order,
            temp,
            seed=335,
            replicate="a",
            output_stem=first,
            source_sha256={"source.py": "a" * 64},
            smoke_artifact_sha256="d" * 64,
            smoke_report_sha256="e" * 64,
            collection_commit=_head(),
        )


def test_prior_authority_rejects_stale_sources_and_smoke() -> None:
    source = {"source.py": "a" * 64}
    metadata = {
        "source_sha256": dict(source),
        "smoke_artifact_sha256": "d" * 64,
        "smoke_report_sha256": "e" * 64,
        "git": {"commit": _head()},
        "runtime_asset_sha256": dict(
            collector.RUNTIME_ASSET_EXPECTED_SHA256
        ),
    }
    collector._validate_prior_authority(
        metadata,
        source_sha256=source,
        smoke_artifact_sha256="d" * 64,
        smoke_report_sha256="e" * 64,
        collection_commit=_head(),
    )
    stale_source = copy.deepcopy(metadata)
    stale_source["source_sha256"]["source.py"] = "b" * 64
    _raises(
        ValueError,
        collector._validate_prior_authority,
        stale_source,
        source_sha256=source,
        smoke_artifact_sha256="d" * 64,
        smoke_report_sha256="e" * 64,
        collection_commit=_head(),
    )
    stale_smoke = copy.deepcopy(metadata)
    stale_smoke["smoke_artifact_sha256"] = "f" * 64
    _raises(
        ValueError,
        collector._validate_prior_authority,
        stale_smoke,
        source_sha256=source,
        smoke_artifact_sha256="d" * 64,
        smoke_report_sha256="e" * 64,
        collection_commit=_head(),
    )
    stale_commit = copy.deepcopy(metadata)
    stale_commit["git"]["commit"] = "f" * 40
    _raises(
        ValueError,
        collector._validate_prior_authority,
        stale_commit,
        source_sha256=source,
        smoke_artifact_sha256="d" * 64,
        smoke_report_sha256="e" * 64,
        collection_commit=_head(),
    )
    stale_runtime = copy.deepcopy(metadata)
    first_asset = next(iter(stale_runtime["runtime_asset_sha256"]))
    stale_runtime["runtime_asset_sha256"][first_asset] = "0" * 64
    _raises(
        ValueError,
        collector._validate_prior_authority,
        stale_runtime,
        source_sha256=source,
        smoke_artifact_sha256="d" * 64,
        smoke_report_sha256="e" * 64,
        collection_commit=_head(),
    )


def test_git_ancestry_is_fail_closed() -> None:
    collector.require_git_ancestor(_root(), _head())
    _raises(ValueError, collector.require_git_ancestor, _root(), "f" * 40)


def test_complete_source_authority_rejects_changed_blob() -> None:
    root = _root()
    relative = "scripts/flashsac/online_handoff.py"
    current = contract.sha256_file(root / relative)
    collector.validate_implementation_source_authority(
        root, {relative: current}, implementation_commit=_head()
    )
    _raises(
        ValueError,
        collector.validate_implementation_source_authority,
        root,
        {relative: "f" * 64},
        implementation_commit=_head(),
    )
    lfs_relative = (
        "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/"
        "textured_mesh.obj"
    )
    collector.validate_implementation_source_authority(
        root,
        {lfs_relative: contract.sha256_file(root / lfs_relative)},
        implementation_commit=_head(),
    )


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    test_smoke_build_spec_freezes_canonical_authorities()
    test_build_spec_rejects_aliases_wrong_outputs_and_bool_size()
    test_dev_requires_smoke_receipts_and_preserves_run_order()
    test_prior_authority_rejects_stale_sources_and_smoke()
    test_git_ancestry_is_fail_closed()
    test_complete_source_authority_rejects_changed_blob()
    print("collect_candidate40_verified_arm_ab tests passed")


if __name__ == "__main__":
    main()
