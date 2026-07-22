#!/usr/bin/env python3
"""Collect one preregistered randomized factual SEARCH-versus-V6 trial arm.

Every first episode executes the same frozen SEARCH policy until the public
readiness state triggers once.  A SHA256-ranked assignment committed before the
rollout then chooses either continued SEARCH or the complete checkpoint-native
V6 router.  Only the executed treatment's outcome is recorded; this collector
does not manufacture per-transition counterfactual labels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence

import torch


MANIFEST_KIND = "pick_tool_public_route_trial_manifest_v2"
DEFAULT_MANIFEST = Path(__file__).with_name("public_route_trial_manifest.json")
COLLECTOR_REPORT_KIND = "pick_tool_public_route_trial_collection_report_v2"
OBSERVATION_DIM = 115
ACTION_DIM = 21
CONFIGURED_MAX_EPISODE_LENGTH = 1000
# DirectRLEnv increments episode_length_buf before _get_dones(), while this
# task truncates at max_episode_length - 1.  The frozen MDP therefore executes
# actions with zero-based pre-action indices 0..998: 999 actions in total.
NATIVE_TIMEOUT_ACTION_COUNT = 999


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains invalid constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _require_regular(path: Path, label: str) -> Path:
    path = path.expanduser()
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _resolve_repository_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"manifest {label} must be a non-empty repository-relative path")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"manifest {label} escapes the repository") from error
    return path


@dataclass(frozen=True)
class CollectionSpec:
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str
    repository_root: Path
    cohort: str
    seed: int
    replicate: str
    num_envs: int
    assignment_salt: str
    search_checkpoint: Path
    search_checkpoint_sha256: str
    route_checkpoint: Path
    route_actor_sha256: str
    route_task_contract_sha256: str
    route_frozen_actor_sha256: str
    route_bridge_state_sha256: str


@dataclass(frozen=True)
class TrialOutputPaths:
    artifact: Path
    report: Path


def load_collection_spec(
    manifest_path: Path,
    *,
    cohort: str,
    seed: int,
    replicate: str,
) -> CollectionSpec:
    """Strictly load the committed trial choices before Isaac is launched."""

    source = _require_regular(manifest_path, "trial manifest")
    digest = sha256_file(source)
    with source.open("r", encoding="utf-8") as stream:
        manifest = json.load(
            stream,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    if not isinstance(manifest, dict):
        raise TypeError("trial manifest root must be a JSON object")
    required_top = {
        "assignment",
        "checkpoints",
        "claim_boundary",
        "cohorts",
        "collection_acceptance",
        "feature_contract",
        "flashsac",
        "format_version",
        "gate",
        "kind",
        "outputs",
        "preregistration",
        "runtime_assets",
        "status",
        "task",
        "training",
        "trial_population",
    }
    if set(manifest) != required_top:
        raise ValueError("trial manifest top-level schema is not exact")
    if (
        manifest["kind"] != MANIFEST_KIND
        or manifest["format_version"] != 2
        or manifest["status"] != "preregistered"
    ):
        raise ValueError("unsupported or non-preregistered trial manifest")
    if cohort not in {"pilot", "train", "development"}:
        raise ValueError("collector cohort must be pilot, train, or development")
    cohorts = manifest["cohorts"]
    if not isinstance(cohorts, dict) or set(cohorts) != {
        "pilot",
        "train",
        "development",
        "blind",
    }:
        raise ValueError("trial cohorts schema is invalid")
    seed_sets: dict[str, set[int]] = {}
    for name, entry in cohorts.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("seeds"), list):
            raise TypeError(f"manifest cohort {name} is invalid")
        values = entry["seeds"]
        if any(type(value) is not int for value in values) or len(set(values)) != len(values):
            raise ValueError(f"manifest cohort {name} seeds must be unique integers")
        seed_sets[name] = set(values)
    names = sorted(seed_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if seed_sets[left] & seed_sets[right]:
                raise ValueError(f"manifest cohorts {left} and {right} overlap")
    entry = cohorts[cohort]
    if seed not in seed_sets[cohort]:
        raise ValueError(f"seed {seed} is not preregistered for cohort {cohort}")
    if replicate not in entry.get("replicates", []):
        raise ValueError(f"replicate {replicate!r} is not preregistered for {cohort}")
    num_envs = entry.get("num_envs")
    if type(num_envs) is not int or num_envs < 2 or num_envs % 2:
        raise ValueError("trial num_envs must be a positive even integer")

    assignment = manifest["assignment"]
    expected_assignment_keys = {
        "algorithm",
        "canonical_bytes",
        "complement_pair_missing_rule",
        "episode_index",
        "known_route_propensity",
        "rank_rule",
        "randomization_block",
        "randomization_unit",
        "replicates",
        "replicate_b_semantics",
        "run_order",
        "salt",
    }
    if not isinstance(assignment, dict) or set(assignment) != expected_assignment_keys:
        raise ValueError("trial assignment contract is invalid")
    expected_assignment_values = {
        "algorithm": "sha256_rank_balanced_v1",
        "canonical_bytes": "UTF-8(salt + NUL + cohort + NUL + decimal_seed + NUL + decimal_env_slot + NUL + episode=0)",
        "complement_pair_missing_rule": "dataset publication requires both successful immutable replicates; never impute or retain one side alone",
        "episode_index": 0,
        "known_route_propensity": 0.5,
        "rank_rule": "lexicographic (sha256_digest_bytes, env_slot); first num_envs/2 route in replicate a; ties break by env_slot",
        "randomization_block": "cohort_x_seed_x_first_episode_vector_run",
        "randomization_unit": "cohort_x_seed_x_env_slot_x_episode0",
        "replicates": ["a", "b"],
        "replicate_b_semantics": "exact_boolean_complement_of_replicate_a",
        "run_order": "a_then_b_for_even_seed__b_then_a_for_odd_seed",
    }
    for key, expected in expected_assignment_values.items():
        if assignment.get(key) != expected:
            raise ValueError(f"trial assignment field {key} is invalid")
    salt = assignment["salt"]
    if not isinstance(salt, str) or len(salt) < 16:
        raise ValueError("trial assignment salt must be a committed non-empty string")
    claims = manifest["claim_boundary"]
    if (
        not isinstance(claims, dict)
        or claims.get("causal_counterfactual_claim_allowed") is not False
        or claims.get("pairing_semantics")
        != "independent_gpu_randomized_factual_arm_trial_v1"
    ):
        raise ValueError("trial claim boundary permits an unsupported causal claim")
    feature = manifest["feature_contract"]
    if (
        not isinstance(feature, dict)
        or feature.get("kind") != "pick_tool_public_route_gate_feature165_v1"
        or feature.get("feature_dim") != 165
        or feature.get("gate_state_dim_before_actions") != 123
        or feature.get("directly_excluded_observation_indices") != [86]
    ):
        raise ValueError("trial feature contract is invalid")
    task = manifest["task"]
    expected_task = {
        "action_dim": ACTION_DIM,
        "close_option_mode": False,
        "coupled_power_align_close_option_mode": False,
        "curriculum_dataset": "",
        "curriculum_joint_noise": 0.0,
        "curriculum_reset_probability": 0.0,
        "episode_length_s": 20.0,
        "hard_force_limit_n": 30.0,
        "hard_force_terminate_steps": 10,
        "grasp_confirm_steps": 4,
        "grasp_release_steps": 6,
        "max_episode_steps": CONFIGURED_MAX_EPISODE_LENGTH,
        "timeout_action_count": NATIVE_TIMEOUT_ACTION_COUNT,
        "observation_contract": "pick_tool_markov115_v1",
        "observation_dim": OBSERVATION_DIM,
        "overforce_limit_n": 60.0,
        "overforce_terminate_steps": 2,
        "hold_arm_until_stable_grasp": False,
        "power_close_option_mode": False,
        "public_gate_state_contract": "pick_tool_public_gate_state_v1",
        "success_hold_steps": 15,
        "success_true_clearance_m": 0.2,
        "task_mode": "full_task",
    }
    if task != expected_task:
        raise ValueError("trial task contract differs from the authored full task")
    gate = manifest["gate"]
    if (
        not isinstance(gate, dict)
        or gate.get("consecutive_eligible_steps") != 4
        or gate.get("once_per_episode") is not True
        or gate.get("grasp_score_stratum_threshold") != 0.35
        or gate.get("close_quality_stratum_threshold") != 0.2
    ):
        raise ValueError("trial public readiness gate is invalid")
    flashsac = manifest["flashsac"]
    if (
        not isinstance(flashsac, dict)
        or flashsac.get("use_compile") is not False
        or not isinstance(flashsac.get("fork_commit"), str)
    ):
        raise ValueError("trial FlashSAC contract is invalid")
    outputs = manifest["outputs"]
    expected_outputs = {
        "artifact_template": "{cohort}_s{seed}_{replicate}.pt",
        "failure_report_template": "{cohort}_s{seed}_{replicate}.failed_attempt_{attempt:03d}.json",
        "global_run_order": "pilot_then_train_then_development; within_cohort_manifest_seed_order; within_seed_even_a_then_b__odd_b_then_a; every_predecessor_requires_immutable_complete_receipt",
        "report_template": "{cohort}_s{seed}_{replicate}.json",
        "repository_relative_root": "logs/flashsac/pick_tool/29_public_route_trial_v2_20260722",
        "run_order_enforced": True,
    }
    if outputs != expected_outputs:
        raise ValueError("trial output contract is invalid")
    preregistration = manifest["preregistration"]
    expected_preregistration_keys = {
        "branch",
        "implementation_commit",
        "implementation_base_commit",
        "manifest_sha256_receipt",
        "partial_or_failed_run_rule",
        "seal_rule",
        "seal_tag",
        "source_binding",
    }
    if not isinstance(preregistration, dict) or set(preregistration) != expected_preregistration_keys:
        raise ValueError("trial preregistration contract is invalid")
    implementation_commit = preregistration.get("implementation_commit")
    base_commit = preregistration.get("implementation_base_commit")
    if not all(
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
        for value in (implementation_commit, base_commit)
    ):
        raise ValueError("trial implementation commits must be lowercase full Git SHA-1 values")
    if preregistration.get("seal_tag") != "pick-tool-public-route-trial-v2-20260722":
        raise ValueError("trial seal tag is invalid")
    if preregistration.get("branch") != "flashsac-pick-tool-curriculum":
        raise ValueError("trial branch is invalid")
    runtime_assets = manifest["runtime_assets"]
    expected_runtime_asset_paths = {
        "/tmp/xhand_inhand/pick_tool_token/.asset_hash",
        "/tmp/xhand_inhand/pick_tool_token/Props/instanceable_meshes.usd",
        "/tmp/xhand_inhand/pick_tool_token/tool_hammer.usd",
        "source/xhand_inhand/xhand_inhand/assets/xarm7_xhand/configuration/xarm7_xhand_base.usd",
        "source/xhand_inhand/xhand_inhand/assets/xarm7_xhand/configuration/xarm7_xhand_physics.usd",
        "source/xhand_inhand/xhand_inhand/assets/xarm7_xhand/configuration/xarm7_xhand_robot.usd",
        "source/xhand_inhand/xhand_inhand/assets/xarm7_xhand/configuration/xarm7_xhand_sensor.usd",
        "source/xhand_inhand/xhand_inhand/assets/xarm7_xhand/xarm7_xhand.usd",
    }
    if (
        not isinstance(runtime_assets, dict)
        or set(runtime_assets) != {"contract", "files"}
        or runtime_assets.get("contract")
        != "exact_regular_nonsymlink_sha256_before_and_after_simulation_v1"
        or not isinstance(runtime_assets.get("files"), dict)
        or set(runtime_assets["files"]) != expected_runtime_asset_paths
    ):
        raise ValueError("trial runtime asset contract is invalid")
    for expected in runtime_assets["files"].values():
        if not (
            isinstance(expected, str)
            and len(expected) == 64
            and all(character in "0123456789abcdef" for character in expected)
        ):
            raise ValueError("trial runtime asset SHA256 is invalid")

    root = Path(__file__).resolve().parents[2]
    checkpoints = manifest["checkpoints"]
    if not isinstance(checkpoints, dict) or set(checkpoints) != {"search", "route_v6"}:
        raise ValueError("trial checkpoint schema is invalid")
    search = checkpoints["search"]
    route = checkpoints["route_v6"]
    if not isinstance(search, dict) or set(search) != {"path", "sha256"}:
        raise ValueError("SEARCH checkpoint manifest entry is invalid")
    if not isinstance(route, dict) or set(route) != {
        "path",
        "actor_sha256",
        "task_contract_sha256",
        "frozen_lift_actor_sha256",
        "torch_bridge_state_sha256",
    }:
        raise ValueError("V6 checkpoint manifest entry is invalid")
    search_path = _require_regular(
        _resolve_repository_path(root, search["path"], "SEARCH checkpoint"),
        "SEARCH checkpoint",
    )
    route_path = _resolve_repository_path(root, route["path"], "V6 checkpoint")
    if route_path.is_symlink() or not route_path.is_dir():
        raise FileNotFoundError("V6 checkpoint must be a real directory")
    hash_entries = (
        (search_path, search["sha256"], "SEARCH checkpoint"),
        (route_path / "actor.pt", route["actor_sha256"], "V6 actor"),
        (
            route_path / "task_contract.json",
            route["task_contract_sha256"],
            "V6 task contract",
        ),
        (
            route_path / "frozen_lift_actor.pt",
            route["frozen_lift_actor_sha256"],
            "V6 frozen actor",
        ),
        (
            route_path / "torch_bridge_state.pt",
            route["torch_bridge_state_sha256"],
            "V6 Torch bridge state",
        ),
    )
    for path, expected, label in hash_entries:
        path = _require_regular(path, label)
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise RuntimeError(f"{label} SHA256 disagrees with the preregistered manifest")
    if sha256_file(source) != digest:
        raise RuntimeError("trial manifest changed while it was being loaded")
    return CollectionSpec(
        manifest=manifest,
        manifest_path=source,
        manifest_sha256=digest,
        repository_root=root,
        cohort=cohort,
        seed=seed,
        replicate=replicate,
        num_envs=num_envs,
        assignment_salt=salt,
        search_checkpoint=search_path,
        search_checkpoint_sha256=search["sha256"],
        route_checkpoint=route_path,
        route_actor_sha256=route["actor_sha256"],
        route_task_contract_sha256=route["task_contract_sha256"],
        route_frozen_actor_sha256=route["frozen_lift_actor_sha256"],
        route_bridge_state_sha256=route["torch_bridge_state_sha256"],
    )


def _git_command(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def runtime_asset_fingerprints(spec: CollectionSpec) -> dict[str, str]:
    """Authenticate ignored/generated USD bytes that directly enter physics."""

    result: dict[str, str] = {}
    for declared, expected in spec.manifest["runtime_assets"]["files"].items():
        declared_path = Path(declared)
        path = (
            declared_path
            if declared_path.is_absolute()
            else _resolve_repository_path(
                spec.repository_root, declared, f"runtime asset {declared}"
            )
        )
        path = _require_regular(path, f"runtime asset {declared}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"runtime asset SHA256 differs from manifest: {declared}")
        result[declared] = actual
    return result


def source_provenance(
    spec: CollectionSpec,
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    from evaluate import diagnostic_handoff_source_fingerprints
    from search_replay_trace import source_fingerprints

    repository_manifest = _require_regular(DEFAULT_MANIFEST, "repository trial manifest")
    if spec.manifest_path != repository_manifest:
        raise RuntimeError("collection must use the sealed repository trial manifest")
    result = source_fingerprints(spec.repository_root)
    diagnostic = diagnostic_handoff_source_fingerprints(spec.repository_root)
    for relative, digest in diagnostic.items():
        if relative in result and result[relative] != digest:
            raise RuntimeError(f"source fingerprint helpers disagree for {relative}")
        result[relative] = digest
    for relative in (
        "scripts/flashsac/collect_public_route_trial.py",
        "scripts/flashsac/public_route_trial_contract.py",
        "scripts/flashsac/public_route_trial_manifest.json",
        "source/xhand_inhand/xhand_inhand/__init__.py",
        "source/xhand_inhand/xhand_inhand/tasks/__init__.py",
        "source/xhand_inhand/xhand_inhand/tasks/direct/__init__.py",
        "source/xhand_inhand/xhand_inhand/tasks/direct/pick_cube/__init__.py",
        "source/xhand_inhand/xhand_inhand/tasks/direct/pick_cube_token/__init__.py",
        "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/__init__.py",
        "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/public_gate_state.py",
    ):
        path = _require_regular(spec.repository_root / relative, f"source {relative}")
        result[relative] = sha256_file(path)
    runtime_assets = runtime_asset_fingerprints(spec)
    for declared, digest in runtime_assets.items():
        if Path(declared).is_absolute():
            continue
        source_digest = result.pop(declared, None)
        if source_digest != digest:
            raise RuntimeError(
                f"runtime asset tree and manifest disagree for {declared}"
            )
    tracked = sorted(result)
    dirty = _git_command(spec.repository_root, "status", "--porcelain", "--", *tracked)
    flash_root = spec.repository_root / "third_party/FlashSAC"
    flash_commit = _git_command(flash_root, "rev-parse", "HEAD")
    flash_dirty = _git_command(flash_root, "status", "--porcelain")
    expected_fork = spec.manifest["flashsac"]["fork_commit"]
    if flash_commit != expected_fork or flash_dirty:
        raise RuntimeError("FlashSAC fork commit or worktree differs from the manifest")
    preregistration = spec.manifest["preregistration"]
    seal_tag = preregistration["seal_tag"]
    tag_ref = f"refs/tags/{seal_tag}"
    if _git_command(spec.repository_root, "cat-file", "-t", tag_ref) != "tag":
        raise RuntimeError("trial seal must be an annotated Git tag")
    seal_commit = _git_command(
        spec.repository_root,
        "rev-parse",
        "--verify",
        f"{tag_ref}^{{commit}}",
    )
    seal_line = _git_command(
        spec.repository_root, "rev-list", "--parents", "-n", "1", seal_commit
    ).split()
    if len(seal_line) != 2 or seal_line[1] != preregistration["implementation_commit"]:
        raise RuntimeError("seal tag must name the sole child of implementation_commit")
    seal_changed_paths = _git_command(
        spec.repository_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        seal_commit,
    ).splitlines()
    if seal_changed_paths != ["scripts/flashsac/public_route_trial_manifest.json"]:
        raise RuntimeError("seal commit must change only the registered manifest")
    implementation_line = _git_command(
        spec.repository_root,
        "rev-list",
        "--parents",
        "-n",
        "1",
        preregistration["implementation_commit"],
    ).split()
    if (
        len(implementation_line) != 2
        or implementation_line[1] != preregistration["implementation_base_commit"]
    ):
        raise RuntimeError("implementation commit must be the sole child of implementation_base_commit")
    current_commit = _git_command(spec.repository_root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", seal_commit, current_commit),
        cwd=spec.repository_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("current HEAD does not descend from the preregistered seal tag")
    for relative in tracked:
        if relative.startswith("third_party/FlashSAC/"):
            continue
        sealed_blob = _git_command(
            spec.repository_root, "rev-parse", f"{seal_commit}:{relative}"
        )
        current_blob = _git_command(
            spec.repository_root, "hash-object", "--", relative
        )
        if current_blob != sealed_blob:
            raise RuntimeError(f"execution source differs from the seal tag: {relative}")
    branch = _git_command(spec.repository_root, "rev-parse", "--abbrev-ref", "HEAD")
    if branch != preregistration["branch"]:
        raise RuntimeError("trial collection must run on its dedicated sealed branch")
    git = {
        "commit": current_commit,
        "branch": branch,
        "seal_tag": seal_tag,
        "seal_commit": seal_commit,
        "implementation_commit": preregistration["implementation_commit"],
        "source_files_dirty": bool(dirty),
        "flashsac_commit": flash_commit,
        "flashsac_dirty": bool(flash_dirty),
    }
    if git["source_files_dirty"]:
        raise RuntimeError("trial source files are dirty; commit before collecting evidence")
    return result, runtime_assets, git


def _installed_versions(names: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def runtime_provenance(seed: int, *, device: torch.device) -> dict[str, Any]:
    from isaaclab.utils.version import get_isaac_sim_version

    try:
        inventory = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,driver_version",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        inventory = f"unavailable:{type(error).__name__}"
    if device.type != "cuda":
        raise ValueError("runtime provenance requires the actual CUDA execution device")
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "cudnn": int(torch.backends.cudnn.version() or 0),
        "cuda_device_index": device_index,
        "cuda_device_name": torch.cuda.get_device_name(device_index),
        "cuda_device_capability": list(torch.cuda.get_device_capability(device_index)),
        "isaac_sim": str(get_isaac_sim_version()),
        "packages": _installed_versions(
            ("isaaclab", "isaaclab_tasks", "isaaclab_assets", "numpy", "gymnasium")
        ),
        "nvidia_smi_inventory": inventory,
        "platform": platform.platform(),
        "seed": seed,
    }


def _make_route_agent(env: Any, spec: CollectionSpec) -> tuple[Any, dict[str, Any]]:
    # Import the sibling contract reader before agent_bridge prepends the
    # FlashSAC source tree, which also contains a train.py.
    from train import read_checkpoint_task_contract
    from agent_bridge import (
        FLASH_SAC_COMMIT,
        FLASH_SAC_FORK_COMMIT,
        ActionAuthorityRule,
        ActionNoiseGroup,
        FlashSACTorchBridge,
        PublicLatchFrozenActorRouter,
        build_agent_config,
    )
    from evaluate import (
        ARM_ACTION_DIM,
        FULL_TASK_GRASP_LATCH_OBSERVATION_INDEX,
        NOISE_GROUP_SPECS,
        PRODUCTION_ACTOR_BLOCKS,
        PRODUCTION_ACTOR_HIDDEN,
        PRODUCTION_CRITIC_BINS,
        PRODUCTION_CRITIC_BLOCKS,
        PRODUCTION_CRITIC_HIDDEN,
        infer_checkpoint_actor_action_dim,
        infer_checkpoint_architecture,
        resolve_checkpoint_directory,
        validate_checkpoint_evaluation_contract,
    )

    checkpoint = resolve_checkpoint_directory(spec.route_checkpoint)
    contract = read_checkpoint_task_contract(checkpoint)
    if contract.get("version") != 6 or contract.get("task_mode") != "full_task":
        raise ValueError("route checkpoint must be the authored full-task V6 contract")
    router = contract.get("policy_router")
    if not isinstance(router, Mapping) or router.get("kind") != "public_latch_frozen_actor_v1":
        raise ValueError("route checkpoint lacks the checkpoint-native V6 router")
    actor_dim = infer_checkpoint_actor_action_dim(checkpoint)
    source_contract, target_contract, source_indices = validate_checkpoint_evaluation_contract(
        checkpoint_task_mode="full_task",
        checkpoint_contract=contract,
        requested_task_mode="full_task",
        actor_action_dim=actor_dim,
    )
    if source_indices is not None or source_contract != target_contract:
        raise RuntimeError("V6 trial requires an exact full-task actor load")
    architecture = infer_checkpoint_architecture(
        checkpoint,
        expected_action_dim=ACTION_DIM,
        expected_observation_dim=OBSERVATION_DIM,
    )
    if architecture != "production":
        raise RuntimeError("V6 trial route actor must use the production architecture")
    expected_authority = [
        {
            "name": "arm_after_public_latch",
            "start": 0,
            "stop": ARM_ACTION_DIM,
            "observation_index": FULL_TASK_GRASP_LATCH_OBSERVATION_INDEX,
            "active_value": 1.0,
        }
    ]
    if contract.get("policy_action_authority") != expected_authority:
        raise ValueError("V6 checkpoint public action authority is invalid")
    if FLASH_SAC_FORK_COMMIT != spec.manifest["flashsac"]["fork_commit"]:
        raise RuntimeError("loaded FlashSAC bridge fork disagrees with the manifest")

    config = build_agent_config(
        seed=spec.seed,
        normalize_reward=True,
        normalized_G_max=5.0,
        device_type=str(env.device),
        buffer_device_type=str(env.device),
        buffer_max_length=max(spec.num_envs, 32),
        buffer_min_length=1,
        sample_batch_size=1,
        n_step=3,
        actor_num_blocks=PRODUCTION_ACTOR_BLOCKS,
        actor_hidden_dim=PRODUCTION_ACTOR_HIDDEN,
        critic_num_blocks=PRODUCTION_CRITIC_BLOCKS,
        critic_hidden_dim=PRODUCTION_CRITIC_HIDDEN,
        critic_num_bins=PRODUCTION_CRITIC_BINS,
        use_compile=False,
        compile_mode="reduce-overhead",
        use_amp=True,
        load_optimizer=False,
        load_reward_normalizer=False,
    )
    noise_groups = tuple(
        ActionNoiseGroup(
            name,
            start,
            stop,
            scale=scale,
            zeta_mu=zeta_mu,
            zeta_max=zeta_max,
        )
        for name, start, stop, scale, zeta_mu, zeta_max in NOISE_GROUP_SPECS
    )
    agent = FlashSACTorchBridge(
        env.observation_space,
        env.action_space,
        env.env_info,
        config,
        noise_groups=noise_groups,
        restore_rng_state_on_load=False,
        action_authority_rules=tuple(
            ActionAuthorityRule(**rule) for rule in expected_authority
        ),
        public_latch_frozen_actor_router=PublicLatchFrozenActorRouter(
            name=str(router["kind"]),
            observation_index=int(router["observation_index"]),
            trainable_start=int(router["trainable_action_slice"][0]),
            trainable_stop=int(router["trainable_action_slice"][1]),
            close_value=float(router["close_value"]),
            frozen_value=float(router["frozen_value"]),
        ),
        unit_normalize_actor_mean_head=True,
    )
    agent.load_actor(str(checkpoint))
    agent.load_frozen_lift_actor_sidecar(str(checkpoint))
    frozen = router["frozen_actor"]
    if (
        agent.frozen_lift_actor_sha256 != frozen["network_sha256"]
        or agent.frozen_lift_actor_source_sha256 != frozen["source_actor_sha256"]
    ):
        raise RuntimeError("loaded V6 frozen actor lineage is invalid")
    agent.reset_exploration(batch_size=spec.num_envs)
    for path, expected, label in (
        (checkpoint / "actor.pt", spec.route_actor_sha256, "V6 actor"),
        (
            checkpoint / "task_contract.json",
            spec.route_task_contract_sha256,
            "V6 task contract",
        ),
        (
            checkpoint / "frozen_lift_actor.pt",
            spec.route_frozen_actor_sha256,
            "V6 frozen actor",
        ),
        (
            checkpoint / "torch_bridge_state.pt",
            spec.route_bridge_state_sha256,
            "V6 Torch bridge state",
        ),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"{label} changed while it was loading")
    return agent, {
        "checkpoint_contract": contract,
        "upstream_commit": FLASH_SAC_COMMIT,
        "fork_commit": FLASH_SAC_FORK_COMMIT,
        "architecture": architecture,
        "loaded_frozen_network_sha256": agent.frozen_lift_actor_sha256,
        "loaded_frozen_source_actor_sha256": agent.frozen_lift_actor_source_sha256,
    }


def _validate_task_runtime(env: Any, spec: CollectionSpec) -> None:
    cfg = env.unwrapped.cfg
    full_task_checks = {
        "close_option_mode": bool(cfg.close_option_mode),
        "power_close_option_mode": bool(cfg.power_close_option_mode),
        "coupled_power_align_close_option_mode": bool(
            cfg.coupled_power_align_close_option_mode
        ),
        "hold_arm_until_stable_grasp": bool(cfg.hold_arm_until_stable_grasp),
    }
    wrong_modes = {name: value for name, value in full_task_checks.items() if value}
    if wrong_modes:
        raise RuntimeError(f"live task enabled non-full-task modes: {wrong_modes}")
    if str(cfg.curriculum_dataset) != "":
        raise RuntimeError("live full task must not load a curriculum reset dataset")
    checks = {
        "observation_space": (int(cfg.observation_space), OBSERVATION_DIM),
        "action_space": (int(cfg.action_space), ACTION_DIM),
        "max_episode_steps": (
            int(env.max_episode_steps),
            CONFIGURED_MAX_EPISODE_LENGTH,
        ),
        "hard_force_terminate_steps": (int(cfg.tactile_hard_terminate_steps), 10),
        "grasp_confirm_steps": (int(cfg.grasp_confirm_steps), 4),
        "grasp_release_steps": (int(cfg.grasp_release_steps), 6),
        "overforce_terminate_steps": (int(cfg.tactile_terminate_steps), 2),
        "success_hold_steps": (int(cfg.success_hold_steps), 15),
    }
    wrong = {name: values for name, values in checks.items() if values[0] != values[1]}
    if wrong:
        raise RuntimeError(f"live task differs from the preregistered contract: {wrong}")
    float_checks = {
        "episode_length_s": (float(cfg.episode_length_s), 20.0),
        "hard_force_limit_n": (float(cfg.tactile_hard_force_limit), 30.0),
        "overforce_limit_n": (float(cfg.tactile_terminate_force_limit), 60.0),
        "success_true_clearance_m": (float(cfg.lift_success_height), 0.2),
        "curriculum_reset_probability": (
            float(cfg.curriculum_reset_probability),
            0.0,
        ),
        "curriculum_joint_noise": (float(cfg.curriculum_joint_noise), 0.0),
    }
    wrong_float = {
        name: values
        for name, values in float_checks.items()
        if not math.isclose(values[0], values[1], rel_tol=0.0, abs_tol=1.0e-9)
    }
    if wrong_float:
        raise RuntimeError(f"live task float contract differs: {wrong_float}")


def _event_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"episodes": 0}
    factual = "outcome_success" in records[0]
    def key(name: str) -> str:
        return f"outcome_{name}" if factual else name

    return {
        "episodes": len(records),
        "success": sum(bool(row[key("success")]) for row in records),
        "failure": sum(bool(row[key("failure")]) for row in records),
        "time_out": sum(bool(row[key("time_out")]) for row in records),
        "dropped": sum(bool(row[key("dropped")]) for row in records),
        "unsafe_force": sum(bool(row[key("unsafe_force")]) for row in records),
        "ever_grasped": sum(bool(row[key("ever_grasped")]) for row in records),
        "ever_clearance_ge_5cm": sum(
            (
                float(row["outcome_max_true_clearance_m"]) >= 0.05
                if factual
                else bool(row["ever_clearance_ge_5cm"])
            )
            for row in records
        ),
        "max_true_clearance_m": max(
            float(row[key("max_true_clearance_m")]) for row in records
        ),
    }


def canonical_output_paths(spec: CollectionSpec) -> TrialOutputPaths:
    """Resolve the only registered successful output pair for one trial run."""

    outputs = spec.manifest["outputs"]
    root = _resolve_repository_path(
        spec.repository_root,
        outputs["repository_relative_root"],
        "output root",
    )
    values = {
        "cohort": spec.cohort,
        "seed": spec.seed,
        "replicate": spec.replicate,
    }
    artifact = root / outputs["artifact_template"].format(**values)
    report = root / outputs["report_template"].format(**values)
    if artifact.parent != root or report.parent != root or artifact == report:
        raise RuntimeError("registered output templates do not produce a safe pair")
    return TrialOutputPaths(artifact=artifact, report=report)


def _ordered_collection_runs(manifest: Mapping[str, Any]) -> list[tuple[str, int, str]]:
    runs: list[tuple[str, int, str]] = []
    for cohort in ("pilot", "train", "development"):
        entry = manifest["cohorts"][cohort]
        for seed in entry["seeds"]:
            replicate_order = ("a", "b") if seed % 2 == 0 else ("b", "a")
            runs.extend((cohort, seed, replicate) for replicate in replicate_order)
    return runs


def _validate_completed_predecessor(
    spec: CollectionSpec, *, cohort: str, seed: int, replicate: str
) -> None:
    predecessor = CollectionSpec(
        **{
            **spec.__dict__,
            "cohort": cohort,
            "seed": seed,
            "replicate": replicate,
            "num_envs": int(spec.manifest["cohorts"][cohort]["num_envs"]),
        }
    )
    paths = canonical_output_paths(predecessor)
    for path, label in ((paths.artifact, "artifact"), (paths.report, "report")):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"registered run order requires complete predecessor {cohort} "
                f"seed={seed} replicate={replicate}; missing {label}"
            )
    with paths.report.open("r", encoding="utf-8") as stream:
        report = json.load(
            stream,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    if (
        not isinstance(report, dict)
        or report.get("kind") != COLLECTOR_REPORT_KIND
        or report.get("status") != "complete"
        or report.get("cohort") != cohort
        or report.get("seed") != seed
        or report.get("replicate") != replicate
        or report.get("num_envs") != predecessor.num_envs
        or report.get("manifest_sha256") != spec.manifest_sha256
        or report.get("artifact_sha256") != sha256_file(paths.artifact)
    ):
        raise RuntimeError(
            f"registered predecessor receipt is invalid for {cohort} "
            f"seed={seed} replicate={replicate}"
        )


def validate_preregistered_run_order(spec: CollectionSpec) -> TrialOutputPaths:
    """Fail closed unless every registered predecessor has completed immutably."""

    ordered = _ordered_collection_runs(spec.manifest)
    current = (spec.cohort, spec.seed, spec.replicate)
    try:
        current_index = ordered.index(current)
    except ValueError as error:
        raise RuntimeError("trial run is absent from the registered execution order") from error
    paths = canonical_output_paths(spec)
    if _path_owned(paths.artifact) or _path_owned(paths.report):
        raise FileExistsError(
            f"canonical trial output is already owned for {spec.cohort} "
            f"seed={spec.seed} replicate={spec.replicate}"
        )
    for cohort, seed, replicate in ordered[:current_index]:
        _validate_completed_predecessor(
            spec, cohort=cohort, seed=seed, replicate=replicate
        )
    for cohort, seed, replicate in ordered[current_index + 1 :]:
        future_spec = CollectionSpec(
            **{
                **spec.__dict__,
                "cohort": cohort,
                "seed": seed,
                "replicate": replicate,
                "num_envs": int(spec.manifest["cohorts"][cohort]["num_envs"]),
            }
        )
        future = canonical_output_paths(future_spec)
        if _path_owned(future.artifact) or _path_owned(future.report):
            raise RuntimeError("a future canonical output exists ahead of registered run order")
    return paths


def next_failure_report_path(spec: CollectionSpec) -> Path:
    outputs = spec.manifest["outputs"]
    root = canonical_output_paths(spec).report.parent
    values = {
        "cohort": spec.cohort,
        "seed": spec.seed,
        "replicate": spec.replicate,
    }
    for attempt in range(1, 10000):
        candidate = root / outputs["failure_report_template"].format(
            **values, attempt=attempt
        )
        if candidate.parent != root:
            raise RuntimeError("registered failure template escapes the output root")
        if not _path_owned(candidate):
            return candidate
    raise RuntimeError("failure report attempt namespace is exhausted")


@torch.inference_mode()
def run_collection(spec: CollectionSpec, *, device_string: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from public_route_trial_contract import (
        assignment_route_mask,
        build_public_route_feature,
        build_randomized_trial_artifact,
        update_public_route_readiness,
    )

    device = torch.device(device_string)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"public route trial requires CUDA Isaac physics, got {device}")
    assignment_cpu = assignment_route_mask(
        cohort=spec.cohort,
        seed=spec.seed,
        num_envs=spec.num_envs,
        salt=spec.assignment_salt,
        replicate=spec.replicate,
    )
    if assignment_cpu.dtype != torch.bool or int(assignment_cpu.sum()) * 2 != spec.num_envs:
        raise RuntimeError("trial assignment is not an exact balanced bool mask")

    # Import task registration only after the complete assignment vector exists.
    # Those imports may validate or generate simulator assets, but they cannot
    # condition the already frozen treatment allocation.
    from adapter import make_pick_tool_env
    from evaluate import (
        FULL_TASK_MODE,
        StrictEpisodeTracker,
        _load_diagnostic_approach_actor,
        _read_physical_truth,
        _seed_everything,
        physical_truth_from_terminal_info,
        validate_terminal_events,
    )
    from xhand_inhand.tasks.direct.pick_tool_token.public_gate_state import (
        PUBLIC_GATE_STATE_CONTRACT,
        PUBLIC_GATE_STATE_EXTRAS_KEY,
        public_gate_feature_tensor,
        validate_public_gate_feature_values,
    )

    _seed_everything(spec.seed)
    source_hashes, runtime_asset_hashes, git = source_provenance(spec)
    if PUBLIC_GATE_STATE_CONTRACT != spec.manifest["task"]["public_gate_state_contract"]:
        raise RuntimeError("loaded public safety-state version disagrees with the manifest")

    env = make_pick_tool_env(
        num_envs=spec.num_envs,
        device=device_string,
        seed=spec.seed,
        strict=True,
        validate_finite=True,
    )
    try:
        _validate_task_runtime(env, spec)
        route_agent, route_load = _make_route_agent(env, spec)
        search_actor = _load_diagnostic_approach_actor(
            spec.search_checkpoint, device=device
        ).eval()
        if sha256_file(spec.search_checkpoint) != spec.search_checkpoint_sha256:
            raise RuntimeError("SEARCH checkpoint changed while it was loading")

        observation, reset_info = env.reset(
            seed=spec.seed, randomize_episode_lengths=False
        )
        if observation.shape != (spec.num_envs, OBSERVATION_DIM):
            raise RuntimeError("trial reset violated the obs115 contract")
        safety_state = public_gate_feature_tensor(
            reset_info[PUBLIC_GATE_STATE_EXTRAS_KEY],
            num_envs=spec.num_envs,
            device=env.device,
        )
        validate_public_gate_feature_values(safety_state)
        initial_truth = _read_physical_truth(env.unwrapped, task_mode=FULL_TASK_MODE)
        tracker = StrictEpisodeTracker(
            episodes=spec.num_envs,
            num_envs=spec.num_envs,
            device=env.device,
            initial_truth=initial_truth,
            task_mode=FULL_TASK_MODE,
        )
        assignment_route = assignment_cpu.to(device=env.device)

        ready_count = torch.zeros(spec.num_envs, dtype=torch.long, device=env.device)
        fork_used = torch.zeros(spec.num_envs, dtype=torch.bool, device=env.device)
        route_active = torch.zeros_like(fork_used)
        episode_step = torch.zeros(spec.num_envs, dtype=torch.long, device=env.device)
        candidate_seen = torch.zeros_like(fork_used)
        candidate_feature = torch.full(
            (spec.num_envs, 165), float("nan"), dtype=torch.float32, device=env.device
        )
        candidate_score = torch.full(
            (spec.num_envs,), float("nan"), dtype=torch.float32, device=env.device
        )
        candidate_step = torch.full(
            (spec.num_envs,), -1, dtype=torch.long, device=env.device
        )
        candidate_stratum = torch.full_like(candidate_step, -1)
        rows: list[dict[str, Any]] = []
        vector_steps = 0
        cfg = env.unwrapped.cfg

        while not tracker.complete and vector_steps < NATIVE_TIMEOUT_ACTION_COUNT:
            active_before = tracker.active.clone()
            readiness = update_public_route_readiness(
                observation,
                ready_count_before=ready_count,
                fork_used_before=fork_used,
            )
            with torch.no_grad():
                search_action = search_actor(observation).clamp(-1.0, 1.0)
            route_action = route_agent.sample_actions(
                vector_steps + 1,
                {"next_observation": observation},
                training=False,
            )
            if search_action.shape != (spec.num_envs, ACTION_DIM) or route_action.shape != (
                spec.num_envs,
                ACTION_DIM,
            ):
                raise RuntimeError("SEARCH or V6 candidate action violated the 21-D contract")
            if not bool(torch.isfinite(search_action).all()) or not bool(
                torch.isfinite(route_action).all()
            ):
                raise FloatingPointError("candidate policy produced NaN or infinity")
            trigger = active_before & readiness["trigger"]
            if bool(trigger.any()):
                features = build_public_route_feature(
                    observation=observation,
                    ready_count_after=readiness["ready_count_after"],
                    fork_used_before=readiness["fork_used_before"],
                    episode_step=episode_step,
                    public_safety=safety_state,
                    search_action=search_action,
                    route_action=route_action,
                )
                candidate_feature[trigger] = features[trigger]
                candidate_score[trigger] = readiness["score"][trigger]
                candidate_step[trigger] = episode_step[trigger]
                candidate_stratum[trigger] = readiness["stratum"][trigger]
                candidate_seen |= trigger
            ready_count = torch.where(
                active_before,
                readiness["ready_count_after"],
                torch.zeros_like(ready_count),
            )
            fork_used = torch.where(
                active_before,
                readiness["fork_used_after"],
                torch.zeros_like(fork_used),
            )
            # Treatment is consulted only after both candidate actions and the
            # complete pre-action feature snapshot have been constructed.
            route_active |= trigger & assignment_route
            action = torch.where(route_active.unsqueeze(-1), route_action, search_action)
            action = torch.where(active_before.unsqueeze(-1), action, torch.zeros_like(action))
            next_observation, reward, terminated, truncated, info = env.step(action)
            vector_steps += 1
            executed = env.last_executed_policy_action
            if executed is None or not torch.equal(executed, action):
                raise RuntimeError("adapter executed an action different from the selected trial arm")
            events = validate_terminal_events(
                info,
                terminated,
                truncated,
                task_mode=FULL_TASK_MODE,
                close_option_confirm_steps=int(cfg.close_option_confirm_steps),
                power_required_other_contacts=int(cfg.power_grasp_required_other_contacts),
                power_grasp_quality_threshold=float(cfg.power_grasp_quality_high),
                close_option_min_hold_quality=float(cfg.close_option_min_hold_quality),
                close_option_safe_force_limit=float(cfg.grasp_bonus_max_force),
            )
            transition_truth = physical_truth_from_terminal_info(
                info,
                num_envs=spec.num_envs,
                device=env.device,
                task_mode=FULL_TASK_MODE,
                close_option_confirm_steps=int(cfg.close_option_confirm_steps),
                close_option_min_hold_quality=float(cfg.close_option_min_hold_quality),
                grasp_quality_threshold=float(cfg.grasp_quality_high),
                safe_force_limit=float(cfg.grasp_bonus_max_force),
            )
            accepted_done = active_before & (terminated | truncated)
            post_reset_truth = _read_physical_truth(env.unwrapped, task_mode=FULL_TASK_MODE)
            records_before = len(tracker.records)
            tracker.step(
                reward=reward.to(dtype=torch.float32),
                terminated=terminated,
                truncated=truncated,
                events=events,
                transition_truth=transition_truth,
                post_reset_truth=post_reset_truth,
            )
            for record in tracker.records[records_before:]:
                env_id = int(record["env_slot"])
                if bool(candidate_seen[env_id]):
                    rows.append(
                        {
                            "feature": candidate_feature[env_id].detach().cpu().clone(),
                            "env_slot": env_id,
                            "slot_episode_index": int(record["slot_episode_index"]),
                            "episode_index": int(record["episode_index"]),
                            "candidate_step": int(candidate_step[env_id].item()),
                            "stratum": int(candidate_stratum[env_id].item()),
                            "readiness_score": float(candidate_score[env_id].item()),
                            "factual_treatment_route": bool(
                                assignment_route[env_id].item()
                            ),
                            "outcome_episode_length": int(record["length"]),
                            "outcome_max_true_clearance_m": float(
                                record["max_true_clearance_m"]
                            ),
                            "outcome_terminated": bool(
                                record["success"] or record["failure"]
                            ),
                            "outcome_truncated": bool(record["time_out"]),
                            "outcome_success": bool(record["success"]),
                            "outcome_failure": bool(record["failure"]),
                            "outcome_time_out": bool(record["time_out"]),
                            "outcome_ever_grasped": bool(record["ever_grasped"]),
                            "outcome_ever_clearance_ge_20cm": bool(
                                record["ever_clearance_ge_20cm"]
                            ),
                            "outcome_dropped": bool(record["dropped"]),
                            "outcome_unsafe_force": bool(record["unsafe_force"]),
                            "outcome_unlatched_clearance_ge_5cm": bool(
                                record["ever_unlatched_clearance_ge_5cm"]
                            ),
                        }
                    )
            next_step = episode_step + active_before.long()
            episode_step = torch.where(accepted_done, torch.zeros_like(next_step), next_step)
            ready_count.masked_fill_(accepted_done, 0)
            fork_used.masked_fill_(accepted_done, False)
            route_active.masked_fill_(accepted_done, False)
            safety_state = public_gate_feature_tensor(
                info[PUBLIC_GATE_STATE_EXTRAS_KEY],
                num_envs=spec.num_envs,
                device=env.device,
            )
            observation = next_observation

        if not tracker.complete:
            raise RuntimeError(
                f"trial completed {len(tracker.records)}/{spec.num_envs} first episodes"
            )
        validate_public_gate_feature_values(safety_state)
        if len(tracker.records) != spec.num_envs:
            raise RuntimeError("trial did not record exactly one episode per environment slot")
        if sha256_file(spec.manifest_path) != spec.manifest_sha256:
            raise RuntimeError("trial manifest changed during collection")
        if sha256_file(spec.search_checkpoint) != spec.search_checkpoint_sha256:
            raise RuntimeError("SEARCH checkpoint changed during collection")
        for path, expected in (
            (spec.route_checkpoint / "actor.pt", spec.route_actor_sha256),
            (
                spec.route_checkpoint / "task_contract.json",
                spec.route_task_contract_sha256,
            ),
            (
                spec.route_checkpoint / "frozen_lift_actor.pt",
                spec.route_frozen_actor_sha256,
            ),
            (
                spec.route_checkpoint / "torch_bridge_state.pt",
                spec.route_bridge_state_sha256,
            ),
        ):
            if sha256_file(path) != expected:
                raise RuntimeError("V6 checkpoint changed during collection")
        source_after, runtime_assets_after, git_after = source_provenance(spec)
        if (
            source_after != source_hashes
            or runtime_assets_after != runtime_asset_hashes
            or git_after != git
        ):
            raise RuntimeError("trial execution source changed during collection")

        provenance = {
            "kind": "pick_tool_public_route_randomized_factual_trial_v2",
            "format_version": 2,
            "cohort": spec.cohort,
            "seed": spec.seed,
            "replicate": spec.replicate,
            "num_envs": spec.num_envs,
            "episodes": spec.num_envs,
            "manifest_path": str(spec.manifest_path),
            "manifest_sha256": spec.manifest_sha256,
            "assignment_algorithm": spec.manifest["assignment"]["algorithm"],
            "assignment_salt": spec.assignment_salt,
            "known_route_propensity": 0.5,
            "pairing_semantics": spec.manifest["claim_boundary"]["pairing_semantics"],
            "causal_counterfactual_claim_allowed": False,
            "feature_contract": spec.manifest["feature_contract"],
            "gate_contract": spec.manifest["gate"],
            "task_contract": spec.manifest["task"],
            "search_checkpoint": str(spec.search_checkpoint),
            "search_checkpoint_sha256": spec.search_checkpoint_sha256,
            "route_checkpoint": str(spec.route_checkpoint),
            "route_actor_sha256": spec.route_actor_sha256,
            "route_task_contract_sha256": spec.route_task_contract_sha256,
            "route_frozen_actor_sha256": spec.route_frozen_actor_sha256,
            "route_bridge_state_sha256": spec.route_bridge_state_sha256,
            "route_load": route_load,
            "source_sha256": source_hashes,
            "runtime_asset_sha256": runtime_asset_hashes,
            "git": git,
            "runtime": runtime_provenance(spec.seed, device=env.device),
            "collection_semantics": (
                "first episode per slot; SEARCH before first public trigger; "
                "preregistered factual treatment thereafter"
            ),
        }
        artifact = build_randomized_trial_artifact(
            rows,
            metadata={
                "seed": spec.seed,
                "num_envs": spec.num_envs,
                "assignment_salt": spec.assignment_salt,
                "replicate": spec.replicate,
                "provenance": provenance,
            },
            assignment_route=assignment_cpu,
        )
        route_rows = [row for row in rows if bool(row["factual_treatment_route"])]
        continue_rows = [row for row in rows if not bool(row["factual_treatment_route"])]
        record_by_slot = {int(row["env_slot"]): row for row in tracker.records}
        assigned_route_records = [
            record_by_slot[index]
            for index in range(spec.num_envs)
            if bool(assignment_cpu[index])
        ]
        assigned_continue_records = [
            record_by_slot[index]
            for index in range(spec.num_envs)
            if not bool(assignment_cpu[index])
        ]
        report = {
            "kind": COLLECTOR_REPORT_KIND,
            "status": "complete",
            "cohort": spec.cohort,
            "seed": spec.seed,
            "replicate": spec.replicate,
            "num_envs": spec.num_envs,
            "vector_steps": vector_steps,
            "candidate_rows": len(rows),
            "untriggered_episodes": spec.num_envs - len(rows),
            "candidate_route_rows": len(route_rows),
            "candidate_continue_rows": len(continue_rows),
            "candidate_route_outcomes": _event_summary(route_rows),
            "candidate_continue_outcomes": _event_summary(continue_rows),
            "all_assigned_route_outcomes": _event_summary(assigned_route_records),
            "all_assigned_continue_outcomes": _event_summary(assigned_continue_records),
            "assignment_route_slots": int(assignment_cpu.sum()),
            "assignment_continue_slots": int((~assignment_cpu).sum()),
            "manifest_sha256": spec.manifest_sha256,
            "source_sha256": source_hashes,
            "runtime_asset_sha256": runtime_asset_hashes,
            "git": git,
            "claim_boundary": spec.manifest["claim_boundary"],
        }
        return artifact, report
    finally:
        env.close()


def _path_owned(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def parse_args() -> tuple[argparse.Namespace, CollectionSpec, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cohort", choices=("pilot", "train", "development"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--replicate", choices=("a", "b"), required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    spec = load_collection_spec(
        DEFAULT_MANIFEST,
        cohort=args.cohort,
        seed=args.seed,
        replicate=args.replicate,
    )
    try:
        outputs = validate_preregistered_run_order(spec)
    except (FileExistsError, RuntimeError) as error:
        parser.error(str(error))
    args.artifact_output = outputs.artifact
    args.report_output = outputs.report
    launcher = AppLauncher(args)
    return args, spec, launcher.app


def main() -> None:
    args, spec, simulation_app = parse_args()
    try:
        from public_route_trial_contract import (
            publish_json_no_clobber,
            publish_trial_and_report_no_clobber,
        )

        try:
            artifact, report = run_collection(
                spec, device_string=str(args.device or "cuda:0")
            )
            artifact_sha = publish_trial_and_report_no_clobber(
                artifact,
                report,
                artifact_output=args.artifact_output,
                report_output=args.report_output,
            )
            print(
                "[public-route-trial] "
                f"cohort={spec.cohort} seed={spec.seed} replicate={spec.replicate} "
                f"rows={report['candidate_rows']} sha256={artifact_sha}",
                flush=True,
            )
        except Exception as error:
            failure_report = next_failure_report_path(spec)
            if not _path_owned(args.report_output):
                publish_json_no_clobber(
                    {
                        "kind": COLLECTOR_REPORT_KIND,
                        "status": "failed",
                        "cohort": spec.cohort,
                        "seed": spec.seed,
                        "replicate": spec.replicate,
                        "manifest_sha256": spec.manifest_sha256,
                        "canonical_artifact_output": str(args.artifact_output),
                        "canonical_report_output": str(args.report_output),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                    failure_report,
                )
            raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
