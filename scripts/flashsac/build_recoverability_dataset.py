#!/usr/bin/env python3
"""Build audited SEARCH-versus-FlashSAC handoff outcome pairs.

The two treatments are independent Isaac GPU rollouts, not simulator-state
forks.  A row is therefore a strong diagnostic pair only when both shadow
gates fire on the same zero-based step and the public observation plus both
deterministic candidate actions agree within explicit tolerances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


HANDOFF_KIND = "pick_tool_diagnostic_search_to_flashsac_handoff_v1"
DATASET_KIND = "pick_tool_recoverability_pairs_v1"
OBSERVATION_DIM = 115
ACTION_DIM = 21
LATCH_INDEX = 106

VECTOR_FIELDS = {
    "observation": OBSERVATION_DIM,
    "search_action": ACTION_DIM,
    "flashsac_action": ACTION_DIM,
}
LONG_FIELDS = (
    "env_slot",
    "slot_episode_index",
    "episode_index",
    "handoff_step",
    "outcome_episode_length",
)
FLOAT_FIELDS = (
    "pregrasp_score",
    "proximity_quality",
    "max_force_n",
    "true_clearance_m",
    "outcome_max_true_clearance_m",
)
BOOL_FIELDS = (
    "outcome_success",
    "outcome_failure",
    "outcome_time_out",
    "outcome_ever_grasped",
    "outcome_ever_clearance_ge_5cm",
    "outcome_ever_clearance_ge_20cm",
    "outcome_dropped",
    "outcome_unsafe_force",
    "outcome_ever_unlatched_clearance_ge_5cm",
    "outcome_ever_post_candidate_latch",
)
PROVENANCE_KEYS = (
    "task_mode",
    "observation_contract",
    "observation_dim",
    "action_dim",
    "seed",
    "requested_episodes",
    "num_envs",
    "episode_length_s",
    "max_episode_steps",
    "deterministic_policy_actions",
    "use_compile",
    "approach_checkpoint_sha256",
    "flashsac_actor_sha256",
    "flashsac_task_contract_sha256",
    "flashsac_fork_commit",
    "flashsac_upstream_commit",
    "source_sha256",
    "supervisor",
    "selector_eligible_fields",
    "audit_only_private_fields",
    "outcome_semantics",
)
CANONICAL_PROVENANCE_KEYS = tuple(
    key for key in PROVENANCE_KEYS if key != "seed"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _atomic_json(payload: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"report already exists: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary report already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _publish_dataset_and_report(
    payload: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Stage both files before no-clobber publication, rolling back a partial link."""

    for path, label in ((output, "dataset"), (report_path, "report")):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"{label} output already exists: {path}")
    dataset_temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    report_temporary = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    for path in (dataset_temporary, report_temporary):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"temporary output already exists: {path}")
    published_dataset = False
    try:
        with dataset_temporary.open("xb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        dataset_sha256 = sha256_file(dataset_temporary)
        final_report = {
            **dict(report),
            "dataset": str(output),
            "dataset_sha256": dataset_sha256,
        }
        with report_temporary.open("x", encoding="utf-8") as stream:
            json.dump(final_report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(dataset_temporary, output)
        published_dataset = True
        try:
            os.link(report_temporary, report_path)
        except BaseException:
            output.unlink()
            published_dataset = False
            raise
        return final_report
    finally:
        for path in (dataset_temporary, report_temporary):
            if path.exists() or path.is_symlink():
                path.unlink()
        # This branch is only reachable if an unexpected exception happened
        # after the dataset link but before the report link/rollback block.
        if published_dataset and not report_path.exists():
            output.unlink()


def _require_tensor(
    payload: Mapping[str, Any],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    source: str,
) -> torch.Tensor:
    value = payload.get(name)
    if not isinstance(value, torch.Tensor) or value.shape != shape:
        actual = tuple(value.shape) if isinstance(value, torch.Tensor) else None
        raise ValueError(f"{source} {name} shape must be {shape}, got {actual}")
    if value.dtype != dtype:
        raise TypeError(f"{source} {name} dtype must be {dtype}, got {value.dtype}")
    if value.device.type != "cpu":
        raise ValueError(f"{source} {name} must be a CPU tensor")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{source} {name} contains NaN or infinity")
    return value.contiguous()


def validate_handoff_artifact(
    payload: Mapping[str, Any],
    *,
    source: str,
    expected_treatment: str | None = None,
) -> dict[str, Any]:
    """Validate one weights-only handoff artifact and return a plain copy."""

    if not isinstance(payload, Mapping):
        raise TypeError(f"{source} payload must be a mapping")
    if payload.get("kind") != HANDOFF_KIND or payload.get("format_version") != 1:
        raise ValueError(f"{source} is not a supported handoff artifact")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{source} metadata must be a mapping")
    metadata = dict(metadata)
    json.dumps(metadata, sort_keys=True, allow_nan=False)
    treatment = metadata.get("treatment")
    if treatment not in {"handoff_to_flashsac", "continue_search"}:
        raise ValueError(f"{source} has invalid treatment={treatment!r}")
    if expected_treatment is not None and treatment != expected_treatment:
        raise ValueError(
            f"{source} treatment is {treatment!r}, expected {expected_treatment!r}"
        )
    missing_meta = sorted(set(PROVENANCE_KEYS) - set(metadata))
    if missing_meta:
        raise KeyError(f"{source} metadata is missing fields: {missing_meta}")
    if metadata["task_mode"] != "full_task":
        raise ValueError(f"{source} must use full_task semantics")
    if metadata["observation_contract"] != "pick_tool_markov115_v1":
        raise ValueError(f"{source} has an unsupported observation contract")
    if int(metadata["observation_dim"]) != OBSERVATION_DIM:
        raise ValueError(f"{source} observation dimension mismatch")
    if int(metadata["action_dim"]) != ACTION_DIM:
        raise ValueError(f"{source} action dimension mismatch")
    if not bool(metadata["deterministic_policy_actions"]):
        raise ValueError(f"{source} must contain deterministic policy actions")
    if int(metadata["requested_episodes"]) > int(metadata["num_envs"]):
        raise ValueError(f"{source} violates the one-reset-trajectory-per-slot contract")
    if not math.isclose(float(metadata["episode_length_s"]), 20.0, abs_tol=1e-9):
        raise ValueError(f"{source} must use the native 20 s horizon")
    if int(metadata["max_episode_steps"]) != 1000:
        raise ValueError(f"{source} must use the native 1000-step horizon")
    if metadata["selector_eligible_fields"] != [
        "observation",
        "search_action",
        "flashsac_action",
    ]:
        raise ValueError(f"{source} changes selector-eligible feature semantics")
    source_sha256 = metadata["source_sha256"]
    if not isinstance(source_sha256, Mapping) or not source_sha256:
        raise TypeError(f"{source} source_sha256 must be a non-empty mapping")
    for relative, digest in source_sha256.items():
        if (
            not isinstance(relative, str)
            or not relative
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{source} has an invalid source SHA-256 entry")
    for name in (
        "approach_checkpoint_sha256",
        "flashsac_actor_sha256",
        "flashsac_task_contract_sha256",
    ):
        digest = metadata[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{source} has an invalid {name}")

    rows = int(metadata.get("handoff_rows", -1))
    if rows < 0:
        raise ValueError(f"{source} has an invalid handoff row count")
    normalized: dict[str, Any] = {
        "kind": payload["kind"],
        "format_version": 1,
        "metadata": metadata,
    }
    for name, width in VECTOR_FIELDS.items():
        normalized[name] = _require_tensor(
            payload,
            name,
            shape=(rows, width),
            dtype=torch.float32,
            source=source,
        )
    for name in LONG_FIELDS:
        normalized[name] = _require_tensor(
            payload,
            name,
            shape=(rows,),
            dtype=torch.long,
            source=source,
        )
    for name in FLOAT_FIELDS:
        normalized[name] = _require_tensor(
            payload,
            name,
            shape=(rows,),
            dtype=torch.float32,
            source=source,
        )
    for name in BOOL_FIELDS:
        normalized[name] = _require_tensor(
            payload,
            name,
            shape=(rows,),
            dtype=torch.bool,
            source=source,
        )

    keys = list(
        zip(
            normalized["env_slot"].tolist(),
            normalized["slot_episode_index"].tolist(),
        )
    )
    if len(keys) != len(set(keys)):
        raise ValueError(f"{source} contains duplicate episode keys")
    if rows and (
        bool((normalized["env_slot"] < 0).any())
        or bool((normalized["env_slot"] >= int(metadata["num_envs"])).any())
    ):
        raise ValueError(f"{source} env_slot is outside the configured vector width")
    if bool((normalized["slot_episode_index"] != 0).any()):
        raise ValueError(f"{source} expected only the first episode in each env slot")
    if bool((normalized["observation"][:, LATCH_INDEX] != 0.0).any()):
        raise ValueError(f"{source} contains a latched candidate observation")
    terminal_count = (
        normalized["outcome_success"].long()
        + normalized["outcome_failure"].long()
        + normalized["outcome_time_out"].long()
    )
    if not bool((terminal_count == 1).all()):
        raise ValueError(f"{source} terminal outcomes are not mutually exhaustive")
    success = normalized["outcome_success"]
    if bool(success.any()) and (
        not bool(normalized["outcome_ever_grasped"][success].all())
        or not bool(normalized["outcome_ever_clearance_ge_20cm"][success].all())
        or bool((normalized["outcome_max_true_clearance_m"][success] < 0.20).any())
        or bool(normalized["outcome_dropped"][success].any())
        or bool(normalized["outcome_unsafe_force"][success].any())
    ):
        raise ValueError(f"{source} strict success violates physical truth")
    return normalized


def validate_companion_metrics(
    artifact: Mapping[str, Any],
    *,
    artifact_path: Path,
    artifact_sha256: str,
    source: str,
) -> dict[str, Any]:
    """Cross-check every captured row against the evaluator's JSON evidence."""

    metadata = artifact["metadata"]
    metrics_path = Path(str(metadata.get("hierarchy_metrics_output", "")))
    if not metrics_path.is_file() or metrics_path.is_symlink():
        raise FileNotFoundError(f"{source} companion metrics are unavailable: {metrics_path}")
    metrics_sha256 = sha256_file(metrics_path)
    with metrics_path.open("r", encoding="utf-8") as stream:
        metrics = json.load(stream)
    if sha256_file(metrics_path) != metrics_sha256:
        raise RuntimeError(f"{source} companion metrics changed while being read")
    if not isinstance(metrics, Mapping):
        raise TypeError(f"{source} companion metrics must be a JSON object")
    json.dumps(metrics, sort_keys=True, allow_nan=False)
    hierarchy = metrics.get("diagnostic_approach_hierarchy")
    if not isinstance(hierarchy, Mapping):
        raise KeyError(f"{source} companion metrics lack hierarchy evidence")
    entry = hierarchy.get("handoff_artifact")
    if not isinstance(entry, Mapping):
        raise KeyError(f"{source} companion metrics lack handoff artifact provenance")
    rows = int(metadata["handoff_rows"])
    expected_treatment = str(metadata["treatment"])
    expected_handoff = expected_treatment == "handoff_to_flashsac"
    expected_policy = (
        "diagnostic_frozen_rlgames_search_then_checkpoint_native_flashsac_close_lift"
        if expected_handoff
        else "diagnostic_frozen_rlgames_base_only"
    )
    if metrics.get("status") != "complete" or metrics.get("policy") != expected_policy:
        raise ValueError(f"{source} companion is not a complete expected treatment run")
    if (
        int(metrics.get("completed_episodes", -1))
        != int(metadata["requested_episodes"])
        or int(metrics.get("requested_episodes", -1))
        != int(metadata["requested_episodes"])
    ):
        raise ValueError(f"{source} companion episode quota is incomplete")
    companion_contract = {
        "task_mode": "task_mode",
        "observation_contract": "observation_contract",
        "observation_dim": "observation_dim",
        "action_dim": "action_dim",
        "episode_length_s": "episode_length_s",
        "max_episode_steps": "max_episode_steps",
        "use_compile": "use_compile",
        "flashsac_fork_commit": "flashsac_fork_commit",
        "flashsac_upstream_commit": "flashsac_upstream_commit",
    }
    for metrics_key, metadata_key in companion_contract.items():
        if metrics.get(metrics_key) != metadata.get(metadata_key):
            raise ValueError(
                f"{source} companion {metrics_key} disagrees with artifact metadata"
            )
    if (
        entry.get("sha256") != artifact_sha256
        or entry.get("kind") != HANDOFF_KIND
        or int(entry.get("rows", -1)) != rows
        or entry.get("treatment") != expected_treatment
    ):
        raise ValueError(f"{source} companion artifact provenance mismatch")
    if int(hierarchy.get("candidate_count", -1)) != rows:
        raise ValueError(f"{source} companion candidate count disagrees with artifact")
    if int(hierarchy.get("handoff_count", -1)) != (rows if expected_handoff else 0):
        raise ValueError(f"{source} companion executed handoff count is inconsistent")
    for key in (
        "seed",
        "num_envs",
        "requested_episodes",
        "max_episode_steps",
        "checkpoint_actor_sha256",
        "checkpoint_task_contract_sha256",
    ):
        expected_key = {
            "checkpoint_actor_sha256": "flashsac_actor_sha256",
            "checkpoint_task_contract_sha256": "flashsac_task_contract_sha256",
        }.get(key, key)
        if metrics.get(key) != metadata.get(expected_key):
            raise ValueError(f"{source} companion {key} disagrees with artifact metadata")
    if hierarchy.get("approach_checkpoint_sha256") != metadata.get(
        "approach_checkpoint_sha256"
    ):
        raise ValueError(f"{source} companion approach checkpoint mismatch")
    if hierarchy.get("supervisor") != metadata.get("supervisor"):
        raise ValueError(f"{source} companion supervisor mismatch")

    episode_records = metrics.get("episodes")
    if not isinstance(episode_records, list):
        raise TypeError(f"{source} companion episodes must be a list")
    if len(episode_records) != int(metadata["requested_episodes"]):
        raise ValueError(f"{source} companion episode record count is incomplete")
    candidate_records: dict[tuple[int, int], Mapping[str, Any]] = {}
    for record in episode_records:
        if not isinstance(record, Mapping):
            raise TypeError(f"{source} companion episode record must be a mapping")
        if bool(record.get("hierarchy_handoff_candidate", False)):
            key = (int(record["env_slot"]), int(record["slot_episode_index"]))
            if key in candidate_records:
                raise ValueError(f"{source} companion has a duplicate candidate key {key}")
            candidate_records[key] = record
    artifact_index = _episode_index(artifact)
    if set(candidate_records) != set(artifact_index):
        raise ValueError(f"{source} companion candidate keys disagree with artifact")
    bool_map = {
        "outcome_success": "success",
        "outcome_failure": "failure",
        "outcome_time_out": "time_out",
        "outcome_ever_grasped": "ever_grasped",
        "outcome_ever_clearance_ge_5cm": "ever_clearance_ge_5cm",
        "outcome_ever_clearance_ge_20cm": "ever_clearance_ge_20cm",
        "outcome_dropped": "dropped",
        "outcome_unsafe_force": "unsafe_force",
        "outcome_ever_unlatched_clearance_ge_5cm": (
            "ever_unlatched_clearance_ge_5cm"
        ),
    }
    for key, row in artifact_index.items():
        record = candidate_records[key]
        if bool(record.get("hierarchy_handoff")) != expected_handoff:
            raise ValueError(f"{source} companion treatment flag mismatch for key {key}")
        scalar_pairs = (
            ("episode_index", "episode_index"),
            ("handoff_step", "hierarchy_handoff_step"),
            ("outcome_episode_length", "length"),
        )
        for artifact_name, record_name in scalar_pairs:
            if int(artifact[artifact_name][row]) != int(record[record_name]):
                raise ValueError(
                    f"{source} companion {record_name} mismatch for key {key}"
                )
        for artifact_name, record_name in bool_map.items():
            if bool(artifact[artifact_name][row]) != bool(record[record_name]):
                raise ValueError(
                    f"{source} companion {record_name} mismatch for key {key}"
                )
        # New evaluator records distinguish a shadow-gate candidate from an
        # actually executed route.  Historical records used the latter name
        # for candidate-relative latch truth, so retain a strict fallback for
        # already-published evidence while preferring the corrected field.
        candidate_latch_name = (
            "hierarchy_ever_post_candidate_latch"
            if "hierarchy_ever_post_candidate_latch" in record
            else "hierarchy_ever_post_handoff_latch"
        )
        if bool(artifact["outcome_ever_post_candidate_latch"][row]) != bool(
            record[candidate_latch_name]
        ):
            raise ValueError(
                f"{source} companion {candidate_latch_name} mismatch for key {key}"
            )
        if not math.isclose(
            float(artifact["outcome_max_true_clearance_m"][row]),
            float(record["max_true_clearance_m"]),
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            raise ValueError(
                f"{source} companion true-clearance maximum mismatch for key {key}"
            )
    return {
        "path": str(metrics_path.resolve()),
        "sha256": metrics_sha256,
        "artifact_path_recorded": str(entry.get("path")),
        "artifact_path_loaded": str(artifact_path.resolve()),
    }


def _episode_index(payload: Mapping[str, Any]) -> dict[tuple[int, int], int]:
    return {
        (int(env_slot), int(slot_episode_index)): row
        for row, (env_slot, slot_episode_index) in enumerate(
            zip(payload["env_slot"].tolist(), payload["slot_episode_index"].tolist())
        )
    }


def pair_handoff_artifacts(
    routed: Mapping[str, Any],
    continued: Mapping[str, Any],
    *,
    max_observation_abs_error: float = 1.0e-5,
    max_action_abs_error: float = 1.0e-5,
    max_score_abs_error: float = 1.0e-5,
    max_proximity_abs_error: float = 1.0e-5,
    max_force_abs_error_n: float = 1.0e-2,
    max_clearance_abs_error_m: float = 1.0e-5,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Pair one seed's routed and continue outcomes with an explicit quality mask."""

    if not math.isfinite(max_observation_abs_error) or max_observation_abs_error < 0:
        raise ValueError("max_observation_abs_error must be finite and non-negative")
    if not math.isfinite(max_action_abs_error) or max_action_abs_error < 0:
        raise ValueError("max_action_abs_error must be finite and non-negative")
    for name, value in (
        ("max_score_abs_error", max_score_abs_error),
        ("max_proximity_abs_error", max_proximity_abs_error),
        ("max_force_abs_error_n", max_force_abs_error_n),
        ("max_clearance_abs_error_m", max_clearance_abs_error_m),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    routed = validate_handoff_artifact(
        routed, source="routed", expected_treatment="handoff_to_flashsac"
    )
    continued = validate_handoff_artifact(
        continued, source="continued", expected_treatment="continue_search"
    )
    routed_meta = routed["metadata"]
    continued_meta = continued["metadata"]
    mismatches = {
        key: (routed_meta.get(key), continued_meta.get(key))
        for key in PROVENANCE_KEYS
        if routed_meta.get(key) != continued_meta.get(key)
    }
    if mismatches:
        raise ValueError(f"paired artifact provenance mismatch: {mismatches}")

    routed_index = _episode_index(routed)
    continued_index = _episode_index(continued)
    common_keys = sorted(set(routed_index) & set(continued_index))
    routed_only = sorted(set(routed_index) - set(continued_index))
    continued_only = sorted(set(continued_index) - set(routed_index))
    route_rows = torch.tensor(
        [routed_index[key] for key in common_keys], dtype=torch.long
    )
    continue_rows = torch.tensor(
        [continued_index[key] for key in common_keys], dtype=torch.long
    )
    count = len(common_keys)

    paired: dict[str, torch.Tensor] = {
        "seed": torch.full((count,), int(routed_meta["seed"]), dtype=torch.long),
        "env_slot": torch.tensor([key[0] for key in common_keys], dtype=torch.long),
        "slot_episode_index": torch.tensor(
            [key[1] for key in common_keys], dtype=torch.long
        ),
    }
    for name in VECTOR_FIELDS:
        paired[f"route_{name}"] = routed[name][route_rows].clone()
        paired[f"continue_{name}"] = continued[name][continue_rows].clone()
    for name in ("episode_index", "handoff_step", "outcome_episode_length"):
        paired[f"route_{name}"] = routed[name][route_rows].clone()
        paired[f"continue_{name}"] = continued[name][continue_rows].clone()
    for name in FLOAT_FIELDS:
        paired[f"route_{name}"] = routed[name][route_rows].clone()
        paired[f"continue_{name}"] = continued[name][continue_rows].clone()
    for name in BOOL_FIELDS:
        output_name = name.removeprefix("outcome_")
        paired[f"route_{output_name}"] = routed[name][route_rows].clone()
        paired[f"continue_{output_name}"] = continued[name][continue_rows].clone()

    paired["same_handoff_step"] = (
        paired["route_handoff_step"] == paired["continue_handoff_step"]
    )
    for name in VECTOR_FIELDS:
        delta = (paired[f"route_{name}"] - paired[f"continue_{name}"]).abs()
        paired[f"{name}_max_abs_error"] = (
            delta.amax(dim=1) if count else torch.empty(0, dtype=torch.float32)
        )
        paired[f"{name}_mean_abs_error"] = (
            delta.mean(dim=1) if count else torch.empty(0, dtype=torch.float32)
        )
    for name in (
        "pregrasp_score",
        "proximity_quality",
        "max_force_n",
        "true_clearance_m",
    ):
        error = (paired[f"route_{name}"] - paired[f"continue_{name}"]).abs()
        paired[f"{name}_abs_error"] = error
    paired["strong_pair"] = (
        paired["same_handoff_step"]
        & (paired["observation_max_abs_error"] <= max_observation_abs_error)
        & (paired["search_action_max_abs_error"] <= max_action_abs_error)
        & (paired["flashsac_action_max_abs_error"] <= max_action_abs_error)
        & (paired["pregrasp_score_abs_error"] <= max_score_abs_error)
        & (paired["proximity_quality_abs_error"] <= max_proximity_abs_error)
        & (paired["max_force_n_abs_error"] <= max_force_abs_error_n)
        & (paired["true_clearance_m_abs_error"] <= max_clearance_abs_error_m)
    )
    # Canonical selector inputs always come from the untreated SEARCH arm.
    # Strong pairs make the choice immaterial, while this convention prevents
    # input semantics from depending on CLI ordering.
    for name in VECTOR_FIELDS:
        paired[name] = paired[f"continue_{name}"].clone()
    paired["strict_success_delta"] = (
        paired["route_success"].to(torch.int8)
        - paired["continue_success"].to(torch.int8)
    )
    paired["observed_advantage"] = paired["strict_success_delta"].clone()
    paired["discordant"] = paired["strict_success_delta"] != 0
    paired["route_label"] = paired["strict_success_delta"] == 1
    paired["preference_label_valid"] = paired["strong_pair"] & paired["discordant"]
    paired["preference_sample_weight"] = paired[
        "preference_label_valid"
    ].to(torch.float32)
    paired["paired_outcome_label_valid"] = paired["strong_pair"].clone()
    paired["observed_oracle_success"] = (
        paired["route_success"] | paired["continue_success"]
    )

    strong = paired["strong_pair"]
    summary = {
        "seed": int(routed_meta["seed"]),
        "route_candidates": len(routed_index),
        "continue_candidates": len(continued_index),
        "common_candidates": count,
        "route_only_candidates": len(routed_only),
        "continue_only_candidates": len(continued_only),
        "strong_pairs": int(strong.sum()),
        "rejected_common_pairs": int((~strong).sum()),
        "same_step_pairs": int(paired["same_handoff_step"].sum()),
        "strong_route_success": int((paired["route_success"] & strong).sum()),
        "strong_continue_success": int((paired["continue_success"] & strong).sum()),
        "strong_route_only_gain": int(
            ((paired["observed_advantage"] == 1) & strong).sum()
        ),
        "strong_route_regression": int(
            ((paired["observed_advantage"] == -1) & strong).sum()
        ),
        "max_observation_abs_error": (
            float(paired["observation_max_abs_error"].max()) if count else None
        ),
        "max_search_action_abs_error": (
            float(paired["search_action_max_abs_error"].max()) if count else None
        ),
        "max_flashsac_action_abs_error": (
            float(paired["flashsac_action_max_abs_error"].max()) if count else None
        ),
        "max_pregrasp_score_abs_error": (
            float(paired["pregrasp_score_abs_error"].max()) if count else None
        ),
        "max_proximity_quality_abs_error": (
            float(paired["proximity_quality_abs_error"].max()) if count else None
        ),
        "max_force_abs_error_n": (
            float(paired["max_force_n_abs_error"].max()) if count else None
        ),
        "max_true_clearance_abs_error_m": (
            float(paired["true_clearance_m_abs_error"].max()) if count else None
        ),
    }
    return paired, summary


def _concat_batches(batches: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batches:
        raise ValueError("at least one paired artifact batch is required")
    keys = set(batches[0])
    if any(set(batch) != keys for batch in batches[1:]):
        raise ValueError("paired batches expose different tensor fields")
    merged = {
        key: torch.cat([batch[key] for batch in batches], dim=0)
        for key in sorted(keys)
    }
    if merged["seed"].numel() < 1:
        raise ValueError("paired artifacts have no common handoff candidates")
    order = sorted(
        range(merged["seed"].numel()),
        key=lambda row: (
            int(merged["seed"][row]),
            int(merged["env_slot"][row]),
            int(merged["slot_episode_index"][row]),
        ),
    )
    index = torch.tensor(order, dtype=torch.long)
    return {key: value[index].contiguous() for key, value in merged.items()}


def build_dataset(
    pairs: Sequence[tuple[Path, Path]],
    *,
    max_observation_abs_error: float,
    max_action_abs_error: float,
    max_score_abs_error: float,
    max_proximity_abs_error: float,
    max_force_abs_error_n: float,
    max_clearance_abs_error_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    batches: list[dict[str, torch.Tensor]] = []
    pair_reports: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    canonical_provenance: dict[str, Any] | None = None
    for route_path, continue_path in pairs:
        for path in (route_path, continue_path):
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"artifact must be a regular non-symlink file: {path}")
        route_sha256 = sha256_file(route_path)
        continue_sha256 = sha256_file(continue_path)
        routed = torch.load(route_path, map_location="cpu", weights_only=True)
        continued = torch.load(continue_path, map_location="cpu", weights_only=True)
        if (
            sha256_file(route_path) != route_sha256
            or sha256_file(continue_path) != continue_sha256
        ):
            raise RuntimeError("handoff artifact changed while it was being loaded")
        batch, pair_report = pair_handoff_artifacts(
            routed,
            continued,
            max_observation_abs_error=max_observation_abs_error,
            max_action_abs_error=max_action_abs_error,
            max_score_abs_error=max_score_abs_error,
            max_proximity_abs_error=max_proximity_abs_error,
            max_force_abs_error_n=max_force_abs_error_n,
            max_clearance_abs_error_m=max_clearance_abs_error_m,
        )
        current_provenance = {
            key: routed["metadata"][key] for key in CANONICAL_PROVENANCE_KEYS
        }
        if canonical_provenance is None:
            canonical_provenance = current_provenance
        elif current_provenance != canonical_provenance:
            mismatches = {
                key: (canonical_provenance.get(key), current_provenance.get(key))
                for key in CANONICAL_PROVENANCE_KEYS
                if canonical_provenance.get(key) != current_provenance.get(key)
            }
            raise ValueError(f"cross-seed provenance mismatch: {mismatches}")
        route_companion = validate_companion_metrics(
            routed,
            artifact_path=route_path,
            artifact_sha256=route_sha256,
            source="routed",
        )
        continue_companion = validate_companion_metrics(
            continued,
            artifact_path=continue_path,
            artifact_sha256=continue_sha256,
            source="continued",
        )
        seed = int(pair_report["seed"])
        if seed in seen_seeds:
            raise ValueError(f"duplicate seed group in recoverability dataset: {seed}")
        seen_seeds.add(seed)
        pair_report.update(
            {
                "route_artifact": str(route_path.resolve()),
                "route_artifact_sha256": route_sha256,
                "continue_artifact": str(continue_path.resolve()),
                "continue_artifact_sha256": continue_sha256,
                "route_companion_metrics": route_companion,
                "continue_companion_metrics": continue_companion,
            }
        )
        batches.append(batch)
        pair_reports.append(pair_report)
    tensors = _concat_batches(batches)
    feature_digests: dict[str, tuple[int, int, int]] = {}
    for row in range(int(tensors["seed"].numel())):
        digest = hashlib.sha256()
        for name in ("observation", "search_action", "flashsac_action"):
            canonical = tensors[name][row].contiguous().clone()
            canonical[canonical == 0.0] = 0.0
            digest.update(canonical.numpy().tobytes())
        semantic_sha256 = digest.hexdigest()
        key = (
            int(tensors["seed"][row]),
            int(tensors["env_slot"][row]),
            int(tensors["slot_episode_index"][row]),
        )
        previous = feature_digests.get(semantic_sha256)
        if previous is not None and previous[0] != key[0]:
            raise ValueError(
                "exact selector feature collision across held-out seed groups: "
                f"{previous} and {key}"
            )
        feature_digests[semantic_sha256] = key
    strong = tensors["strong_pair"]
    if not bool(strong.any()):
        raise ValueError("recoverability dataset has no strong audited pairs")
    metadata = {
        "pairing_semantics": "independent_gpu_rollout_diagnostic_v1",
        "causal_counterfactual_claim_allowed": False,
        "strong_pair_semantics": (
            "same handoff step and matching public observation plus deterministic "
            "SEARCH/FlashSAC candidate actions and audit-only gate truth"
        ),
        "max_observation_abs_error": max_observation_abs_error,
        "max_action_abs_error": max_action_abs_error,
        "max_score_abs_error": max_score_abs_error,
        "max_proximity_abs_error": max_proximity_abs_error,
        "max_force_abs_error_n": max_force_abs_error_n,
        "max_clearance_abs_error_m": max_clearance_abs_error_m,
        "seeds": sorted(seen_seeds),
        "rows": int(strong.numel()),
        "strong_rows": int(strong.sum()),
        "rejected_common_rows": int((~strong).sum()),
        "strong_discordant_rows": int((tensors["discordant"] & strong).sum()),
        "strong_route_only_gain": int(
            ((tensors["observed_advantage"] == 1) & strong).sum()
        ),
        "strong_route_regression": int(
            ((tensors["observed_advantage"] == -1) & strong).sum()
        ),
        "feature_policy": (
            "training may use only canonical observation/search_action/flashsac_action "
            "from the continue_search arm; direct preference uses "
            "preference_label_valid, while dual outcome heads use "
            "paired_outcome_label_valid; audit scalars and outcomes are forbidden "
            "as features"
        ),
        "strict_success_semantics": (
            "reset-before terminal truth with real convex-mesh clearance >= 0.20 m"
        ),
        "source_pairs": pair_reports,
        "canonical_provenance": canonical_provenance,
        "builder_source_sha256": sha256_file(Path(__file__).resolve()),
        "selector_input_fields": [
            "observation",
            "search_action",
            "flashsac_action",
        ],
        "preference_label_contract": {
            "target": "route_label",
            "valid_mask": "preference_label_valid",
            "sample_weight": "preference_sample_weight",
            "tie_semantics": "invalid_for_direct_preference",
        },
        "dual_outcome_label_contract": {
            "targets": ["continue_success", "route_success"],
            "valid_mask": "paired_outcome_label_valid",
            "tie_semantics": "valid_for_two_independent_outcome_heads",
        },
        "forbidden_selector_feature_patterns": [
            "*_success",
            "*_failure",
            "*_time_out",
            "*_dropped",
            "*_unsafe_force",
            "*clearance*",
            "*pregrasp_score*",
            "*proximity_quality*",
            "*max_force*",
            "*handoff_step*",
            "seed",
            "env_slot",
            "slot_episode_index",
        ],
    }
    payload: dict[str, Any] = {
        "kind": DATASET_KIND,
        "format_version": 1,
        "metadata": metadata,
        **tensors,
    }
    report = {
        "status": "complete",
        "kind": DATASET_KIND,
        **{key: value for key, value in metadata.items() if key != "source_pairs"},
        "source_pairs": pair_reports,
    }
    return payload, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("ROUTED_PT", "CONTINUE_PT"),
        type=Path,
        required=True,
    )
    parser.add_argument("--max_observation_abs_error", type=float, default=1.0e-5)
    parser.add_argument("--max_action_abs_error", type=float, default=1.0e-5)
    parser.add_argument("--max_score_abs_error", type=float, default=1.0e-5)
    parser.add_argument("--max_proximity_abs_error", type=float, default=1.0e-5)
    parser.add_argument("--max_force_abs_error_n", type=float, default=1.0e-2)
    parser.add_argument("--max_clearance_abs_error_m", type=float, default=1.0e-5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, label in ((args.output, "dataset"), (args.report, "report")):
        if path.is_symlink():
            raise FileExistsError(f"{label} output must not be a symlink: {path}")
        if path.exists():
            raise FileExistsError(f"{label} output already exists: {path}")
    for route, continued in args.pair:
        if route.is_symlink() or continued.is_symlink():
            raise FileNotFoundError("handoff artifact inputs must not be symlinks")
    output = args.output.resolve()
    report_path = args.report.resolve()
    if output == report_path:
        raise ValueError("dataset and report outputs must differ")
    payload, report = build_dataset(
        [(route.resolve(), continued.resolve()) for route, continued in args.pair],
        max_observation_abs_error=args.max_observation_abs_error,
        max_action_abs_error=args.max_action_abs_error,
        max_score_abs_error=args.max_score_abs_error,
        max_proximity_abs_error=args.max_proximity_abs_error,
        max_force_abs_error_n=args.max_force_abs_error_n,
        max_clearance_abs_error_m=args.max_clearance_abs_error_m,
    )
    final_report = _publish_dataset_and_report(
        payload,
        report,
        output=output,
        report_path=report_path,
    )
    print(json.dumps(final_report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
