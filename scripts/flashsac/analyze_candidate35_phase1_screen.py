#!/usr/bin/env python3
"""Fail-closed, simulation-free screen for Candidate 35 phase one.

The immutable manifest fixes model identities, six complementary 64-env
trials, estimators, gates, and output before collection.  This screen is a
small engineering gate, not a replacement for the sealed 1024-env formal A/B
analysis used for Candidate 34.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

from online_close_ab import (
    ARTIFACT_KIND,
    ASSIGNMENT_CONTRACT,
    ASSIGNMENT_SALT,
    COLLECTION_CONTRACT,
    FORMAT_VERSION as ARTIFACT_FORMAT_VERSION,
    HANDOFF_HOLD_STEPS,
    HANDOFF_MIN_SCORE,
    KIT_ARGS,
    REPORT_KIND as COLLECTION_REPORT_KIND,
    SOURCE_PATH_COUNT,
    SOURCE_PATH_SET_SHA256,
    assignment_candidate_mask,
    publish_json_no_clobber,
    sha256_file,
    validate_artifact,
    validate_report,
)


MANIFEST_KIND = "pick_tool_candidate35_phase1_screen_manifest_v1"
REPORT_KIND = "pick_tool_candidate35_phase1_screen_v1"
SEEDS = (320, 321, 322)
REPLICATES = ("a", "b")
RUN_ORDER = ("320a", "320b", "321b", "321a", "322a", "322b")
NUM_ENVS = 64
PLAN_FIELD = "phase1_plan"
PLAN_SHA_FIELD = "phase1_plan_sha256"
PLAN_LABEL = "phase-one plan"
OUTPUT_PATTERN = (
    "logs/flashsac/pick_tool/41_close_ab_c35_p1_s{seed}_{replicate}/trial"
)
ANALYSIS_OUTPUT = "logs/flashsac/pick_tool/41_close_ab_c35_p1_screen.json"
PROTECTED_ANALYZER_PATHS = (
    "scripts/flashsac/analyze_candidate35_phase1_screen.py",
)
PASS_DECISION = "continue_phase2"
FAIL_DECISION = "reject_phase2"
EVENTS = (
    "success",
    "ever_grasped",
    "dropped",
    "unsafe_force",
    "unlatched_clearance_ge_5cm",
)
EXPECTED_GATES = {
    "conditional_success_delta_min": 0.03,
    "itt_success_delta_min": 0.01,
    "conditional_ever_grasped_delta_min": 0.0,
    "positive_conditional_success_seeds_min": 2,
    "per_seed_conditional_success_delta_floor": -0.1,
    "candidate_triggered_dropped_max": 0,
    "candidate_triggered_unsafe_force_max": 0,
    "conditional_unlatched_clearance_ge_5cm_delta_max": 0.02,
}


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {token}")
            ),
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    return payload


def _read_regular(path: Path, *, label: str) -> bytes:
    path = Path(os.path.abspath(os.fspath(path)))
    try:
        path.relative_to(_root())
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is not a regular non-symlink file: {path}")
    return path.read_bytes()


def _sha(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=_root(), text=True
    ).strip()


def _resolve(relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(relative)
    if not path.is_absolute():
        path = _root() / path
    path = Path(os.path.abspath(path))
    try:
        path.relative_to(_root())
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    return path


def _checkpoint_hash(path: Path, filename: str) -> str:
    target = path / filename
    if target.is_symlink() or not target.is_file():
        raise FileNotFoundError(target)
    return sha256_file(target)


def validate_manifest_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_top = {
        "kind",
        "status",
        PLAN_FIELD,
        PLAN_SHA_FIELD,
        "branch",
        "collector",
        "baseline",
        "candidate",
        "search",
        "run_order",
        "output_pattern",
        "estimator",
        "gates",
        "analysis_output",
    }
    if not isinstance(payload, dict) or set(payload) != expected_top:
        raise ValueError("screen manifest top-level schema changed")
    if payload["kind"] != MANIFEST_KIND or payload["status"] != "sealed_before_collection":
        raise ValueError("screen manifest is not sealed")
    if payload["branch"] != "flashsac-pick-tool-curriculum":
        raise ValueError("screen branch changed")
    if payload["run_order"] != list(RUN_ORDER):
        raise ValueError("screen run order changed")
    if payload["output_pattern"] != OUTPUT_PATTERN:
        raise ValueError("screen evidence path pattern changed")
    if payload["gates"] != EXPECTED_GATES:
        raise ValueError("screen gates changed")
    if payload["estimator"] != {
        "name": "equal_seed_horvitz_thompson",
        "known_propensity": 0.5,
        "contrast": "candidate_minus_baseline",
        "seed_weighting": "equal_arithmetic_mean",
    }:
        raise ValueError("screen estimator changed")

    collector = payload["collector"]
    expected_collector = {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "report_kind": COLLECTION_REPORT_KIND,
        "collection_contract": COLLECTION_CONTRACT,
        "assignment_contract": ASSIGNMENT_CONTRACT,
        "assignment_salt": ASSIGNMENT_SALT,
        "kit_args": KIT_ARGS,
        "num_envs_per_run": NUM_ENVS,
        "source_path_count": SOURCE_PATH_COUNT,
        "source_path_set_sha256": SOURCE_PATH_SET_SHA256,
        "require_identical_source_manifest_across_runs": True,
        "git_branch": "flashsac-pick-tool-curriculum",
        "flashsac_upstream_commit": "87edc9061150ae9e962dd84e6544e27a1554b3ab",
        "flashsac_fork_commit": "5ecf331fa11cd457dd39018b3d68af571b257666",
        "common_frozen_lift_semantic_sha256": (
            "8af0fdbc2572346b7519fbb4b29354fb5bc882a0c0020c8c3ace8376696ab324"
        ),
        "common_frozen_lift_source_actor_sha256": (
            "7757869eaa1df02f5f52c2dcd1353fb4a486b7512019650a8234d76702b9a1fb"
        ),
    }
    if collector != expected_collector:
        raise ValueError("screen collector identity changed")

    required_model = {
        "checkpoint",
        "actor_sha256",
        "task_contract_sha256",
        "torch_bridge_state_sha256",
        "frozen_lift_actor_sha256",
    }
    for arm in ("baseline", "candidate"):
        model = payload[arm]
        if not isinstance(model, dict) or set(model) != required_model:
            raise ValueError(f"screen {arm} schema changed")
        for key in required_model - {"checkpoint"}:
            _sha(model[key], name=f"{arm}.{key}")
    if payload["baseline"]["actor_sha256"] == payload["candidate"]["actor_sha256"]:
        raise ValueError("screen actors are identical")
    if payload["baseline"]["task_contract_sha256"] != payload["candidate"][
        "task_contract_sha256"
    ] or payload["baseline"]["frozen_lift_actor_sha256"] != payload["candidate"][
        "frozen_lift_actor_sha256"
    ]:
        raise ValueError("screen task or frozen LIFT differs between arms")
    search = payload["search"]
    if set(search) != {"checkpoint", "sha256", "min_score", "hold_steps"}:
        raise ValueError("screen SEARCH schema changed")
    _sha(search["sha256"], name="search.sha256")
    if search["min_score"] != HANDOFF_MIN_SCORE or search["hold_steps"] != HANDOFF_HOLD_STEPS:
        raise ValueError("screen SEARCH handoff changed")
    _sha(payload[PLAN_SHA_FIELD], name=PLAN_SHA_FIELD)
    if payload["analysis_output"] != ANALYSIS_OUTPUT:
        raise ValueError("screen analysis output changed")
    return dict(payload)


def load_manifest(path: Path) -> tuple[dict[str, Any], str, str]:
    path = _resolve(os.fspath(path), label="screen manifest")
    raw = _read_regular(path, label="screen manifest")
    payload = validate_manifest_payload(_strict_json_bytes(raw, label="screen manifest"))
    digest = hashlib.sha256(raw).hexdigest()
    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    if branch != payload["branch"]:
        raise ValueError("current branch differs from screen manifest")
    relative = path.relative_to(_root()).as_posix()
    committed = subprocess.check_output(
        ("git", "show", f"HEAD:{relative}"), cwd=_root()
    )
    if committed != raw:
        raise ValueError("screen manifest differs from its committed HEAD blob")
    protected = (
        relative,
        *PROTECTED_ANALYZER_PATHS,
        "scripts/flashsac/collect_online_close_ab.py",
        "scripts/flashsac/online_close_ab.py",
    )
    if subprocess.run(
        ("git", "diff", "--quiet", "HEAD", "--", *protected),
        cwd=_root(),
        check=False,
    ).returncode:
        raise ValueError("screen or collection source is dirty")
    plan = _resolve(payload[PLAN_FIELD], label=PLAN_LABEL)
    if sha256_file(plan) != payload[PLAN_SHA_FIELD]:
        raise ValueError(f"{PLAN_LABEL} changed")

    for arm in ("baseline", "candidate"):
        checkpoint = _resolve(payload[arm]["checkpoint"], label=f"{arm} checkpoint")
        for filename, key in (
            ("actor.pt", "actor_sha256"),
            ("task_contract.json", "task_contract_sha256"),
            ("torch_bridge_state.pt", "torch_bridge_state_sha256"),
            ("frozen_lift_actor.pt", "frozen_lift_actor_sha256"),
        ):
            if _checkpoint_hash(checkpoint, filename) != payload[arm][key]:
                raise ValueError(f"{arm} checkpoint {filename} changed")
    search = _resolve(payload["search"]["checkpoint"], label="SEARCH checkpoint")
    if sha256_file(search) != payload["search"]["sha256"]:
        raise ValueError("SEARCH checkpoint changed")
    return payload, digest, head


def _torch_load(raw: bytes) -> Any:
    try:
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("weights_only PyTorch loading is required") from error


def _semantic_sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_key(value: str) -> tuple[int, str]:
    return int(value[:-1]), value[-1]


def load_evidence(
    manifest: Mapping[str, Any], *, expected_git_commit: str
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    loaded: dict[tuple[int, str], dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    reference_identity: dict[str, Any] | None = None
    for token in RUN_ORDER:
        seed, replicate = _run_key(token)
        stem = manifest["output_pattern"].format(seed=seed, replicate=replicate)
        artifact_path = _resolve(f"{stem}.pt", label=f"{token} artifact")
        report_path = _resolve(f"{stem}.json", label=f"{token} report")
        artifact_raw = _read_regular(artifact_path, label=f"{token} artifact")
        artifact_sha = hashlib.sha256(artifact_raw).hexdigest()
        artifact = validate_artifact(_torch_load(artifact_raw))
        report_raw = _read_regular(report_path, label=f"{token} report")
        report = _strict_json_bytes(report_raw, label=f"{token} report")
        validate_report(report, artifact, published=True)
        if report["artifact_sha256"] != artifact_sha:
            raise ValueError(f"{token} report receipt is stale")
        if Path(os.path.abspath(report["artifact_output"])) != artifact_path:
            raise ValueError(f"{token} report points at another artifact")

        metadata = artifact["metadata"]
        fixed = {
            "kind": ARTIFACT_KIND,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "collection_contract": COLLECTION_CONTRACT,
            "assignment_contract": ASSIGNMENT_CONTRACT,
            "assignment_salt": ASSIGNMENT_SALT,
            "kit_args": KIT_ARGS,
            "handoff_min_score": HANDOFF_MIN_SCORE,
            "handoff_hold_steps": HANDOFF_HOLD_STEPS,
            "seed": seed,
            "replicate": replicate,
            "num_envs": NUM_ENVS,
            "baseline_actor_sha256": manifest["baseline"]["actor_sha256"],
            "candidate_actor_sha256": manifest["candidate"]["actor_sha256"],
            "baseline_task_contract_sha256": manifest["baseline"][
                "task_contract_sha256"
            ],
            "candidate_task_contract_sha256": manifest["candidate"][
                "task_contract_sha256"
            ],
            "baseline_bridge_state_sha256": manifest["baseline"][
                "torch_bridge_state_sha256"
            ],
            "candidate_bridge_state_sha256": manifest["candidate"][
                "torch_bridge_state_sha256"
            ],
            "common_frozen_lift_actor_sha256": manifest["baseline"][
                "frozen_lift_actor_sha256"
            ],
            "common_frozen_lift_semantic_sha256": manifest["collector"][
                "common_frozen_lift_semantic_sha256"
            ],
            "common_frozen_lift_source_actor_sha256": manifest["collector"][
                "common_frozen_lift_source_actor_sha256"
            ],
            "search_checkpoint_sha256": manifest["search"]["sha256"],
            "flashsac_upstream_commit": manifest["collector"][
                "flashsac_upstream_commit"
            ],
            "flashsac_fork_commit": manifest["collector"]["flashsac_fork_commit"],
        }
        for key, expected in fixed.items():
            if metadata.get(key) != expected:
                raise ValueError(f"{token} metadata {key!r} changed")
        git = metadata["git"]
        if git != {
            "commit": expected_git_commit,
            "branch": manifest["collector"]["git_branch"],
            "source_files_dirty": False,
            "flashsac_commit": manifest["collector"]["flashsac_fork_commit"],
            "flashsac_dirty": False,
        }:
            raise ValueError(f"{token} Git identity changed")
        source = metadata["source_sha256"]
        if len(source) != SOURCE_PATH_COUNT or hashlib.sha256(
            "\0".join(sorted(source)).encode("utf-8")
        ).hexdigest() != SOURCE_PATH_SET_SHA256:
            raise ValueError(f"{token} source path contract changed")
        runtime = dict(metadata["runtime"])
        runtime.pop("seed")
        identity = {
            "source": _semantic_sha(source),
            "assets": _semantic_sha(metadata["runtime_asset_sha256"]),
            "runtime": _semantic_sha(runtime),
        }
        if reference_identity is None:
            reference_identity = identity
        elif identity != reference_identity:
            raise ValueError("source, assets, or runtime differ across screen runs")
        key = (seed, replicate)
        loaded[key] = artifact
        receipts.append(
            {
                "run": token,
                "artifact": str(artifact_path),
                "artifact_sha256": artifact_sha,
                "report": str(report_path),
                "report_sha256": hashlib.sha256(report_raw).hexdigest(),
            }
        )

    for seed in SEEDS:
        a = loaded[(seed, "a")]["tensors"]
        b = loaded[(seed, "b")]["tensors"]
        if not torch.equal(a["env_slot"], b["env_slot"]):
            raise ValueError(f"seed {seed} slot identities differ")
        if not torch.equal(a["assignment_candidate"], ~b["assignment_candidate"]):
            raise ValueError(f"seed {seed} assignments are not complementary")
        for replicate, tensors in (("a", a), ("b", b)):
            expected = assignment_candidate_mask(
                seed=seed, num_envs=NUM_ENVS, replicate=replicate
            )
            if not torch.equal(tensors["assignment_candidate"], expected):
                raise ValueError(f"seed {seed}{replicate} assignment changed")

    cube = {
        name: torch.stack(
            [
                torch.stack([loaded[(seed, replicate)]["tensors"][name] for replicate in REPLICATES])
                for seed in SEEDS
            ]
        )
        for name in ("assignment_candidate", "triggered", *EVENTS)
    }
    return cube, receipts


def _ht_event(
    assignment: torch.Tensor,
    domain: torch.Tensor,
    event: torch.Tensor,
) -> dict[str, float | int]:
    denominator = int(domain.sum())
    candidate_rows = int((assignment & domain).sum())
    baseline_rows = int(((~assignment) & domain).sum())
    if denominator <= 0 or candidate_rows <= 0 or baseline_rows <= 0:
        raise ValueError("screen seed lacks triggered support in one randomized arm")
    candidate_count = int((assignment & domain & event).sum())
    baseline_count = int(((~assignment) & domain & event).sum())
    candidate = 2.0 * candidate_count / denominator
    baseline = 2.0 * baseline_count / denominator
    return {
        "candidate": candidate,
        "baseline": baseline,
        "delta": candidate - baseline,
        "candidate_count": candidate_count,
        "baseline_count": baseline_count,
        "domain_rows": denominator,
        "candidate_rows": candidate_rows,
        "baseline_rows": baseline_rows,
    }


def compute_screen(cube: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    expected_shape = (len(SEEDS), len(REPLICATES), NUM_ENVS)
    required = {"assignment_candidate", "triggered", *EVENTS}
    if set(cube) != required:
        raise ValueError("screen cube schema changed")
    for name, tensor in cube.items():
        if tensor.shape != expected_shape or tensor.dtype != torch.bool:
            raise ValueError(f"screen cube {name} must be bool{expected_shape}")
    per_seed: dict[str, Any] = {}
    for index, seed in enumerate(SEEDS):
        assignment = cube["assignment_candidate"][index]
        if not torch.equal(assignment[0], ~assignment[1]):
            raise ValueError(f"seed {seed} cube assignments are not complementary")
        triggered = cube["triggered"][index]
        conditional = {
            event: _ht_event(assignment, triggered, cube[event][index])
            for event in EVENTS
        }
        all_rows = torch.ones_like(triggered)
        itt_success = _ht_event(
            assignment, all_rows, cube["success"][index]
        )
        per_seed[str(seed)] = {
            "conditional_triggered": conditional,
            "itt_success": itt_success,
        }

    def mean_delta(section: str, event: str | None = None) -> float:
        values = []
        for seed in SEEDS:
            row = per_seed[str(seed)][section]
            if event is not None:
                row = row[event]
            values.append(float(row["delta"]))
        return sum(values) / len(values)

    aggregate = {
        "conditional_success_delta": mean_delta("conditional_triggered", "success"),
        "itt_success_delta": mean_delta("itt_success"),
        "conditional_ever_grasped_delta": mean_delta(
            "conditional_triggered", "ever_grasped"
        ),
        "conditional_unlatched_clearance_ge_5cm_delta": mean_delta(
            "conditional_triggered", "unlatched_clearance_ge_5cm"
        ),
        "positive_conditional_success_seeds": sum(
            per_seed[str(seed)]["conditional_triggered"]["success"]["delta"] > 0.0
            for seed in SEEDS
        ),
        "minimum_seed_conditional_success_delta": min(
            per_seed[str(seed)]["conditional_triggered"]["success"]["delta"]
            for seed in SEEDS
        ),
        "candidate_triggered_dropped": int(
            (
                cube["assignment_candidate"]
                & cube["triggered"]
                & cube["dropped"]
            ).sum()
        ),
        "candidate_triggered_unsafe_force": int(
            (
                cube["assignment_candidate"]
                & cube["triggered"]
                & cube["unsafe_force"]
            ).sum()
        ),
    }
    gates = {
        "conditional_success_delta": {
            "value": aggregate["conditional_success_delta"],
            "threshold": EXPECTED_GATES["conditional_success_delta_min"],
            "comparison": ">=",
            "pass": aggregate["conditional_success_delta"]
            >= EXPECTED_GATES["conditional_success_delta_min"],
        },
        "itt_success_delta": {
            "value": aggregate["itt_success_delta"],
            "threshold": EXPECTED_GATES["itt_success_delta_min"],
            "comparison": ">=",
            "pass": aggregate["itt_success_delta"]
            >= EXPECTED_GATES["itt_success_delta_min"],
        },
        "conditional_ever_grasped_delta": {
            "value": aggregate["conditional_ever_grasped_delta"],
            "threshold": EXPECTED_GATES["conditional_ever_grasped_delta_min"],
            "comparison": ">=",
            "pass": aggregate["conditional_ever_grasped_delta"]
            >= EXPECTED_GATES["conditional_ever_grasped_delta_min"],
        },
        "positive_conditional_success_seeds": {
            "value": aggregate["positive_conditional_success_seeds"],
            "threshold": EXPECTED_GATES["positive_conditional_success_seeds_min"],
            "comparison": ">=",
            "pass": aggregate["positive_conditional_success_seeds"]
            >= EXPECTED_GATES["positive_conditional_success_seeds_min"],
        },
        "minimum_seed_conditional_success_delta": {
            "value": aggregate["minimum_seed_conditional_success_delta"],
            "threshold": EXPECTED_GATES[
                "per_seed_conditional_success_delta_floor"
            ],
            "comparison": ">=",
            "pass": aggregate["minimum_seed_conditional_success_delta"]
            >= EXPECTED_GATES["per_seed_conditional_success_delta_floor"],
        },
        "candidate_triggered_dropped": {
            "value": aggregate["candidate_triggered_dropped"],
            "threshold": EXPECTED_GATES["candidate_triggered_dropped_max"],
            "comparison": "<=",
            "pass": aggregate["candidate_triggered_dropped"]
            <= EXPECTED_GATES["candidate_triggered_dropped_max"],
        },
        "candidate_triggered_unsafe_force": {
            "value": aggregate["candidate_triggered_unsafe_force"],
            "threshold": EXPECTED_GATES["candidate_triggered_unsafe_force_max"],
            "comparison": "<=",
            "pass": aggregate["candidate_triggered_unsafe_force"]
            <= EXPECTED_GATES["candidate_triggered_unsafe_force_max"],
        },
        "conditional_unlatched_clearance_ge_5cm_delta": {
            "value": aggregate["conditional_unlatched_clearance_ge_5cm_delta"],
            "threshold": EXPECTED_GATES[
                "conditional_unlatched_clearance_ge_5cm_delta_max"
            ],
            "comparison": "<=",
            "pass": aggregate["conditional_unlatched_clearance_ge_5cm_delta"]
            <= EXPECTED_GATES[
                "conditional_unlatched_clearance_ge_5cm_delta_max"
            ],
        },
    }
    return {
        "per_seed": per_seed,
        "aggregate": aggregate,
        "gates": gates,
        "all_gates_pass": all(item["pass"] for item in gates.values()),
    }


def analyze(manifest_path: Path) -> tuple[dict[str, Any], Path]:
    manifest, manifest_sha, head = load_manifest(manifest_path)
    cube, receipts = load_evidence(manifest, expected_git_commit=head)
    screen = compute_screen(cube)
    report = {
        "kind": REPORT_KIND,
        "status": "complete",
        "manifest": str(_resolve(os.fspath(manifest_path), label="screen manifest")),
        "manifest_sha256": manifest_sha,
        "git_commit": head,
        "baseline_actor_sha256": manifest["baseline"]["actor_sha256"],
        "candidate_actor_sha256": manifest["candidate"]["actor_sha256"],
        "receipts": receipts,
        **screen,
        "decision": (
            PASS_DECISION if screen["all_gates_pass"] else FAIL_DECISION
        ),
    }
    for value in report["aggregate"].values():
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError("screen produced a non-finite result")
    output = _resolve(manifest["analysis_output"], label="analysis output")
    return report, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report, output = analyze(args.manifest)
    publish_json_no_clobber(report, output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
