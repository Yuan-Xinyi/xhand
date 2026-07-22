#!/usr/bin/env python3
"""One-shot preregistered development audit for the public-route gate.

Only randomized factual outcomes are used.  The primary estimand is the
Horvitz--Thompson (known propensity 1/2) difference between the frozen gate
and fixed V6 routing.  A train-frozen ensemble supplies a doubly-robust
diagnostic, but that diagnostic can never change acceptance.  Inference uses
the exact seed/slot cluster bootstrap and balanced complementary-label Monte
Carlo permutation scheme sealed in ``public_route_gate_analysis_plan.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import torch

from public_route_gate_dataset import (
    DEFAULT_ANALYSIS_PLAN,
    DEFAULT_MANIFEST,
    _require_nonsymlink_chain,
    load_analysis_plan,
    load_registered_factual_cohort,
    semantic_dataset_sha256,
    sha256_file,
    validate_collection_acceptance,
    validate_factual_dataset,
)
from public_route_gate_runtime import (
    PublicRouteGateRuntime,
    load_frozen_public_route_gate,
)
from train_public_route_gate import (
    analysis_seal_provenance,
    load_tagged_evidence_ledger,
)


AUDIT_KIND = "pick_tool_public_route_gate_development_audit_v1"
EVENTS = ("success", "dropped", "unsafe_force")
EVENT_COLUMNS = {name: index for index, name in enumerate(EVENTS)}
PRODUCTION_REPLICATES = 20_000
PRODUCTION_LOWER_RANK = 500
PRODUCTION_UPPER_RANK = 19_500
PRODUCTION_NUM_ENVS = 512
PRODUCTION_ROUTE_SLOTS = 256
_MONTE_CARLO_BATCH = 256
AUDIT_SOURCE_FILES = (
    "scripts/flashsac/public_route_gate_runtime.py",
    "scripts/flashsac/audit_public_route_gate_development.py",
)


def _hash_seed(manifest_sha256: str, label: str) -> int:
    if (
        not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in manifest_sha256)
    ):
        raise ValueError("randomization seed requires a lowercase SHA256 receipt")
    digest = hashlib.sha256(
        manifest_sha256.encode("ascii") + b"\0" + label.encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _validate_production_analysis_contract(plan: Mapping[str, Any]) -> None:
    """Reject any weakening of the sealed development inference choices."""

    inference = plan.get("randomization_inference")
    expected_keys = {
        "bootstrap",
        "cluster_key",
        "events_reported_separately",
        "permutation",
        "safety_events_do_not_cancel",
    }
    if not isinstance(inference, dict) or set(inference) != expected_keys:
        raise ValueError("development randomization-inference schema is not exact")
    bootstrap = inference.get("bootstrap")
    if not isinstance(bootstrap, dict) or set(bootstrap) != {
        "ci",
        "replicates",
        "resampling_unit",
        "seed",
    }:
        raise ValueError("development bootstrap schema is not exact")
    if (
        bootstrap["replicates"] != PRODUCTION_REPLICATES
        or bootstrap["ci"]
        != "sort 20000 estimates ascending and use fixed one-based order-statistic ranks 500 and 19500"
        or bootstrap["seed"]
        != "unsigned big-endian integer from the first 8 bytes of SHA256(trial_manifest_sha256 + NUL + development_cluster_bootstrap_v1)"
    ):
        raise ValueError("development bootstrap choices drifted")
    permutation = inference.get("permutation")
    if not isinstance(permutation, dict) or set(permutation) != {
        "p_value",
        "replicates",
        "seed",
        "swap",
    }:
        raise ValueError("development permutation schema is not exact")
    if (
        permutation["replicates"] != PRODUCTION_REPLICATES
        or permutation["p_value"]
        != "two-sided plus-one Monte Carlo: (1 + count(abs(permuted)>=abs(observed))) / (replicates + 1)"
        or permutation["seed"]
        != "unsigned big-endian integer from the first 8 bytes of SHA256(trial_manifest_sha256 + NUL + development_cluster_permutation_v1)"
    ):
        raise ValueError("development permutation choices drifted")
    if (
        inference["cluster_key"] != ["seed", "env_slot"]
        or inference["events_reported_separately"] != list(EVENTS)
        or inference["safety_events_do_not_cancel"] is not True
    ):
        raise ValueError("development cluster/event contract drifted")
    development = plan.get("development")
    if not isinstance(development, dict):
        raise ValueError("analysis plan lacks development choices")
    if development.get("estimators") != {
        "fixed_route": "mean(2 * I(factual_route) * outcome)",
        "frozen_gate": "mean(2 * I(factual_arm equals frozen_gate_decision) * outcome)",
        "known_propensity": 0.5,
    }:
        raise ValueError("development Horvitz--Thompson estimators drifted")
    if development.get("doubly_robust_nuisance") != (
        "frozen five-member ensemble probability mean for each event and arm, "
        "clipped to [1e-6,1-1e-6]"
    ):
        raise ValueError("development doubly-robust nuisance contract drifted")


def _event_matrix(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    values = torch.stack(tuple(tensors[name] for name in EVENTS), dim=1)
    if values.dtype != torch.bool or values.device.type != "cpu":
        raise TypeError("development factual events must be CPU bool tensors")
    return values.to(torch.float64)


def _validate_analysis_vectors(
    treatment_route: torch.Tensor,
    gate_continue: torch.Tensor,
    outcomes: torch.Tensor,
) -> int:
    if (
        treatment_route.ndim != 1
        or treatment_route.dtype != torch.bool
        or treatment_route.device.type != "cpu"
    ):
        raise TypeError("factual treatment must be a CPU bool vector")
    rows = int(treatment_route.numel())
    if (
        gate_continue.shape != (rows,)
        or gate_continue.dtype != torch.bool
        or gate_continue.device.type != "cpu"
    ):
        raise TypeError("gate decision must be a matching CPU bool vector")
    if (
        outcomes.shape != (rows, len(EVENTS))
        or outcomes.device.type != "cpu"
        or not outcomes.dtype.is_floating_point
        or not bool(torch.isfinite(outcomes).all())
        or bool(((outcomes < 0.0) | (outcomes > 1.0)).any())
    ):
        raise ValueError("event matrix must be finite CPU [N,3] indicators")
    if rows < 1:
        raise ValueError("development audit requires at least one factual row")
    return rows


def _horvitz_thompson(
    treatment_route: torch.Tensor,
    gate_continue: torch.Tensor,
    outcomes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return row effects and the gate/fixed-route HT point values."""

    _validate_analysis_vectors(treatment_route, gate_continue, outcomes)
    gate_route = ~gate_continue
    gate_match = treatment_route == gate_route
    gate_rows = 2.0 * gate_match[:, None].to(torch.float64) * outcomes
    fixed_rows = 2.0 * treatment_route[:, None].to(torch.float64) * outcomes
    effects = gate_rows - fixed_rows
    return effects, gate_rows.mean(dim=0), fixed_rows.mean(dim=0)


def _doubly_robust_effects(
    treatment_route: torch.Tensor,
    gate_continue: torch.Tensor,
    outcomes: torch.Tensor,
    ensemble_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Return train-nuisance DR row effects; never used for acceptance."""

    rows = _validate_analysis_vectors(treatment_route, gate_continue, outcomes)
    if (
        ensemble_probabilities.shape != (5, rows, 6)
        or ensemble_probabilities.device.type != "cpu"
        or not ensemble_probabilities.dtype.is_floating_point
        or not bool(torch.isfinite(ensemble_probabilities).all())
    ):
        raise ValueError("DR nuisance probabilities must be finite CPU [5,N,6]")
    nuisance = ensemble_probabilities.to(torch.float64).mean(dim=0).clamp(
        1.0e-6, 1.0 - 1.0e-6
    )
    continue_mean = nuisance[:, :3]
    route_mean = nuisance[:, 3:]
    treatment_mean = torch.where(
        treatment_route[:, None], route_mean, continue_mean
    )
    gate_mean = torch.where(gate_continue[:, None], continue_mean, route_mean)
    gate_match = treatment_route == ~gate_continue
    gate_value = gate_mean + 2.0 * gate_match[:, None].to(torch.float64) * (
        outcomes - treatment_mean
    )
    fixed_value = route_mean + 2.0 * treatment_route[:, None].to(torch.float64) * (
        outcomes - route_mean
    )
    result = gate_value - fixed_value
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("doubly-robust estimator produced NaN or infinity")
    return result


def _cluster_bootstrap_samples(
    row_effects: torch.Tensor,
    seed: torch.Tensor,
    env_slot: torch.Tensor,
    *,
    seed_order: Sequence[int],
    replicates: int,
    random_seed: int,
    batch_size: int = _MONTE_CARLO_BATCH,
) -> torch.Tensor:
    """Resample observed ``(seed, slot)`` clusters within each seed."""

    rows = int(row_effects.shape[0])
    if (
        row_effects.shape != (rows, len(EVENTS))
        or row_effects.device.type != "cpu"
        or not row_effects.dtype.is_floating_point
        or not bool(torch.isfinite(row_effects).all())
    ):
        raise ValueError("bootstrap row effects must be finite CPU [N,3]")
    for name, value in (("seed", seed), ("env_slot", env_slot)):
        if value.shape != (rows,) or value.dtype != torch.long or value.device.type != "cpu":
            raise TypeError(f"bootstrap {name} must be a matching CPU long vector")
    if type(replicates) is not int or replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("bootstrap batch size must be positive")
    if not seed_order or len(set(seed_order)) != len(seed_order):
        raise ValueError("bootstrap seed order must be non-empty and unique")
    if set(seed.tolist()) != set(seed_order):
        raise ValueError("bootstrap rows differ from the registered seed order")

    numerator = torch.zeros((replicates, len(EVENTS)), dtype=torch.float64)
    denominator = torch.zeros(replicates, dtype=torch.float64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))
    for registered_seed in seed_order:
        mask = seed == int(registered_seed)
        slots, inverse = torch.unique(env_slot[mask], sorted=True, return_inverse=True)
        clusters = int(slots.numel())
        if clusters < 1:
            raise ValueError(f"development seed {registered_seed} has no observed cluster")
        cluster_sum = torch.zeros((clusters, len(EVENTS)), dtype=torch.float64)
        cluster_sum.index_add_(0, inverse, row_effects[mask].to(torch.float64))
        cluster_size = torch.zeros(clusters, dtype=torch.float64)
        cluster_size.index_add_(
            0, inverse, torch.ones(int(mask.sum()), dtype=torch.float64)
        )
        for start in range(0, replicates, batch_size):
            stop = min(start + batch_size, replicates)
            sampled = torch.randint(
                clusters,
                (stop - start, clusters),
                generator=generator,
                dtype=torch.long,
            )
            numerator[start:stop] += cluster_sum[sampled].sum(dim=1)
            denominator[start:stop] += cluster_size[sampled].sum(dim=1)
    if bool((denominator <= 0.0).any()):
        raise RuntimeError("cluster bootstrap produced an empty resample")
    result = numerator / denominator[:, None]
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("cluster bootstrap produced NaN or infinity")
    return result


def _fixed_rank_intervals(
    samples: torch.Tensor, *, lower_rank: int, upper_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if samples.ndim != 2 or samples.shape[1] != len(EVENTS):
        raise ValueError("interval samples must have shape [B,3]")
    replicates = int(samples.shape[0])
    if not (1 <= lower_rank <= upper_rank <= replicates):
        raise ValueError("fixed bootstrap ranks are outside the sample")
    ordered = torch.sort(samples.to(torch.float64), dim=0).values
    return ordered[lower_rank - 1], ordered[upper_rank - 1]


def _permutation_samples(
    outcomes: torch.Tensor,
    gate_continue: torch.Tensor,
    seed: torch.Tensor,
    env_slot: torch.Tensor,
    replicate_index: torch.Tensor,
    *,
    seed_order: Sequence[int],
    num_envs: int,
    route_slots: int,
    replicates: int,
    random_seed: int,
    batch_size: int = _MONTE_CARLO_BATCH,
) -> torch.Tensor:
    """Balanced a-label randomization with b as its exact complement."""

    rows = int(outcomes.shape[0])
    dummy_treatment = torch.zeros(rows, dtype=torch.bool)
    _validate_analysis_vectors(dummy_treatment, gate_continue, outcomes)
    for name, value in (
        ("seed", seed),
        ("env_slot", env_slot),
        ("replicate_index", replicate_index),
    ):
        if value.shape != (rows,) or value.dtype != torch.long or value.device.type != "cpu":
            raise TypeError(f"permutation {name} must be a matching CPU long vector")
    if (
        type(num_envs) is not int
        or type(route_slots) is not int
        or num_envs < 2
        or route_slots <= 0
        or route_slots >= num_envs
        or bool(((env_slot < 0) | (env_slot >= num_envs)).any())
        or bool(((replicate_index < 0) | (replicate_index > 1)).any())
    ):
        raise ValueError("permutation slot/replicate contract is invalid")
    if type(replicates) is not int or replicates < 1 or batch_size < 1:
        raise ValueError("permutation replicate/batch counts must be positive")
    if set(seed.tolist()) != set(seed_order):
        raise ValueError("permutation rows differ from the registered seed order")

    # Only rows where the gate continues differ from fixed route.  Replicate a
    # contributes +Y and b contributes -Y for the same candidate route-a bit.
    numerator = torch.zeros((replicates, len(EVENTS)), dtype=torch.float64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))
    for registered_seed in seed_order:
        mask = seed == int(registered_seed)
        coefficient = torch.zeros((num_envs, len(EVENTS)), dtype=torch.float64)
        signed_replicate = torch.where(
            replicate_index[mask] == 0,
            torch.ones(int(mask.sum()), dtype=torch.float64),
            -torch.ones(int(mask.sum()), dtype=torch.float64),
        )
        row_coefficient = (
            outcomes[mask].to(torch.float64)
            * gate_continue[mask, None].to(torch.float64)
            * signed_replicate[:, None]
        )
        coefficient.index_add_(0, env_slot[mask], row_coefficient)
        for start in range(0, replicates, batch_size):
            stop = min(start + batch_size, replicates)
            batch = stop - start
            # Ordering i.i.d. continuous uniforms gives a uniform fixed-size
            # subset; float64 makes ties negligible and topk enforces exactly k.
            scores = torch.rand(
                (batch, num_envs), dtype=torch.float64, generator=generator
            )
            selected = torch.topk(
                scores, k=route_slots, dim=1, largest=False, sorted=False
            ).indices
            route_a = torch.zeros((batch, num_envs), dtype=torch.bool)
            route_a.scatter_(1, selected, True)
            if not bool((route_a.sum(dim=1) == route_slots).all()):
                raise RuntimeError("permutation failed exact balanced assignment")
            sign = 1.0 - 2.0 * route_a.to(torch.float64)
            numerator[start:stop] += 2.0 * (sign @ coefficient)
    result = numerator / float(rows)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("permutation inference produced NaN or infinity")
    return result


def _two_sided_plus_one_pvalues(
    observed: torch.Tensor, samples: torch.Tensor
) -> torch.Tensor:
    if observed.shape != (len(EVENTS),) or samples.ndim != 2 or samples.shape[1] != len(EVENTS):
        raise ValueError("permutation statistic dimensions are invalid")
    exceed = (samples.abs() >= observed.to(torch.float64).abs()[None, :]).sum(dim=0)
    return (1.0 + exceed.to(torch.float64)) / (float(samples.shape[0]) + 1.0)


def _event_report(
    gate_value: torch.Tensor,
    fixed_value: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    permutation_p: torch.Tensor,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, column in EVENT_COLUMNS.items():
        result[name] = {
            "frozen_gate_ht": float(gate_value[column]),
            "fixed_route_ht": float(fixed_value[column]),
            "gate_minus_fixed_route_ht": float(gate_value[column] - fixed_value[column]),
            "cluster_bootstrap_95pct_lower": float(lower[column]),
            "cluster_bootstrap_95pct_upper": float(upper[column]),
            "balanced_permutation_two_sided_p": float(permutation_p[column]),
        }
    return result


def _acceptance(
    *,
    collection_acceptance: Mapping[str, Any],
    continue_rate: float,
    ht_events: Mapping[str, Mapping[str, float]],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = plan["development"]["acceptance"]
    checks = {
        "development_collection_accepted": bool(
            collection_acceptance.get("accepted") is True
        ),
        "continue_rate_min": continue_rate
        >= float(thresholds["continue_rate_min"]),
        "continue_rate_max": continue_rate
        <= float(thresholds["continue_rate_max"]),
        "success_delta_95pct_lower": ht_events["success"][
            "cluster_bootstrap_95pct_lower"
        ]
        >= float(thresholds["success_delta_95pct_lower_min"]),
        "added_drop_95pct_upper": ht_events["dropped"][
            "cluster_bootstrap_95pct_upper"
        ]
        <= float(thresholds["added_drop_95pct_upper_max"]),
        "added_unsafe_force_95pct_upper": ht_events["unsafe_force"][
            "cluster_bootstrap_95pct_upper"
        ]
        <= float(thresholds["added_unsafe_force_95pct_upper_max"]),
    }
    return {
        "thresholds": dict(thresholds),
        "checks": checks,
        "accepted": all(checks.values()),
        "dr_used_for_acceptance": False,
        "permutation_p_used_for_acceptance": False,
        "safety_events_evaluated_separately": True,
    }


def _build_development_audit(
    dataset: Mapping[str, Any],
    runtime: PublicRouteGateRuntime,
    plan: Mapping[str, Any],
    *,
    bootstrap_replicates: int,
    lower_rank: int,
    upper_rank: int,
    permutation_replicates: int,
    permutation_num_envs: int,
    permutation_route_slots: int,
) -> dict[str, Any]:
    """Low-level implementation; tests may supply small Monte Carlo counts."""

    dataset = validate_factual_dataset(dataset, expected_cohort="development")
    runtime.validate_cohort(dataset["metadata"]["cohort"])
    runtime.verify_unchanged()
    if (
        dataset["metadata"]["analysis_plan_sha256"] != plan["_sha256"]
        or dataset["metadata"]["manifest_sha256"]
        != plan["preregistration"]["trial_manifest_sha256"]
        or runtime.analysis_plan_sha256 != plan["_sha256"]
    ):
        raise ValueError("development dataset/model/plan receipts disagree")
    if list(dataset["metadata"]["seeds"]) != plan["development"]["seeds"]:
        raise ValueError("development dataset seed order differs from the plan")
    for receipt in dataset["receipts"]:
        if receipt["num_envs"] != permutation_num_envs:
            raise ValueError("development run scale differs from permutation plan")

    tensors = dataset["tensors"]
    outcomes = _event_matrix(tensors)
    decision = runtime.decide_development(tensors["feature"])
    row_effects, gate_value, fixed_value = _horvitz_thompson(
        tensors["treatment_route"], decision.continue_search, outcomes
    )
    manifest_sha = dataset["metadata"]["manifest_sha256"]
    bootstrap_seed = _hash_seed(manifest_sha, "development_cluster_bootstrap_v1")
    bootstrap_samples = _cluster_bootstrap_samples(
        row_effects,
        tensors["seed"],
        tensors["env_slot"],
        seed_order=dataset["metadata"]["seeds"],
        replicates=bootstrap_replicates,
        random_seed=bootstrap_seed,
    )
    lower, upper = _fixed_rank_intervals(
        bootstrap_samples, lower_rank=lower_rank, upper_rank=upper_rank
    )
    permutation_seed = _hash_seed(manifest_sha, "development_cluster_permutation_v1")
    permutation_samples = _permutation_samples(
        outcomes,
        decision.continue_search,
        tensors["seed"],
        tensors["env_slot"],
        tensors["replicate_index"],
        seed_order=dataset["metadata"]["seeds"],
        num_envs=permutation_num_envs,
        route_slots=permutation_route_slots,
        replicates=permutation_replicates,
        random_seed=permutation_seed,
    )
    observed = row_effects.mean(dim=0)
    if not torch.allclose(
        observed, gate_value - fixed_value, atol=1.0e-15, rtol=0.0
    ):
        raise RuntimeError("HT point estimate decompositions disagree")
    permutation_p = _two_sided_plus_one_pvalues(observed, permutation_samples)
    ht_events = _event_report(
        gate_value, fixed_value, lower, upper, permutation_p
    )
    dr_effects = _doubly_robust_effects(
        tensors["treatment_route"],
        decision.continue_search,
        outcomes,
        decision.probabilities,
    )
    dr = {
        name: {"gate_minus_fixed_route_dr": float(dr_effects[:, column].mean())}
        for name, column in EVENT_COLUMNS.items()
    }
    collection = validate_collection_acceptance(dataset)
    continue_rows = int(decision.continue_search.sum())
    rows = int(decision.continue_search.numel())
    continue_rate = continue_rows / rows
    acceptance = _acceptance(
        collection_acceptance=collection,
        continue_rate=continue_rate,
        ht_events=ht_events,
        plan=plan,
    )
    accepted = bool(acceptance["accepted"])
    report = {
        "kind": AUDIT_KIND,
        "format_version": 1,
        "status": "pass" if accepted else "fail",
        "overall_pass": accepted,
        "blind_collection_open": accepted,
        "failure_default": "fixed_route_v6",
        "development_disposition": (
            "open_blind_collection"
            if accepted
            else "keep_blind_closed_and_use_fixed_route_v6"
        ),
        "cohort": "development",
        "row_count": rows,
        "cluster_count": len(
            set(zip(tensors["seed"].tolist(), tensors["env_slot"].tolist()))
        ),
        "gate_policy": {
            "continue_rows": continue_rows,
            "route_rows": rows - continue_rows,
            "continue_rate": continue_rate,
            "equality_is_continue": True,
        },
        "collection_acceptance": collection,
        "primary_horvitz_thompson": {
            "known_propensity": 0.5,
            "estimand": "frozen_gate_minus_fixed_route_among_triggered_candidate_rows",
            "events": ht_events,
        },
        "doubly_robust_report_only": {
            "used_for_acceptance": False,
            "nuisance": "frozen_five_member_ensemble_mean_clipped_1e-6",
            "events": dr,
        },
        "randomization_inference": {
            "bootstrap": {
                "replicates": bootstrap_replicates,
                "seed": bootstrap_seed,
                "cluster_key": ["seed", "env_slot"],
                "resampling": "within_seed_observed_clusters_with_replacement",
                "lower_one_based_rank": lower_rank,
                "upper_one_based_rank": upper_rank,
            },
            "permutation": {
                "replicates": permutation_replicates,
                "seed": permutation_seed,
                "num_envs_per_seed": permutation_num_envs,
                "route_a_slots_per_seed": permutation_route_slots,
                "replicate_b_exact_complement": True,
                "p_value": "two_sided_plus_one",
            },
        },
        "acceptance": acceptance,
        "receipts": {
            "analysis_plan_sha256": plan["_sha256"],
            "trial_manifest_sha256": manifest_sha,
            "development_audit_tag": plan["preregistration"][
                "development_audit_tag"
            ],
            "development_dataset_semantic_sha256": semantic_dataset_sha256(dataset),
            "frozen_model_sha256": runtime.model_sha256,
            "frozen_model_semantic_sha256": runtime.model_semantic_sha256,
            "training_report_sha256": runtime.training_report_sha256,
        },
        "runtime": runtime.audit_metadata(),
    }
    json.dumps(report, sort_keys=True, allow_nan=False)
    runtime.verify_unchanged()
    return report


def build_development_audit(
    dataset: Mapping[str, Any],
    runtime: PublicRouteGateRuntime,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Production entry point with no tunable inference arguments."""

    _validate_production_analysis_contract(plan)
    return _build_development_audit(
        dataset,
        runtime,
        plan,
        bootstrap_replicates=PRODUCTION_REPLICATES,
        lower_rank=PRODUCTION_LOWER_RANK,
        upper_rank=PRODUCTION_UPPER_RANK,
        permutation_replicates=PRODUCTION_REPLICATES,
        permutation_num_envs=PRODUCTION_NUM_ENVS,
        permutation_route_slots=PRODUCTION_ROUTE_SLOTS,
    )


def publish_json_no_clobber(
    payload: Mapping[str, Any], output: Path, *, repository_root: Path
) -> None:
    root = Path(os.path.abspath(os.fspath(repository_root)))
    if root.is_symlink() or not root.is_dir():
        raise ValueError("development audit repository root must be a directory")
    output = Path(os.path.abspath(os.fspath(output)))
    try:
        output.relative_to(root)
    except ValueError as error:
        raise ValueError("development audit output escapes the repository") from error
    _require_nonsymlink_chain(output, root, "development audit output")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"development audit already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _require_nonsymlink_chain(output.parent, root, "development audit parent")
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", dir=output.parent
    )
    temporary = Path(temporary_name)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        linked = True
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if linked and (output.exists() or output.is_symlink()):
            output.unlink()
        raise
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def audit_seal_provenance(
    plan: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    """Extend trainer provenance to the runtime and audit source bytes."""

    base = analysis_seal_provenance(plan, repository_root=repository_root)
    seal_commit = base["analysis_seal_commit"]
    source = dict(base["source_sha256"])
    for relative in AUDIT_SOURCE_FILES:
        path = repository_root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"audit source must be regular: {relative}")
        current_blob = subprocess.run(
            ("git", "hash-object", "--", relative),
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        sealed_blob = subprocess.run(
            ("git", "rev-parse", f"{seal_commit}:{relative}"),
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        if current_blob != sealed_blob:
            raise RuntimeError(f"audit source differs from seal: {relative}")
        source[relative] = sha256_file(path)
    dirty = subprocess.run(
        ("git", "status", "--porcelain", "--", *AUDIT_SOURCE_FILES),
        cwd=repository_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("development audit source files are dirty")
    return {**base, "source_sha256": source}


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    plan = load_analysis_plan(DEFAULT_ANALYSIS_PLAN, require_preregistered=True)
    _validate_production_analysis_contract(plan)
    provenance = audit_seal_provenance(plan, repository_root=repository_root)
    model_path = repository_root / plan["outputs"]["frozen_model"]
    training_report_path = repository_root / plan["outputs"]["training_report"]
    runtime = load_frozen_public_route_gate(
        model_path,
        training_report_path,
        analysis_plan_path=DEFAULT_ANALYSIS_PLAN,
        repository_root=repository_root,
        cohort="development",
    )
    dataset = load_registered_factual_cohort(
        "development",
        manifest_path=DEFAULT_MANIFEST,
        analysis_plan_path=DEFAULT_ANALYSIS_PLAN,
        repository_root=repository_root,
    )
    _development_ledger, development_ledger_receipt = load_tagged_evidence_ledger(
        dataset,
        plan,
        cohort="development",
        repository_root=repository_root,
    )
    report = build_development_audit(dataset, runtime, plan)
    report["receipts"]["audit_source_sha256"] = provenance["source_sha256"]
    report["receipts"]["analysis_seal_commit"] = provenance[
        "analysis_seal_commit"
    ]
    report["receipts"]["development_ledger_tag"] = development_ledger_receipt[
        "tag"
    ]
    report["receipts"]["development_ledger_commit"] = development_ledger_receipt[
        "commit"
    ]
    report["receipts"]["development_ledger_sha256"] = development_ledger_receipt[
        "sha256"
    ]
    json.dumps(report, sort_keys=True, allow_nan=False)
    output = repository_root / plan["outputs"]["development_audit"]
    publish_json_no_clobber(report, output, repository_root=repository_root)
    print(
        f"[public-route-gate-development] status={report['status']} "
        f"rows={report['row_count']} "
        f"continue_rate={report['gate_policy']['continue_rate']:.6f} "
        f"blind_open={report['blind_collection_open']}"
    )


if __name__ == "__main__":
    main()
