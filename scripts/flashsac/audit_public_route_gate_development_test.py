#!/usr/bin/env python3
"""Fast simulation-free tests for preregistered development inference."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile

import torch

from audit_public_route_gate_development import (
    _acceptance,
    _cluster_bootstrap_samples,
    _doubly_robust_effects,
    _fixed_rank_intervals,
    _hash_seed,
    _horvitz_thompson,
    _permutation_samples,
    _two_sided_plus_one_pvalues,
    _validate_production_analysis_contract,
    publish_json_no_clobber,
)
from public_route_gate_dataset import load_analysis_plan


def _expect_error(error_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _plan() -> dict:
    return load_analysis_plan(
        Path(__file__).with_name("public_route_gate_analysis_plan.json"),
        require_preregistered=False,
    )


def test_ht_is_gate_minus_fixed_route_and_dr_is_separate() -> None:
    treatment_route = torch.tensor([False, True, False, True])
    gate_continue = torch.tensor([True, True, False, False])
    outcomes = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    effects, gate, fixed = _horvitz_thompson(
        treatment_route, gate_continue, outcomes
    )
    assert torch.equal(
        effects,
        torch.tensor(
            [
                [2.0, 0.0, 2.0],
                [-2.0, -2.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
    )
    assert torch.allclose(effects.mean(dim=0), gate - fixed)

    probabilities = torch.full((5, 4, 6), 0.5, dtype=torch.float64)
    dr = _doubly_robust_effects(
        treatment_route, gate_continue, outcomes, probabilities
    )
    assert dr.shape == effects.shape
    assert bool(torch.isfinite(dr).all())
    # If the gate always routes, its DR value is identically fixed route.
    assert torch.equal(
        _doubly_robust_effects(
            treatment_route,
            torch.zeros(4, dtype=torch.bool),
            outcomes,
            probabilities,
        ),
        torch.zeros_like(outcomes),
    )


def test_cluster_bootstrap_keeps_available_pair_rows_together() -> None:
    # Slot 0 is a two-row a/b cluster with value one per row.  Slot 1 has one
    # available row with value three.  Cluster resampling can therefore yield
    # only means 1, 5/3, or 3; row-wise resampling would create other values.
    effects = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    seed = torch.tensor([274, 274, 274], dtype=torch.long)
    slot = torch.tensor([0, 0, 1], dtype=torch.long)
    first = _cluster_bootstrap_samples(
        effects,
        seed,
        slot,
        seed_order=[274],
        replicates=64,
        random_seed=123,
        batch_size=7,
    )
    second = _cluster_bootstrap_samples(
        effects,
        seed,
        slot,
        seed_order=[274],
        replicates=64,
        random_seed=123,
        batch_size=7,
    )
    assert torch.equal(first, second)
    allowed = torch.tensor([1.0, 5.0 / 3.0, 3.0], dtype=torch.float64)
    assert all(
        bool(torch.isclose(value, allowed, atol=1.0e-15, rtol=0.0).any())
        for value in first[:, 0]
    )
    samples = torch.arange(1, 31, dtype=torch.float64).reshape(10, 3)
    lower, upper = _fixed_rank_intervals(samples, lower_rank=2, upper_rank=9)
    assert torch.equal(lower, samples[1])
    assert torch.equal(upper, samples[8])


def test_balanced_permutation_uses_exact_complement_and_plus_one() -> None:
    num_envs = 4
    slot = torch.tensor(list(range(4)) + list(range(4)), dtype=torch.long)
    replicate = torch.tensor([0] * 4 + [1] * 4, dtype=torch.long)
    seed = torch.full((8,), 274, dtype=torch.long)
    gate_continue = torch.ones(8, dtype=torch.bool)
    paired_equal = torch.ones((8, 3), dtype=torch.float64)
    samples = _permutation_samples(
        paired_equal,
        gate_continue,
        seed,
        slot,
        replicate,
        seed_order=[274],
        num_envs=num_envs,
        route_slots=2,
        replicates=40,
        random_seed=17,
        batch_size=9,
    )
    assert torch.equal(samples, torch.zeros_like(samples))

    outcomes = torch.zeros((8, 3), dtype=torch.float64)
    outcomes[0] = 1.0
    samples = _permutation_samples(
        outcomes,
        gate_continue,
        seed,
        slot,
        replicate,
        seed_order=[274],
        num_envs=num_envs,
        route_slots=2,
        replicates=40,
        random_seed=17,
        batch_size=9,
    )
    assert set(samples[:, 0].tolist()) == {-0.25, 0.25}
    p_value = _two_sided_plus_one_pvalues(
        torch.tensor([0.3, 0.0, 0.0]), samples
    )
    assert p_value[0] == 1.0 / 41.0
    assert p_value[1] == 1.0
    assert p_value[2] == 1.0


def test_exact_acceptance_never_cancels_safety_events() -> None:
    plan = _plan()
    _validate_production_analysis_contract(plan)
    event = {
        "frozen_gate_ht": 0.1,
        "fixed_route_ht": 0.1,
        "gate_minus_fixed_route_ht": 0.0,
        "cluster_bootstrap_95pct_lower": 0.0,
        "cluster_bootstrap_95pct_upper": 0.0,
        "balanced_permutation_two_sided_p": 1.0,
    }
    events = {name: copy.deepcopy(event) for name in ("success", "dropped", "unsafe_force")}
    acceptance = _acceptance(
        collection_acceptance={"accepted": True},
        continue_rate=0.02,
        ht_events=events,
        plan=plan,
    )
    assert acceptance["accepted"] is True
    assert acceptance["dr_used_for_acceptance"] is False
    assert acceptance["permutation_p_used_for_acceptance"] is False

    events["dropped"]["cluster_bootstrap_95pct_upper"] = 1.0e-12
    events["unsafe_force"]["cluster_bootstrap_95pct_upper"] = -1.0
    acceptance = _acceptance(
        collection_acceptance={"accepted": True},
        continue_rate=0.02,
        ht_events=events,
        plan=plan,
    )
    assert acceptance["accepted"] is False
    assert acceptance["checks"]["added_drop_95pct_upper"] is False
    assert acceptance["checks"]["added_unsafe_force_95pct_upper"] is True


def test_hash_seed_contract_plan_drift_and_no_clobber() -> None:
    manifest_sha = "a" * 64
    label = "development_cluster_bootstrap_v1"
    expected = int.from_bytes(
        hashlib.sha256(
            manifest_sha.encode("ascii") + b"\0" + label.encode("ascii")
        ).digest()[:8],
        "big",
    )
    assert _hash_seed(manifest_sha, label) == expected
    plan = _plan()
    changed = copy.deepcopy(plan)
    changed["randomization_inference"]["bootstrap"]["replicates"] = 19_999
    _expect_error(ValueError, _validate_production_analysis_contract, changed)

    with tempfile.TemporaryDirectory() as directory_name:
        output = Path(directory_name) / "audit.json"
        publish_json_no_clobber(
            {"overall_pass": False}, output, repository_root=Path(directory_name)
        )
        original = output.read_bytes()
        _expect_error(
            FileExistsError,
            publish_json_no_clobber,
            {"overall_pass": True},
            output,
            repository_root=Path(directory_name),
        )
        assert output.read_bytes() == original
        assert not list(output.parent.glob(".audit.json.tmp-*"))
