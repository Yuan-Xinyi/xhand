#!/usr/bin/env python3
"""CPU-only tests for Candidate 39 residual direction discovery."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

import torch

import candidate39_episode_residual as contract
from analyze_candidate39_residual_discovery import (
    FIXED_Z_KIND,
    GATE_THRESHOLDS,
    REPLICATES,
    SEEDS,
    _registered_development_gates,
    build_fixed_z_payload,
    compute_discovery,
    publish_fixed_z_no_clobber,
    publish_json_no_clobber,
    validate_discovery_artifacts,
    validate_fixed_z_payload,
)
from option_residual_screen import (
    apply_option_residual,
    build_option_design,
    grouped_pre_tanh_scale,
)


NUM_ENVS = 64
SHA = "1" * 64
GIT_SHA = "2" * 40


def _expect(error_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _metadata(seed: int, replicate: str) -> dict[str, object]:
    source = {"scripts/flashsac/source.py": SHA}
    assets = {"asset.usd": SHA}
    return {
        **contract.REQUIRED_METADATA,
        "seed": seed,
        "replicate": replicate,
        "num_envs": NUM_ENVS,
        "v6_checkpoint": "logs/v6",
        "search_checkpoint": "logs/search.pth",
        "v6_actor_sha256": SHA,
        "v6_task_contract_sha256": SHA,
        "v6_bridge_state_sha256": SHA,
        "frozen_lift_actor_sha256": SHA,
        "frozen_lift_semantic_sha256": SHA,
        "frozen_lift_source_actor_sha256": SHA,
        "search_checkpoint_sha256": SHA,
        "source_manifest_sha256": contract.manifest_sha256(source),
        "runtime_asset_manifest_sha256": contract.manifest_sha256(assets),
        "source_sha256": source,
        "runtime_asset_sha256": assets,
        "flashsac_upstream_commit": GIT_SHA,
        "flashsac_fork_commit": GIT_SHA,
        "git": {
            "commit": GIT_SHA,
            "branch": "flashsac-pick-tool-curriculum",
            "source_files_dirty": False,
            "flashsac_commit": GIT_SHA,
            "flashsac_dirty": False,
        },
        "runtime": {
            "python": "test",
            "torch": str(torch.__version__),
            "cuda": "test",
            "cudnn": 0,
            "cuda_device_index": 0,
            "cuda_device_name": "test",
            "cuda_device_capability": [0, 0],
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
            "seed": seed,
        },
    }


def _artifact(
    *, seed: int, replicate: str, favored_slots: tuple[int, ...]
) -> dict[str, object]:
    design = build_option_design(
        seed=seed, num_envs=NUM_ENVS, replicate=replicate
    )
    target = torch.zeros(NUM_ENVS, dtype=torch.bool)
    target[list(favored_slots)] = True
    # A target slot latches only in whichever complementary replicate assigned
    # it to treatment.  Thus every target supplies paired advantage +1.
    latched = target & design.treatment
    triggered = torch.ones(NUM_ENVS, dtype=torch.bool)
    trigger_step = torch.full((NUM_ENVS,), 3, dtype=torch.int64)
    first_latch = torch.where(
        latched,
        trigger_step,
        torch.full((NUM_ENVS,), -1, dtype=torch.int64),
    )

    row_env: list[int] = []
    row_age: list[int] = []
    for age in range(32):
        for env_slot in range(NUM_ENVS):
            if bool(latched[env_slot]) and age > 0:
                continue
            row_env.append(env_slot)
            row_age.append(age)
    env = torch.tensor(row_env, dtype=torch.int64)
    age = torch.tensor(row_age, dtype=torch.int64)
    rows = env.numel()
    observation = torch.zeros((rows, 115), dtype=torch.float32)
    observation[:, 92:97] = 0.4
    baseline = torch.zeros((rows, 21), dtype=torch.float32)
    row_treatment = design.treatment[env]
    overlay = apply_option_residual(
        baseline_action=baseline,
        option_active=torch.ones(rows, dtype=torch.bool),
        public_latch=torch.zeros(rows, dtype=torch.bool),
        treatment=row_treatment,
        raw_z=design.raw_z[env],
        pre_tanh_scale=grouped_pre_tanh_scale(
            token_scale=0.05, distal_scale=0.025
        ),
        raw_z_abs_cap=2.0,
        pre_tanh_abs_cap=torch.tensor([0.1] * 9 + [0.05] * 5),
        pre_tanh_l2_cap=0.2,
    )
    transition_grasped = latched[env] & (age == 0)
    steps = {
        "row_env_slot": env,
        "row_episode_step": trigger_step[env] + age,
        "row_close_age": age,
        "row_public_latch_before": torch.zeros(rows, dtype=torch.bool),
        "row_residual_active": row_treatment,
        "row_observation": observation,
        "row_base_mean_hand": torch.zeros((rows, 14), dtype=torch.float32),
        "row_applied_delta": overlay.pre_tanh_residual,
        "row_baseline_action": baseline,
        "row_candidate_action": overlay.action,
        "row_executed_action": overlay.action,
        "row_transition_grasped": transition_grasped,
        "row_transition_true_clearance_m": torch.zeros(rows),
        "row_grasp_quality": transition_grasped.to(torch.float32),
        "row_hold_quality": transition_grasped.to(torch.float32),
        "row_max_force_n": torch.ones(rows),
    }
    intervention_steps = torch.bincount(
        env[row_treatment], minlength=NUM_ENVS
    )
    episodes = {
        "env_slot": torch.arange(NUM_ENVS, dtype=torch.int64),
        "treatment": design.treatment,
        "raw_z": design.raw_z,
        "pair_slot": design.pair_slot,
        "antithetic_sign": design.antithetic_sign,
        "triggered": triggered,
        "trigger_step": trigger_step,
        "trigger_score": torch.full((NUM_ENVS,), 0.4),
        "first_latch_step": first_latch,
        "latch_released_after_first": torch.zeros(NUM_ENVS, dtype=torch.bool),
        "intervention_steps": intervention_steps,
        "episode_length": torch.full(
            (NUM_ENVS,), contract.MAX_EPISODE_ACTIONS, dtype=torch.int64
        ),
        "trajectory_max_force_n": torch.ones(NUM_ENVS),
        "max_true_clearance_m": torch.zeros(NUM_ENVS),
        "success": torch.zeros(NUM_ENVS, dtype=torch.bool),
        "failure": torch.zeros(NUM_ENVS, dtype=torch.bool),
        "time_out": torch.ones(NUM_ENVS, dtype=torch.bool),
        "dropped": torch.zeros(NUM_ENVS, dtype=torch.bool),
        "unsafe_force": torch.zeros(NUM_ENVS, dtype=torch.bool),
        "unlatched_clearance_ge_5cm": torch.zeros(NUM_ENVS, dtype=torch.bool),
        "ever_grasped": latched,
        "ever_clearance_ge_20cm": torch.zeros(NUM_ENVS, dtype=torch.bool),
        "latched_within_window": latched,
    }
    return contract.build_artifact(
        metadata=_metadata(seed, replicate), episodes=episodes, steps=steps
    )


def _artifacts(*, discordants_per_seed: int) -> dict[tuple[int, str], dict[str, object]]:
    favored = tuple(range(discordants_per_seed))
    return {
        (seed, replicate): _artifact(
            seed=seed, replicate=replicate, favored_slots=favored
        )
        for seed in SEEDS
        for replicate in REPLICATES
    }


def test_positive_discovery_uses_common_slot_pairing_and_exact_float32_direction() -> None:
    artifacts = _artifacts(discordants_per_seed=4)
    report = compute_discovery(artifacts)
    assert report["all_gates_pass"] is True
    assert report["decision"] == "extract_fixed_direction"
    assert report["paired_discovery"]["common_exposure_opportunity_slots"] == 128
    assert report["paired_discovery"][
        "discordant_latched_within_window_slots"
    ] == 8
    assert "not within-arm observed proportions" in report["estimand_note"]
    assert report["raw_funnel"]["treatment"]["latched_within_window"]["count"] == 8
    assert report["raw_funnel"]["control"]["latched_within_window"]["count"] == 0

    expected_gradient = torch.zeros(14, dtype=torch.float64)
    for seed in SEEDS:
        raw_z = artifacts[(seed, "a")]["episodes"]["raw_z"]
        expected_gradient += raw_z[:4].to(torch.float64).clamp(-2, 2).sum(dim=0)
    expected = (
        expected_gradient
        / torch.sqrt(torch.mean(expected_gradient.square()))
    ).clamp(-2, 2).to(torch.float32)
    payload = build_fixed_z_payload(report)
    assert payload["kind"] == FIXED_Z_KIND
    assert payload["fixed_z"].dtype == torch.float32
    assert torch.equal(payload["fixed_z"], expected)
    assert report["step_audit"]["aggregate"][
        "component_and_l2_budget_violations"
    ] == 0


def test_registered_development_gates_match_analyzer_exactly() -> None:
    assert _registered_development_gates() == GATE_THRESHOLDS


def test_negative_discovery_never_exposes_or_publishes_candidate_z() -> None:
    report = compute_discovery(_artifacts(discordants_per_seed=3))
    assert report["all_gates_pass"] is False
    assert report["gates"]["discordant_latched_within_window_slots"]["pass"] is False
    assert "candidate_direction" not in report
    assert "gradient" not in report["paired_discovery"]
    assert "fixed_z" not in report["paired_discovery"]
    _expect(ValueError, build_fixed_z_payload, report)


def test_assignment_latent_and_cross_run_identity_corruption_is_rejected() -> None:
    artifacts = _artifacts(discordants_per_seed=4)

    corrupted = copy.deepcopy(artifacts)
    corrupted[(327, "a")]["episodes"]["treatment"][0] ^= True
    _expect(ValueError, validate_discovery_artifacts, corrupted)

    corrupted = copy.deepcopy(artifacts)
    corrupted[(327, "a")]["episodes"]["raw_z"][0, 0] += 1.0
    _expect(ValueError, validate_discovery_artifacts, corrupted)

    # Keep each seed's complementary pair internally consistent, but change a
    # policy lineage hash across seeds.  Only the discovery analyzer's global
    # identity check can catch this corruption.
    corrupted = copy.deepcopy(artifacts)
    for replicate in REPLICATES:
        corrupted[(328, replicate)]["metadata"]["v6_actor_sha256"] = "3" * 64
    _expect(ValueError, validate_discovery_artifacts, corrupted)


def test_pretrigger_latch_without_intervention_is_excluded_from_direction_domain() -> None:
    artifacts = _artifacts(discordants_per_seed=4)
    slot = 63
    for replicate in REPLICATES:
        artifact = copy.deepcopy(artifacts[(327, replicate)])
        episodes = artifact["episodes"]
        steps = artifact["steps"]
        keep = steps["row_env_slot"] != slot
        for name in contract.STEP_FIELDS:
            steps[name] = steps[name][keep]
        episodes["first_latch_step"][slot] = 1
        episodes["latch_released_after_first"][slot] = True
        episodes["intervention_steps"][slot] = 0
        episodes["ever_grasped"][slot] = True
        episodes["latched_within_window"][slot] = False
        artifacts[(327, replicate)] = contract.validate_artifact(artifact)

    report = compute_discovery(artifacts)
    seed = report["paired_discovery"]["per_seed"]["327"]
    assert seed["triggered_in_both_replicates"] == 64
    assert seed["common_exposure_opportunity_slots"] == 63
    assert seed["excluded_without_common_intervention_opportunity"] == 1
    assert report["paired_discovery"]["common_exposure_opportunity_slots"] == 127


def test_no_clobber_json_and_weights_only_fixed_z_publication() -> None:
    report = compute_discovery(_artifacts(discordants_per_seed=4))
    payload = build_fixed_z_payload(report)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        json_path = root / "discovery.json"
        fixed_path = root / "fixed_z.pt"
        publish_json_no_clobber(report, json_path)
        assert json.loads(json_path.read_text(encoding="utf-8"))[
            "all_gates_pass"
        ] is True
        _expect(FileExistsError, publish_json_no_clobber, report, json_path)

        digest = publish_fixed_z_no_clobber(payload, fixed_path)
        assert len(digest) == 64
        restored = torch.load(fixed_path, map_location="cpu", weights_only=True)
        validate_fixed_z_payload(restored)
        assert torch.equal(restored["fixed_z"], payload["fixed_z"])
        _expect(FileExistsError, publish_fixed_z_no_clobber, payload, fixed_path)


if __name__ == "__main__":
    test_positive_discovery_uses_common_slot_pairing_and_exact_float32_direction()
    test_registered_development_gates_match_analyzer_exactly()
    test_negative_discovery_never_exposes_or_publishes_candidate_z()
    test_assignment_latent_and_cross_run_identity_corruption_is_rejected()
    test_pretrigger_latch_without_intervention_is_excluded_from_direction_domain()
    test_no_clobber_json_and_weights_only_fixed_z_publication()
    print("analyze_candidate39_residual_discovery_test: PASS")
