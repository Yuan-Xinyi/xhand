#!/usr/bin/env python3
"""CPU tests for Candidate 39 fixed-direction artifacts and build_spec."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile

import torch

import candidate39_fixed_episode_residual as contract
from collect_candidate39_fixed_residual_ab import build_spec


N = 64
SEED = 329


def _raises(error: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def _metadata(replicate: str = "a") -> dict:
    sources = {"scripts/flashsac/fake.py": "a" * 64}
    assets = {"/tmp/fake.usd": "b" * 64}
    return {
        **contract.REQUIRED_METADATA,
        "seed": SEED,
        "replicate": replicate,
        "num_envs": N,
        "assignment_mask_sha256": contract.assignment_mask_sha256(
            contract.exact_balanced_treatment_mask(
                seed=SEED, num_envs=N, replicate=replicate
            )
        ),
        "v6_checkpoint": "logs/v6",
        "search_checkpoint": "logs/search.pth",
        "validation_plan": contract.VALIDATION_PLAN,
        "validation_plan_sha256": contract._validation_plan_digest(),
        "fixed_direction_manifest": contract.FIXED_DIRECTION_MANIFEST,
        "fixed_direction_manifest_sha256": (
            contract.FIXED_DIRECTION_MANIFEST_SHA256
        ),
        "fixed_direction_path": contract.FIXED_DIRECTION_PATH,
        "fixed_direction_sha256": contract.FIXED_DIRECTION_SHA256,
        "v6_actor_sha256": contract.V6_ACTOR_SHA256,
        "v6_task_contract_sha256": contract.V6_TASK_CONTRACT_SHA256,
        "v6_bridge_state_sha256": contract.V6_BRIDGE_STATE_SHA256,
        "frozen_lift_actor_sha256": contract.FROZEN_LIFT_ACTOR_SHA256,
        "frozen_lift_semantic_sha256": contract.FROZEN_LIFT_SEMANTIC_SHA256,
        "frozen_lift_source_actor_sha256": (
            contract.FROZEN_LIFT_SOURCE_ACTOR_SHA256
        ),
        "search_checkpoint_sha256": contract.SEARCH_CHECKPOINT_SHA256,
        "source_manifest_sha256": contract.manifest_sha256(sources),
        "runtime_asset_manifest_sha256": contract.manifest_sha256(assets),
        "source_sha256": sources,
        "runtime_asset_sha256": assets,
        "flashsac_upstream_commit": "8" * 40,
        "flashsac_fork_commit": contract.FLASHSAC_FORK_COMMIT,
        "git": {
            "commit": "a" * 40,
            "branch": "flashsac-pick-tool-curriculum",
            "source_files_dirty": False,
            "flashsac_commit": contract.FLASHSAC_FORK_COMMIT,
            "flashsac_dirty": False,
        },
        "runtime": {
            "python": "3.11",
            "torch": "2",
            "cuda": "12",
            "cudnn": 9000,
            "cuda_device_index": 0,
            "cuda_device_name": "fake",
            "cuda_device_capability": [8, 9],
            "isaac_sim": "5",
            "packages": {
                name: "1" for name in contract.RUNTIME_PACKAGE_FIELDS
            },
            "nvidia_smi_inventory": "fake",
            "platform": "linux",
            "seed": SEED,
        },
    }


def _payload(replicate: str = "a") -> tuple[dict, dict, dict]:
    treatment = contract.exact_balanced_treatment_mask(
        seed=SEED, num_envs=N, replicate=replicate
    )
    trigger_step = torch.full((N,), 3, dtype=torch.long)
    episode_length = torch.full((N,), 5, dtype=torch.long)
    episodes = {
        "env_slot": torch.arange(N),
        "treatment": treatment,
        "assignment_rank": contract.assignment_rank(seed=SEED, num_envs=N),
        "fixed_z": contract.expected_fixed_z().expand(N, -1).clone(),
        "triggered": torch.ones(N, dtype=torch.bool),
        "trigger_step": trigger_step,
        "trigger_score": torch.full((N,), 0.4),
        "first_latch_step": torch.full((N,), -1, dtype=torch.long),
        "latch_released_after_first": torch.zeros(N, dtype=torch.bool),
        "intervention_steps": treatment.long() * 2,
        "episode_length": episode_length,
        "trajectory_max_force_n": torch.full((N,), 10.0),
        "max_true_clearance_m": torch.full((N,), 0.01),
        "success": torch.zeros(N, dtype=torch.bool),
        "failure": torch.ones(N, dtype=torch.bool),
        "time_out": torch.zeros(N, dtype=torch.bool),
        "dropped": torch.ones(N, dtype=torch.bool),
        "unsafe_force": torch.zeros(N, dtype=torch.bool),
        "unlatched_clearance_ge_5cm": torch.zeros(N, dtype=torch.bool),
        "ever_grasped": torch.zeros(N, dtype=torch.bool),
        "ever_clearance_ge_20cm": torch.zeros(N, dtype=torch.bool),
        "latched_within_window": torch.zeros(N, dtype=torch.bool),
    }
    env_slot = torch.arange(N).repeat(2)
    episode_step = torch.arange(3, 5).repeat_interleave(N)
    close_age = episode_step - 3
    rows = int(env_slot.numel())
    observation = torch.zeros((rows, contract.OBSERVATION_DIM))
    observation[:, 92:97] = 0.4
    base_mean = torch.zeros((rows, contract.HAND_ACTION_DIM))
    baseline = torch.zeros((rows, contract.ACTION_DIM))
    active = treatment[env_slot]
    delta = torch.where(
        active[:, None],
        contract.expected_applied_delta().expand(rows, -1),
        torch.zeros((rows, contract.HAND_ACTION_DIM)),
    )
    candidate = baseline.clone()
    candidate[:, contract.ARM_ACTION_DIM :] = torch.tanh(delta)
    candidate[~active] = baseline[~active]
    steps = {
        "row_env_slot": env_slot,
        "row_episode_step": episode_step,
        "row_close_age": close_age,
        "row_public_latch_before": torch.zeros(rows, dtype=torch.bool),
        "row_residual_active": active,
        "row_observation": observation,
        "row_base_mean_hand": base_mean,
        "row_applied_delta": delta,
        "row_baseline_action": baseline,
        "row_candidate_action": candidate,
        "row_executed_action": candidate.clone(),
        "row_transition_public_latch": torch.zeros(rows, dtype=torch.bool),
        "row_transition_grasped": torch.zeros(rows, dtype=torch.bool),
        "row_transition_true_clearance_m": torch.zeros(rows),
        "row_grasp_quality": torch.zeros(rows),
        "row_hold_quality": torch.zeros(rows),
        "row_max_force_n": torch.ones(rows),
    }
    return _metadata(replicate), episodes, steps


def _artifact(replicate: str = "a") -> dict:
    return contract.build_artifact(*_payload(replicate))


def _report(artifact: dict) -> dict:
    metadata = artifact["metadata"]
    return {
        "kind": contract.REPORT_KIND,
        "status": "complete",
        "collector": contract.COLLECTOR,
        "seed": metadata["seed"],
        "replicate": metadata["replicate"],
        "num_envs": metadata["num_envs"],
        "vector_steps": int(artifact["episodes"]["episode_length"].max()),
        "v6_actor_sha256": metadata["v6_actor_sha256"],
        "search_checkpoint_sha256": metadata["search_checkpoint_sha256"],
        "fixed_direction_sha256": metadata["fixed_direction_sha256"],
        "summary": contract.summarize_artifact(artifact),
    }


def test_assignment_is_independent_balanced_stable_and_complementary() -> None:
    for seed in (329, 330):
        a = contract.exact_balanced_treatment_mask(
            seed=seed, num_envs=64, replicate="a"
        )
        b = contract.exact_balanced_treatment_mask(
            seed=seed, num_envs=64, replicate="b"
        )
        rank = contract.assignment_rank(seed=seed, num_envs=64)
        assert int(a.sum()) == 32
        assert torch.equal(b, ~a)
        assert torch.equal(a, rank < 32)
        assert torch.equal(torch.sort(rank).values, torch.arange(64))
    assert contract.ASSIGNMENT_SALT != contract.FIXED_DIRECTION_SOURCE_DESIGN_SALT
    _raises(
        ValueError,
        contract.exact_balanced_treatment_mask,
        seed=329,
        num_envs=63,
        replicate="a",
    )


def test_fixed_payload_and_bounded_delta_are_exact() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = contract.load_fixed_direction(root / contract.FIXED_DIRECTION_PATH)
    assert torch.equal(payload["fixed_z"], contract.expected_fixed_z())
    delta = contract.expected_applied_delta()
    assert delta.shape == (14,)
    assert float(delta[:9].abs().max()) <= contract.TOKEN_COMPONENT_CAP
    assert float(delta[9:].abs().max()) <= contract.DISTAL_COMPONENT_CAP
    assert float(torch.linalg.vector_norm(delta)) <= contract.PRE_TANH_L2_CAP
    corrupt = copy.deepcopy(payload)
    corrupt["fixed_z"][0] += 1e-6
    _raises(ValueError, contract.validate_fixed_direction_payload, corrupt)


def test_valid_artifact_report_transition_edge_and_complement() -> None:
    a = _artifact("a")
    b = _artifact("b")
    contract.validate_report(_report(a), a)
    contract.validate_complementary_artifacts(a, b)
    assert contract.summarize_artifact(a)["episodes"] == N

    metadata, episodes, steps = _payload("a")
    slot = int(steps["row_env_slot"][-1])
    episodes["first_latch_step"][slot] = 4
    episodes["latched_within_window"][slot] = True
    row = (steps["row_env_slot"] == slot) & (steps["row_episode_step"] == 4)
    steps["row_transition_public_latch"][row] = True
    contract.build_artifact(metadata, episodes, steps)


def test_corruption_fails_closed() -> None:
    base = _artifact()
    mutations = []
    bad = copy.deepcopy(base); bad["episodes"]["assignment_rank"][[0, 1]] = bad["episodes"]["assignment_rank"][[1, 0]]; mutations.append(bad)
    bad = copy.deepcopy(base); bad["episodes"]["fixed_z"][0, 0] += 1e-4; mutations.append(bad)
    bad = copy.deepcopy(base); bad["episodes"]["treatment"][0] ^= True; mutations.append(bad)
    bad = copy.deepcopy(base); bad["steps"]["row_applied_delta"][0, 0] += 0.01; mutations.append(bad)
    bad = copy.deepcopy(base); bad["steps"]["row_candidate_action"][0, 0] = 0.1; mutations.append(bad)
    bad = copy.deepcopy(base); bad["steps"]["row_transition_public_latch"][0] = True; mutations.append(bad)
    bad = copy.deepcopy(base); bad["metadata"]["fixed_direction_sha256"] = "f" * 64; mutations.append(bad)
    bad = copy.deepcopy(base); bad["metadata"]["fixed_direction_manifest_sha256"] = "e" * 64; mutations.append(bad)
    bad = copy.deepcopy(base); bad["metadata"]["validation_plan_sha256"] = "d" * 64; mutations.append(bad)
    bad = copy.deepcopy(base); bad["metadata"]["v6_actor_sha256"] = "0" * 64; mutations.append(bad)
    bad = copy.deepcopy(base); bad["metadata"]["search_checkpoint_sha256"] = "0" * 64; mutations.append(bad)
    bad = copy.deepcopy(base); bad["metadata"]["source_manifest_sha256"] = "c" * 64; mutations.append(bad)
    bad = copy.deepcopy(base); bad["metadata"]["git"]["branch"] = "main"; mutations.append(bad)
    bad = copy.deepcopy(base); bad["metadata"]["git"]["flashsac_commit"] = "c" * 40; mutations.append(bad)
    for artifact in mutations:
        _raises(ValueError, contract.validate_artifact, artifact)
    bad = copy.deepcopy(base)
    bad["steps"]["row_observation"][0, 0] = float("nan")
    _raises(FloatingPointError, contract.validate_artifact, bad)


def test_zero_rows_legal_only_after_pretrigger_latch_retirement() -> None:
    metadata, episodes, steps = _payload()
    episodes["first_latch_step"][:] = 1
    episodes["latch_released_after_first"][:] = True
    episodes["intervention_steps"][:] = 0
    episodes["ever_grasped"][:] = True
    for name in contract.STEP_FIELDS:
        steps[name] = steps[name][:0]
    artifact = contract.build_artifact(metadata, episodes, steps)
    assert contract.summarize_artifact(artifact)["step_rows"] == 0
    invalid = copy.deepcopy(artifact)
    invalid["episodes"]["first_latch_step"][:] = -1
    _raises(ValueError, contract.validate_artifact, invalid)


def test_weights_only_atomic_publication_and_no_clobber() -> None:
    artifact = _artifact()
    report = _report(artifact)
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        artifact_output = root / "trial.pt"
        report_output = root / "trial.json"
        digest = contract.publish_artifact_and_report_no_clobber(
            artifact, report, artifact_output, report_output
        )
        assert digest == contract.sha256_file(artifact_output)
        restored = torch.load(
            artifact_output, map_location="cpu", weights_only=True
        )
        contract.validate_artifact(restored)
        published = json.loads(report_output.read_text())
        contract.validate_report(published, restored, published=True)
        _raises(
            FileExistsError,
            contract.publish_artifact_and_report_no_clobber,
            artifact,
            report,
            artifact_output,
            report_output,
        )


def _spec_args(root: Path, *, seed: int, replicate: str, num_envs: int) -> argparse.Namespace:
    output_stem = (
        root
        / "logs/flashsac/pick_tool/51_c39_fixed_direction_smoke_s326_b/trial"
        if num_envs == 8
        else root
        / (
            "logs/flashsac/pick_tool/51_c39_fixed_direction_dev_"
            f"s{seed}_{replicate}/trial"
        )
    )
    return argparse.Namespace(
        v6_checkpoint=root
        / "logs/flashsac/pick_tool/16_v6_router_zero_actor_s254/checkpoint_final",
        search_checkpoint=root
        / "logs/rl_games/pick_tool_token/0_bootstrap_handoff_20260720/nn/pick_tool_stage7_dagger_full_iter3_bc.pth",
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
        kit_args=contract.KIT_ARGS,
        output_stem=output_stem,
    )


def test_checkpoint_build_spec_freezes_fixed_design_before_launcher() -> None:
    root = Path(__file__).resolve().parents[2]
    a = build_spec(_spec_args(root, seed=329, replicate="a", num_envs=64))
    b = build_spec(_spec_args(root, seed=329, replicate="b", num_envs=64))
    assert torch.equal(b.treatment, ~a.treatment)
    assert torch.equal(a.assignment_rank, b.assignment_rank)
    assert torch.equal(a.fixed_z, b.fixed_z)
    assert a.immutable_sha256["fixed_direction"] == contract.FIXED_DIRECTION_SHA256
    smoke = build_spec(_spec_args(root, seed=326, replicate="b", num_envs=8))
    assert int(smoke.treatment.sum()) == 4
    _raises(
        ValueError,
        build_spec,
        _spec_args(root, seed=328, replicate="a", num_envs=64),
    )
    _raises(
        ValueError,
        build_spec,
        _spec_args(root, seed=331, replicate="a", num_envs=8),
    )
    _raises(
        ValueError,
        build_spec,
        _spec_args(root, seed=326, replicate="a", num_envs=8),
    )


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    test_assignment_is_independent_balanced_stable_and_complementary()
    test_fixed_payload_and_bounded_delta_are_exact()
    test_valid_artifact_report_transition_edge_and_complement()
    test_corruption_fails_closed()
    test_zero_rows_legal_only_after_pretrigger_latch_retirement()
    test_weights_only_atomic_publication_and_no_clobber()
    test_checkpoint_build_spec_freezes_fixed_design_before_launcher()
    print("candidate39_fixed_episode_residual tests passed")


if __name__ == "__main__":
    main()
