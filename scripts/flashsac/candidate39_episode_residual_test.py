#!/usr/bin/env python3
"""CPU corruption tests for the Candidate 39 episode-residual artifact."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

import torch

import candidate39_episode_residual as contract
from option_residual_screen import build_option_design


N = 4
SEED = 327


def _raises(error: type[BaseException], fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def _metadata(replicate: str = "a") -> dict:
    fake = "a" * 64
    sources = {"scripts/flashsac/fake.py": fake}
    assets = {"/tmp/fake.usd": "b" * 64}
    return {
        **contract.REQUIRED_METADATA,
        "seed": SEED,
        "replicate": replicate,
        "num_envs": N,
        "v6_checkpoint": "logs/v6",
        "search_checkpoint": "logs/search.pth",
        "v6_actor_sha256": "1" * 64,
        "v6_task_contract_sha256": "2" * 64,
        "v6_bridge_state_sha256": "3" * 64,
        "frozen_lift_actor_sha256": "4" * 64,
        "frozen_lift_semantic_sha256": "5" * 64,
        "frozen_lift_source_actor_sha256": "6" * 64,
        "search_checkpoint_sha256": "7" * 64,
        "source_manifest_sha256": contract.manifest_sha256(sources),
        "runtime_asset_manifest_sha256": contract.manifest_sha256(assets),
        "source_sha256": sources,
        "runtime_asset_sha256": assets,
        "flashsac_upstream_commit": "8" * 40,
        "flashsac_fork_commit": "9" * 40,
        "git": {
            "commit": "a" * 40,
            "branch": "flashsac-pick-tool-curriculum",
            "source_files_dirty": False,
            "flashsac_commit": "9" * 40,
            "flashsac_dirty": False,
        },
        "runtime": {
            "python": "3.11", "torch": "2", "cuda": "12", "cudnn": 9000,
            "cuda_device_index": 0, "cuda_device_name": "fake",
            "cuda_device_capability": [8, 9], "isaac_sim": "5",
            "packages": {name: "1" for name in contract.RUNTIME_PACKAGE_FIELDS},
            "nvidia_smi_inventory": "fake", "platform": "linux", "seed": SEED,
        },
    }


def _payload(replicate: str = "a") -> tuple[dict, dict, dict]:
    design = build_option_design(seed=SEED, num_envs=N, replicate=replicate)
    trigger_step = torch.full((N,), 3, dtype=torch.long)
    episode_length = torch.full((N,), 5, dtype=torch.long)
    episodes = {
        "env_slot": torch.arange(N),
        "treatment": design.treatment,
        "raw_z": design.raw_z,
        "pair_slot": design.pair_slot,
        "antithetic_sign": design.antithetic_sign,
        "triggered": torch.ones(N, dtype=torch.bool),
        "trigger_step": trigger_step,
        "trigger_score": torch.full((N,), .4),
        "first_latch_step": torch.full((N,), -1, dtype=torch.long),
        "latch_released_after_first": torch.zeros(N, dtype=torch.bool),
        "intervention_steps": design.treatment.long() * 2,
        "episode_length": episode_length,
        "trajectory_max_force_n": torch.full((N,), 10.0),
        "max_true_clearance_m": torch.full((N,), .01),
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
    # Canonical step-major order: every slot at step 3, then every slot at 4.
    env = torch.arange(N).repeat(2)
    step = torch.arange(3, 5).repeat_interleave(N)
    age = step - 3
    rows = env.numel()
    observation = torch.zeros((rows, contract.OBSERVATION_DIM))
    observation[:, 92:97] = .4
    base_mean = torch.zeros((rows, contract.HAND_ACTION_DIM))
    baseline = torch.zeros((rows, contract.ACTION_DIM))
    active = design.treatment[env]
    delta = contract._expected_delta(design.raw_z[env])
    delta = torch.where(active[:, None], delta, torch.zeros_like(delta))
    candidate = baseline.clone()
    candidate[:, contract.ARM_ACTION_DIM:] = torch.tanh(delta)
    candidate[~active] = baseline[~active]
    executed = torch.where(active[:, None], candidate, baseline)
    steps = {
        "row_env_slot": env,
        "row_episode_step": step,
        "row_close_age": age,
        "row_public_latch_before": torch.zeros(rows, dtype=torch.bool),
        "row_residual_active": active,
        "row_observation": observation,
        "row_base_mean_hand": base_mean,
        "row_applied_delta": delta,
        "row_baseline_action": baseline,
        "row_candidate_action": candidate,
        "row_executed_action": executed,
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
    meta = artifact["metadata"]
    return {
        "kind": contract.REPORT_KIND,
        "status": "complete",
        "collector": contract.COLLECTOR,
        "seed": meta["seed"],
        "replicate": meta["replicate"],
        "num_envs": meta["num_envs"],
        "vector_steps": int(artifact["episodes"]["episode_length"].max()),
        "v6_actor_sha256": meta["v6_actor_sha256"],
        "search_checkpoint_sha256": meta["search_checkpoint_sha256"],
        "summary": contract.summarize_artifact(artifact),
    }


def test_valid_build_summary_and_complement() -> None:
    a, b = _artifact("a"), _artifact("b")
    contract.validate_artifact(a)
    contract.validate_report(_report(a), a)
    contract.validate_complementary_artifacts(a, b)
    assert contract.summarize_artifact(a)["episodes"] == N
    corrupt = copy.deepcopy(b)
    corrupt["episodes"]["raw_z"][0, 0] += 1e-4
    _raises(ValueError, contract.validate_complementary_artifacts, a, corrupt)


def test_corruption_fails_closed() -> None:
    base = _artifact()
    mutations = []
    bad = copy.deepcopy(base); bad["episodes"]["raw_z"][0, 0] += .1; mutations.append(bad)
    bad = copy.deepcopy(base); bad["episodes"]["success"][0] = True; mutations.append(bad)
    bad = copy.deepcopy(base); bad["steps"]["row_env_slot"][[0, 1]] = bad["steps"]["row_env_slot"][[1, 0]]; mutations.append(bad)
    bad = copy.deepcopy(base); bad["steps"]["row_candidate_action"][0, 0] = .1; mutations.append(bad)
    bad = copy.deepcopy(base); bad["steps"]["row_applied_delta"][0, 0] += .01; mutations.append(bad)
    bad = copy.deepcopy(base); bad["metadata"]["source_manifest_sha256"] = "f" * 64; mutations.append(bad)
    bad = copy.deepcopy(base); bad["metadata"]["git"]["branch"] = "main"; mutations.append(bad)
    bad = copy.deepcopy(base); bad["metadata"]["git"]["flashsac_commit"] = "c" * 40; mutations.append(bad)
    for artifact in mutations:
        _raises(ValueError, contract.validate_artifact, artifact)
    bad = copy.deepcopy(base); bad["steps"]["row_observation"][0, 0] = float("nan")
    _raises(FloatingPointError, contract.validate_artifact, bad)


def test_all_zero_rows_is_legal_only_for_pretrigger_latch_retirement() -> None:
    metadata, episodes, steps = _payload()
    episodes["first_latch_step"][:] = 1
    episodes["latch_released_after_first"][:] = True
    episodes["intervention_steps"][:] = 0
    episodes["ever_grasped"][:] = True
    for name in contract.STEP_FIELDS:
        steps[name] = steps[name][:0]
    artifact = contract.build_artifact(metadata, episodes, steps)
    assert contract.summarize_artifact(artifact)["step_rows"] == 0
    assert contract.summarize_artifact(artifact)["action_audit"]["applied_delta_l2_max"] == 0.0

    invalid = copy.deepcopy(artifact)
    invalid["episodes"]["first_latch_step"][:] = -1
    _raises(ValueError, contract.validate_artifact, invalid)


def test_atomic_weights_only_publication_and_no_clobber() -> None:
    artifact = _artifact()
    report = _report(artifact)
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        pt, js = root / "trial.pt", root / "trial.json"
        digest = contract.publish_artifact_and_report_no_clobber(artifact, report, pt, js)
        assert digest == contract.sha256_file(pt)
        loaded = torch.load(pt, map_location="cpu", weights_only=True)
        contract.validate_artifact(loaded)
        published = json.loads(js.read_text())
        assert published["artifact_sha256"] == digest
        contract.validate_report(published, loaded, published=True)
        _raises(FileExistsError, contract.publish_artifact_and_report_no_clobber, artifact, report, pt, js)
        standalone = root / "one.json"
        contract.publish_json_no_clobber({"ok": True}, standalone)
        _raises(FileExistsError, contract.publish_json_no_clobber, {"ok": False}, standalone)


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    test_valid_build_summary_and_complement()
    test_corruption_fails_closed()
    test_all_zero_rows_is_legal_only_for_pretrigger_latch_retirement()
    test_atomic_weights_only_publication_and_no_clobber()
    print("candidate39_episode_residual tests passed")


if __name__ == "__main__":
    main()
