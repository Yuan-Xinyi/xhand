#!/usr/bin/env python3
"""CPU-only tests for the V6 success self-imitation artifact contract."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
from typing import Any, Callable

import torch

from v6_success_self_imitation import (
    ACTION_DIM,
    ARM_ACTION_DIM,
    COHORT_ASSIGNMENT_SALT,
    EXPLORATORY_HAND_NOISE_SCALE,
    OBSERVATION_DIM,
    PUBLIC_LATCH_INDEX,
    REPORT_KIND,
    REQUIRED_METADATA,
    build_cohort_noise_scale,
    exploratory_cohort_mask,
    manifest_sha256,
    publish_dataset_and_report_no_clobber,
    public_latch_transition_masks,
    select_executed_action,
    summarize_dataset,
    validate_dataset,
)


def _expect(error: type[BaseException], function: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    try:
        function(*args, **kwargs)
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def _payload() -> dict[str, Any]:
    seed = 700
    num_envs = 4
    replicate = "a"
    exploratory = exploratory_cohort_mask(
        seed=seed, num_envs=num_envs, replicate=replicate
    )
    deterministic_ids = (~exploratory).nonzero(as_tuple=False).flatten()
    exploratory_ids = exploratory.nonzero(as_tuple=False).flatten()
    retained_ids = torch.sort(
        torch.stack((deterministic_ids[0], exploratory_ids[0]))
    ).values

    trigger_step = torch.full((num_envs,), 3, dtype=torch.int64)
    first_latch_step = torch.full((num_envs,), -1, dtype=torch.int64)
    first_latch_step[retained_ids] = torch.tensor(
        [5 + index for index in range(retained_ids.numel())], dtype=torch.int64
    )
    episode_length = torch.full((num_envs,), 999, dtype=torch.int64)
    episode_length[retained_ids] = 10
    native_success = torch.zeros(num_envs, dtype=torch.bool)
    native_success[retained_ids] = True
    failure = torch.zeros(num_envs, dtype=torch.bool)
    time_out = ~(native_success | failure)
    ever_latched = torch.zeros(num_envs, dtype=torch.bool)
    ever_latched[retained_ids] = True
    ever_grasped = ever_latched.clone()
    max_clearance = torch.zeros(num_envs, dtype=torch.float32)
    max_clearance[retained_ids] = torch.tensor([0.21, 0.23])
    ever_20cm = max_clearance >= 0.20
    trajectory_force = torch.tensor([8.0, 9.0, 10.0, 11.0])
    retained = native_success & ever_latched & ever_grasped & ever_20cm

    episode_obs: list[torch.Tensor] = []
    episode_action: list[torch.Tensor] = []
    episode_ids: list[torch.Tensor] = []
    episode_steps: list[torch.Tensor] = []
    source_steps: list[torch.Tensor] = []
    offsets = [0]
    for episode, env_id_value in enumerate(retained_ids.tolist()):
        env_id = int(env_id_value)
        sources = torch.arange(
            int(trigger_step[env_id]), int(first_latch_step[env_id]) + 1
        )
        rows = sources.numel()
        obs = torch.zeros((rows, OBSERVATION_DIM), dtype=torch.float32)
        obs[:, PUBLIC_LATCH_INDEX] = 0.0
        action = torch.zeros((rows, ACTION_DIM), dtype=torch.float32)
        action[:, ARM_ACTION_DIM:] = (episode + 1) * 0.1
        episode_obs.append(obs)
        episode_action.append(action)
        episode_ids.append(torch.full((rows,), episode, dtype=torch.int64))
        episode_steps.append(torch.arange(rows, dtype=torch.int64))
        source_steps.append(sources)
        offsets.append(offsets[-1] + rows)

    source_sha = {"source.py": "1" * 64}
    asset_sha = {"asset.usd": "2" * 64}
    metadata = {
        **REQUIRED_METADATA,
        "seed": seed,
        "num_envs": num_envs,
        "min_retained": 2,
        "cohort_assignment_salt": COHORT_ASSIGNMENT_SALT,
        "cohort_replicate": replicate,
        "v6_checkpoint": "/checkpoint/v6",
        "search_checkpoint": "/checkpoint/search.pth",
        "search_checkpoint_sha256": "3" * 64,
        "v6_actor_sha256": "4" * 64,
        "v6_task_contract_sha256": "5" * 64,
        "v6_bridge_state_sha256": "6" * 64,
        "frozen_lift_actor_sha256": "7" * 64,
        "frozen_lift_semantic_sha256": "8" * 64,
        "frozen_lift_source_actor_sha256": "9" * 64,
        "source_manifest_sha256": manifest_sha256(source_sha),
        "runtime_asset_manifest_sha256": manifest_sha256(asset_sha),
        "source_sha256": source_sha,
        "runtime_asset_sha256": asset_sha,
        "flashsac_upstream_commit": "a" * 40,
        "flashsac_fork_commit": "b" * 40,
        "git": {
            "commit": "c" * 40,
            "branch": "test",
            "source_files_dirty": False,
            "flashsac_commit": "b" * 40,
            "flashsac_dirty": False,
        },
        "runtime": {"seed": seed},
    }
    trial_cohort = exploratory.to(torch.int64)
    payload: dict[str, Any] = {
        "obs": torch.cat(episode_obs),
        "action": torch.cat(episode_action),
        "phase": torch.ones(offsets[-1], dtype=torch.uint8),
        "episode_id": torch.cat(episode_ids),
        "step": torch.cat(episode_steps),
        "source_step": torch.cat(source_steps),
        "episode_offsets": torch.tensor(offsets, dtype=torch.int64),
        "episode_success": torch.ones(2, dtype=torch.bool),
        "episode_native_success": native_success[retained_ids],
        "episode_dropped": torch.zeros(2, dtype=torch.bool),
        "episode_unsafe_force": torch.zeros(2, dtype=torch.bool),
        "episode_unlatched_clearance_ge_5cm": torch.zeros(2, dtype=torch.bool),
        "episode_ever_grasped": ever_grasped[retained_ids],
        "episode_triggered": torch.ones(2, dtype=torch.bool),
        "episode_ever_latched": ever_latched[retained_ids],
        "episode_latch_released_after_first": torch.zeros(2, dtype=torch.bool),
        "episode_source_env_id": retained_ids,
        "episode_cohort": trial_cohort[retained_ids],
        "episode_trigger_step": trigger_step[retained_ids],
        "episode_first_latch_step": first_latch_step[retained_ids],
        "episode_terminal_step": (episode_length - 1)[retained_ids],
        "episode_hand_noise_scale": trial_cohort[retained_ids].float()
        * EXPLORATORY_HAND_NOISE_SCALE,
        "episode_trigger_score": torch.tensor(
            [0.32 + 0.10 * index for index in range(2)], dtype=torch.float32
        ),
        "episode_max_true_clearance_m": max_clearance[retained_ids],
        "episode_trajectory_max_force": trajectory_force[retained_ids],
        "trial_env_slot": torch.arange(num_envs, dtype=torch.int64),
        "trial_cohort": trial_cohort,
        "trial_episode_length": episode_length,
        "trial_trigger_step": trigger_step,
        "trial_first_latch_step": first_latch_step,
        "trial_terminal_step": episode_length - 1,
        "trial_hand_noise_scale": trial_cohort.float()
        * EXPLORATORY_HAND_NOISE_SCALE,
        "trial_trigger_score": torch.tensor([0.32, 0.35, 0.42, 0.31]),
        "trial_max_true_clearance_m": max_clearance,
        "trial_trajectory_max_force": trajectory_force,
        "trial_triggered": torch.ones(num_envs, dtype=torch.bool),
        "trial_ever_latched": ever_latched,
        "trial_latch_released_after_first": torch.zeros(num_envs, dtype=torch.bool),
        "trial_native_success": native_success,
        "trial_failure": failure,
        "trial_time_out": time_out,
        "trial_dropped": torch.zeros(num_envs, dtype=torch.bool),
        "trial_unsafe_force": torch.zeros(num_envs, dtype=torch.bool),
        "trial_unlatched_clearance_ge_5cm": torch.zeros(num_envs, dtype=torch.bool),
        "trial_ever_grasped": ever_grasped,
        "trial_ever_clearance_ge_20cm": ever_20cm,
        "trial_retained": retained,
        "meta": metadata,
    }
    # Bind retained trigger scores to the full trial evidence regardless of
    # which stable slots the hash assignment selected.
    payload["episode_trigger_score"] = payload["trial_trigger_score"][retained_ids]
    return payload


def test_assignment_and_noise_contract() -> None:
    a = exploratory_cohort_mask(seed=13, num_envs=8, replicate="a")
    b = exploratory_cohort_mask(seed=13, num_envs=8, replicate="b")
    assert int(a.sum()) == 4
    assert torch.equal(a, ~b)
    assert torch.equal(a, exploratory_cohort_mask(seed=13, num_envs=8, replicate="a"))

    obs = torch.zeros((4, OBSERVATION_DIM), dtype=torch.float32)
    obs[2:, PUBLIC_LATCH_INDEX] = 1.0
    exploratory = torch.tensor([False, True, False, True])
    scale = build_cohort_noise_scale(obs, exploratory)
    assert torch.equal(scale[:, :ARM_ACTION_DIM], torch.zeros_like(scale[:, :ARM_ACTION_DIM]))
    assert float(scale[0].abs().max()) == 0.0
    assert torch.equal(
        scale[1, ARM_ACTION_DIM:],
        torch.full((ACTION_DIM - ARM_ACTION_DIM,), EXPLORATORY_HAND_NOISE_SCALE),
    )
    assert float(scale[2:].abs().max()) == 0.0

    search = torch.full((4, ACTION_DIM), -0.2)
    routed = torch.full_like(search, 0.4)
    selected = select_executed_action(
        search_action=search,
        routed_v6_action=routed,
        option_active=torch.tensor([False, True, False, True]),
    )
    torch.testing.assert_close(selected[0], search[0])
    torch.testing.assert_close(selected[1], routed[1])


def test_public_latch_transition_uses_post_action_observation_clock() -> None:
    obs = torch.zeros((4, OBSERVATION_DIM), dtype=torch.float32)
    obs[0, PUBLIC_LATCH_INDEX] = 1.0
    obs[1, PUBLIC_LATCH_INDEX] = 1.0
    obs[3, PUBLIC_LATCH_INDEX] = 1.0
    newly, released = public_latch_transition_masks(
        transition_observation=obs,
        active_before=torch.tensor([True, True, True, True]),
        ever_latched_before=torch.tensor([False, True, True, False]),
    )
    # Row 0's current action caused the first observable latch edge.  Row 2
    # released a previously seen latch.  Row 3 demonstrates that the adapter's
    # captured transition observation remains authoritative on a done row.
    assert torch.equal(newly, torch.tensor([True, False, False, True]))
    assert torch.equal(released, torch.tensor([False, False, True, False]))
    episode_step = torch.full((4,), 5, dtype=torch.int64)
    first_before = torch.full((4,), -1, dtype=torch.int64)
    first_after = torch.where(newly, episode_step, first_before)
    assert int(first_after[0]) == 5
    assert torch.equal(torch.arange(3, int(first_after[0]) + 1), torch.tensor([3, 4, 5]))

    malformed = obs.clone()
    malformed[0, PUBLIC_LATCH_INDEX] = 0.5
    _expect(
        ValueError,
        public_latch_transition_masks,
        transition_observation=malformed,
        active_before=torch.ones(4, dtype=torch.bool),
        ever_latched_before=torch.zeros(4, dtype=torch.bool),
    )


def test_dataset_cross_tensor_contract_and_buckets() -> None:
    payload = _payload()
    validated = validate_dataset(payload)
    assert validated["obs"].shape[0] == int(validated["episode_offsets"][-1])
    summary = summarize_dataset(validated)
    assert summary["deterministic"]["all_assigned"]["episodes"] == 2
    assert summary["exploratory"]["all_assigned"]["episodes"] == 2
    bucket_total = sum(
        entry["episodes"]
        for entry in summary["exploratory"]["trigger_score_buckets"].values()
    )
    assert bucket_total == 2

    malformed = copy.deepcopy(payload)
    malformed["action"][0, 0] = 0.01
    _expect(ValueError, validate_dataset, malformed)
    malformed = copy.deepcopy(payload)
    malformed["obs"][0, PUBLIC_LATCH_INDEX] = 1.0
    _expect(ValueError, validate_dataset, malformed)
    malformed = copy.deepcopy(payload)
    malformed["episode_hand_noise_scale"][0] = 0.123
    _expect(ValueError, validate_dataset, malformed)
    malformed = copy.deepcopy(payload)
    retained_id = int(malformed["episode_source_env_id"][0])
    malformed["trial_trajectory_max_force"][retained_id] = 30.1
    malformed["episode_trajectory_max_force"][0] = 30.1
    _expect(ValueError, validate_dataset, malformed)


def test_transactional_publication_is_weights_only_and_no_clobber() -> None:
    payload = _payload()
    report = {
        "kind": REPORT_KIND,
        "status": "complete",
        "collector": payload["meta"]["collector"],
        "seed": payload["meta"]["seed"],
        "cohort_replicate": payload["meta"]["cohort_replicate"],
        "num_envs": payload["meta"]["num_envs"],
        "vector_steps": int(payload["trial_episode_length"].max()),
        "retained_episodes": int(payload["trial_retained"].sum()),
        "transitions": int(payload["obs"].shape[0]),
        "v6_actor_sha256": payload["meta"]["v6_actor_sha256"],
        "search_checkpoint_sha256": payload["meta"]["search_checkpoint_sha256"],
        "summary": summarize_dataset(payload),
    }
    with tempfile.TemporaryDirectory(prefix="v6_self_imitation_") as directory:
        dataset_output = Path(directory) / "dataset.pt"
        report_output = Path(directory) / "dataset.json"
        digest = publish_dataset_and_report_no_clobber(
            payload,
            report,
            dataset_output=dataset_output,
            report_output=report_output,
        )
        assert len(digest) == 64
        loaded = torch.load(dataset_output, map_location="cpu", weights_only=True)
        validate_dataset(loaded)
        _expect(
            FileExistsError,
            publish_dataset_and_report_no_clobber,
            payload,
            report,
            dataset_output=dataset_output,
            report_output=report_output,
        )


def main() -> None:
    test_assignment_and_noise_contract()
    test_public_latch_transition_uses_post_action_observation_clock()
    test_dataset_cross_tensor_contract_and_buckets()
    test_transactional_publication_is_weights_only_and_no_clobber()
    print("v6 success self-imitation contract tests passed")


if __name__ == "__main__":
    main()
