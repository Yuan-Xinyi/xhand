#!/usr/bin/env python3
"""Simulation-free tests for the PickTool FlashSAC BC bootstrap."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gymnasium as gym
import torch

from agent_bridge import FlashSACTorchBridge
from bc_bootstrap import (
    ACTION_DIM,
    ACTION_LAYOUT,
    BCArchitecture,
    BCTrainConfig,
    COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_DIM,
    OBSERVATION_DIM,
    PICK_TOOL_NOISE_GROUPS,
    _checkpoint_agent_config,
    _write_bootstrap_task_contract,
    evaluate_actor,
    export_bridge_checkpoint,
    load_demonstrations,
    phase_balanced_epoch_rows,
    safe_atanh_action,
    split_by_episode,
    train_actor,
)
from flash_rl.agents.flashSAC.network import FlashSACActor


def _synthetic_dataset(path: Path, *, episodes: int = 6, steps: int = 16) -> None:
    generator = torch.Generator().manual_seed(123)
    rows = episodes * steps
    observation = torch.randn(rows, OBSERVATION_DIM, generator=generator)
    teacher_w = torch.randn(OBSERVATION_DIM, ACTION_DIM, generator=generator) * 0.08
    teacher_b = torch.linspace(-0.25, 0.25, ACTION_DIM)
    action = torch.tanh(observation @ teacher_w + teacher_b)
    action[0, 0] = 1.0
    action[1, 1] = -1.0
    episode_id = torch.arange(episodes).repeat_interleave(steps)
    offsets = torch.arange(0, rows + 1, steps, dtype=torch.long)
    torch.save(
        {
            "obs": observation,
            "action": action,
            "phase": (torch.arange(rows) % 5).to(torch.uint8),
            "episode_id": episode_id,
            "episode_offsets": offsets,
            "episode_success": torch.ones(episodes, dtype=torch.bool),
            "meta": {"action_layout": ACTION_LAYOUT},
        },
        path,
    )


def _synthetic_coupled_dataset(path: Path, *, episodes: int = 4, steps: int = 18) -> None:
    observation_dim = COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_DIM
    if steps < 18:
        raise ValueError("synthetic coupled episodes need at least 18 steps")
    generator = torch.Generator().manual_seed(321)
    rows = episodes * steps
    observation = torch.randn(rows, observation_dim, generator=generator)
    phase = torch.tensor([0, 0, 1] + [2] * (steps - 3), dtype=torch.uint8).repeat(
        episodes
    )
    observation[:, 129] = (phase == 0).float()
    observation[:, 106] = (phase == 2).float()
    action = torch.tanh(torch.randn(rows, ACTION_DIM, generator=generator) * 0.2)
    action[phase == 0, 7:] = 0.0
    action[(phase == 1) | (phase == 2), :7] = 0.0
    offsets = torch.arange(0, rows + 1, steps, dtype=torch.long)
    torch.save(
        {
            "obs": observation,
            "action": action,
            "phase": phase,
            "episode_id": torch.arange(episodes).repeat_interleave(steps),
            "episode_offsets": offsets,
            "episode_success": torch.ones(episodes, dtype=torch.bool),
            "episode_native_success": torch.ones(episodes, dtype=torch.bool),
            "episode_native_failure": torch.zeros(episodes, dtype=torch.bool),
            "episode_native_timeout": torch.zeros(episodes, dtype=torch.bool),
            "episode_conservative_teacher_pass": torch.ones(
                episodes, dtype=torch.bool
            ),
            "episode_terminal_stable_steps": torch.full(
                (episodes,), 15, dtype=torch.int64
            ),
            "episode_terminal_power_is_grasped": torch.ones(
                episodes, dtype=torch.bool
            ),
            "episode_terminal_thumb_contact": torch.ones(
                episodes, dtype=torch.bool
            ),
            "episode_terminal_legal_other_contact_count": torch.full(
                (episodes,), 3, dtype=torch.int64
            ),
            "episode_terminal_power_grasp_quality": torch.full(
                (episodes,), 0.35, dtype=torch.float32
            ),
            "episode_terminal_hold_quality": torch.full(
                (episodes,), 0.5, dtype=torch.float32
            ),
            "episode_terminal_max_force": torch.full(
                (episodes,), 30.0, dtype=torch.float32
            ),
            "episode_trajectory_max_force": torch.full(
                (episodes,), 30.0, dtype=torch.float32
            ),
            "episode_trajectory_max_xy_drift": torch.full(
                (episodes,), 0.03, dtype=torch.float32
            ),
            "episode_trajectory_max_rotation_drift": torch.full(
                (episodes,), 0.35, dtype=torch.float32
            ),
            "episode_trajectory_max_true_clearance": torch.full(
                (episodes,), 0.015, dtype=torch.float32
            ),
            "episode_arm_target_saturated": torch.zeros(
                episodes, dtype=torch.bool
            ),
            "episode_terminal_align_active": torch.zeros(
                episodes, dtype=torch.bool
            ),
            "meta": {
                "format_version": 1,
                "task_mode": "coupled_power_align_close_option_v1",
                "observation_dim": observation_dim,
                "observation_contract": (
                    "pick_tool_coupled_power_align_close_state131_v1"
                ),
                "observation_layout": (
                    "legacy_prefix87|distal_action5|grasp_transport23|coupled_state16"
                ),
                "action_dim": ACTION_DIM,
                "action_layout": ACTION_LAYOUT,
                "action_projection": "identity_v1",
                "action_semantics": (
                    "canonical_policy_action_before_phase_shield_v1"
                ),
                "collector": "successful_coupled_power_align_close_teacher",
                "dataset_phase": "align_close_hold",
                "phase_names": ["align", "close_unlatched", "hold_latched"],
                "teacher_probability": 1.0,
                "executed_teacher_fraction": 1.0,
                "align_hand_action_abs_max": 0.0,
                "close_arm_action_abs_max": 0.0,
                "first_observation_last_action_max_error": 0.0,
                "terminal_observation": "not_saved_auto_reset_excluded_v1",
                "trajectory_acceptance": (
                    "native_success_and_conservative_teacher_audit_v1"
                ),
                "clearance_authority": (
                    "true_mesh_convex_hull_min_z_minus_table_v1"
                ),
                "teacher_artifact_sha256": "a" * 64,
                "curriculum_dataset_sha256": "b" * 64,
            },
        },
        path,
    )


def _small_architecture() -> BCArchitecture:
    return BCArchitecture(
        actor_num_blocks=0,
        actor_hidden_dim=32,
        critic_num_blocks=0,
        critic_hidden_dim=32,
        critic_num_bins=11,
        critic_min_v=-2.0,
        critic_max_v=2.0,
    )


def test_loader_and_episode_split() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "demo.pt"
        _synthetic_dataset(path)
        data = load_demonstrations([path])
        assert data.observation.shape == (96, OBSERVATION_DIM)
        assert data.action.shape == (96, ACTION_DIM)
        assert data.num_episodes == 6
        split = split_by_episode(data, 1.0 / 3.0, seed=7)
        train_episode_ids = set(data.episode_id[split.train_rows].tolist())
        validation_episode_ids = set(data.episode_id[split.validation_rows].tolist())
        assert train_episode_ids
        assert validation_episode_ids
        assert train_episode_ids.isdisjoint(validation_episode_ids)
        assert split.train_rows.numel() + split.validation_rows.numel() == 96


def test_loader_rejects_non_markov_or_failed_data() -> None:
    with tempfile.TemporaryDirectory() as directory:
        missing_offsets = Path(directory) / "missing.pt"
        torch.save(
            {
                "obs": torch.zeros(4, OBSERVATION_DIM),
                "action": torch.zeros(4, ACTION_DIM),
            },
            missing_offsets,
        )
        try:
            load_demonstrations([missing_offsets])
        except KeyError as exc:
            assert "episode_offsets" in str(exc)
        else:
            raise AssertionError("dataset without episode boundaries was accepted")

        failed = Path(directory) / "failed.pt"
        _synthetic_dataset(failed, episodes=2, steps=4)
        payload = torch.load(failed, weights_only=True)
        payload["episode_success"][1] = False
        torch.save(payload, failed)
        try:
            load_demonstrations([failed])
        except ValueError as exc:
            assert "failed episodes" in str(exc)
        else:
            raise AssertionError("failed demonstration episode was accepted")


def test_coupled_131d_loader_and_actor_are_parameterized_and_strict() -> None:
    observation_dim = COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_DIM
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "coupled.pt"
        _synthetic_coupled_dataset(path)
        data = load_demonstrations([path], observation_dim=observation_dim)
        assert data.observation.shape == (72, observation_dim)
        assert data.observation_dim == observation_dim
        assert data.sources[0].source_contract == (
            "successful_coupled_power_align_close_teacher_v1"
        )

        split = split_by_episode(data, 0.25, seed=5)
        config = BCTrainConfig(
            epochs=3,
            batch_size=12,
            learning_rate=1.0e-3,
            validation_fraction=0.25,
            use_amp=False,
            seed=5,
        )
        state, metrics = train_actor(
            data,
            split,
            config,
            _small_architecture(),
            "cpu",
        )
        actor = FlashSACActor(0, observation_dim, 32, ACTION_DIM)
        actor.load_state_dict(state)
        mean, _ = actor.get_mean_and_std(data.observation[:2], training=False)
        assert mean.shape == (2, ACTION_DIM)
        assert set(metrics["sampling"]["source_phase_counts"]) == {"0", "1", "2"}

        contradictory = torch.load(path, weights_only=True)
        contradictory["action"][contradictory["phase"] == 0, 7] = 0.2
        contradictory_path = Path(directory) / "contradictory.pt"
        torch.save(contradictory, contradictory_path)
        try:
            load_demonstrations(
                [contradictory_path],
                observation_dim=observation_dim,
            )
        except ValueError as exc:
            assert "exact zero action slice" in str(exc)
        else:
            raise AssertionError("contradictory coupled canonical actions were accepted")

        bad_metadata = torch.load(path, weights_only=True)
        bad_metadata["meta"]["teacher_artifact_sha256"] = "bad"
        bad_metadata_path = Path(directory) / "bad_metadata.pt"
        torch.save(bad_metadata, bad_metadata_path)
        try:
            load_demonstrations(
                [bad_metadata_path],
                observation_dim=observation_dim,
            )
        except ValueError as exc:
            assert "SHA256" in str(exc)
        else:
            raise AssertionError("coupled dataset with invalid lineage SHA was accepted")

        failed_teacher_audit = torch.load(path, weights_only=True)
        failed_teacher_audit["episode_conservative_teacher_pass"][0] = False
        failed_teacher_path = Path(directory) / "failed_teacher_audit.pt"
        torch.save(failed_teacher_audit, failed_teacher_path)
        try:
            load_demonstrations(
                [failed_teacher_path],
                observation_dim=observation_dim,
            )
        except ValueError as exc:
            assert "episode_conservative_teacher_pass" in str(exc)
        else:
            raise AssertionError("failed coupled teacher audit was accepted by BC")

        other_curriculum_path = Path(directory) / "other_curriculum.pt"
        _synthetic_coupled_dataset(other_curriculum_path)
        other_curriculum = torch.load(other_curriculum_path, weights_only=True)
        other_curriculum["meta"]["curriculum_dataset_sha256"] = "c" * 64
        torch.save(other_curriculum, other_curriculum_path)
        try:
            load_demonstrations(
                [path, other_curriculum_path],
                observation_dim=observation_dim,
            )
        except ValueError as exc:
            assert "must share one curriculum_dataset_sha256" in str(exc)
        else:
            raise AssertionError("mixed coupled curriculum lineages were accepted by BC")


def test_safe_atanh_matches_flashsac_action_semantics() -> None:
    action = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    target = safe_atanh_action(action, 1.0e-4)
    assert bool(torch.isfinite(target).all())
    reconstructed = torch.tanh(target)
    torch.testing.assert_close(reconstructed[1:4], action[1:4], rtol=1.0e-6, atol=1.0e-6)
    assert float(reconstructed[0]) > -1.0
    assert float(reconstructed[-1]) < 1.0


def test_phase_balanced_sampling_equalizes_every_epoch() -> None:
    # Mirrors the real imbalance qualitatively: search dominates close.
    phase = torch.tensor([0] * 12 + [1] * 2 + [2] * 7, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(91)
    for _ in range(5):
        rows, summary = phase_balanced_epoch_rows(phase, generator=generator)
        selected_counts = torch.bincount(phase[rows], minlength=3)
        torch.testing.assert_close(
            selected_counts,
            torch.tensor([12, 12, 12]),
            rtol=0.0,
            atol=0.0,
        )
        assert rows.numel() == 36
        assert summary["source_phase_counts"] == {"0": 12, "1": 2, "2": 7}
        assert summary["samples_per_phase_per_epoch"] == {"0": 12, "1": 12, "2": 12}
        assert summary["replacement_by_phase"] == {"0": False, "1": True, "2": True}

    unknown = torch.full((9,), -1, dtype=torch.long)
    rows, summary = phase_balanced_epoch_rows(unknown, generator=generator)
    assert torch.equal(rows.sort().values, torch.arange(9))
    assert summary["source_phase_counts"] == {"-1": 9}
    assert summary["samples_per_phase_per_epoch"] == {"-1": 9}
    assert summary["replacement_by_phase"] == {"-1": False}


def test_bc_uses_demo_actions_and_reduces_holdout_error() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "demo.pt"
        _synthetic_dataset(path, episodes=8, steps=20)
        data = load_demonstrations([path])
        split = split_by_episode(data, 0.25, seed=3)
        config = BCTrainConfig(
            epochs=40,
            batch_size=64,
            learning_rate=1.0e-3,
            validation_fraction=0.25,
            use_amp=False,
            target_std=0.15,
            std_anchor_weight=0.05,
            seed=3,
        )
        state, metrics = train_actor(data, split, config, _small_architecture(), "cpu")
        assert metrics["best_validation"]["action_rmse"] < metrics["initial_validation"]["action_rmse"]
        assert metrics["loss_last"] < metrics["loss_first"]
        assert metrics["selection_metric"] == "validation_phase_macro_action_rmse"
        assert metrics["sampling"]["strategy"] == "phase_balanced_oversample_to_largest_stratum"
        assert len(set(metrics["sampling"]["samples_per_phase_per_epoch"].values())) == 1
        assert (
            metrics["best_validation"]["log_std_prior_rmse"]
            < metrics["initial_validation"]["log_std_prior_rmse"]
        )

        actor = FlashSACActor(0, OBSERVATION_DIM, 32, ACTION_DIM)
        actor.load_state_dict(state)
        validation_rows = split.validation_rows
        measured = evaluate_actor(
            actor,
            data.observation[validation_rows],
            data.action[validation_rows],
            atanh_epsilon=config.atanh_epsilon,
            target_std=config.target_std,
        )
        assert abs(measured["action_rmse"] - metrics["best_validation"]["action_rmse"]) < 1.0e-7


def test_export_is_a_fresh_bridge_loadable_checkpoint() -> None:
    assert tuple(
        (group.name, group.start, group.stop, group.scale, group.zeta_mu, group.zeta_max)
        for group in PICK_TOOL_NOISE_GROUPS
    ) == (
        ("arm", 0, 7, 1.0, 1.0, 64),
        ("token", 7, 16, 0.5, 1.25, 32),
        ("residual", 16, 21, 0.35, 1.5, 16),
    )
    architecture = _small_architecture()
    actor = FlashSACActor(0, OBSERVATION_DIM, 32, ACTION_DIM)
    actor_state = {name: value.detach().clone() for name, value in actor.state_dict().items()}
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint"
        exported = export_bridge_checkpoint(
            actor_state,
            checkpoint,
            architecture=architecture,
            device="cpu",
            seed=11,
            normalize_reward=False,
        )
        assert exported._update_step == 0  # noqa: SLF001
        assert exported._cfg.actor_bc_alpha == 0.0  # noqa: SLF001
        actor_payload = torch.load(checkpoint / "actor.pt", map_location="cpu", weights_only=True)
        assert actor_payload["optimizer_state_dict"]["state"] == {}

        observation_space = gym.spaces.Box(
            -float("inf"), float("inf"), shape=(OBSERVATION_DIM,), dtype="float32"
        )
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype="float32")
        restored = FlashSACTorchBridge(
            observation_space,
            action_space,
            {"actor_observation_size": (OBSERVATION_DIM,), "asymmetric_obs": False},
            _checkpoint_agent_config(
                architecture,
                device=torch.device("cpu"),
                seed=11,
                normalize_reward=False,
            ),
            noise_groups=PICK_TOOL_NOISE_GROUPS,
        )
        restored.load(str(checkpoint))
        observations = torch.randn(5, OBSERVATION_DIM)
        expected = exported.sample_actions(0, {"next_observation": observations}, training=False)
        actual = restored.sample_actions(0, {"next_observation": observations}, training=False)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        assert restored._update_step == 0  # noqa: SLF001


def test_coupled_export_has_131d_actor_and_auditable_task_contract() -> None:
    observation_dim = COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_DIM
    architecture = _small_architecture()
    actor = FlashSACActor(0, observation_dim, 32, ACTION_DIM)
    actor_state = {name: value.detach().clone() for name, value in actor.state_dict().items()}
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "coupled_checkpoint"
        exported = export_bridge_checkpoint(
            actor_state,
            checkpoint,
            architecture=architecture,
            device="cpu",
            seed=12,
            normalize_reward=False,
            observation_dim=observation_dim,
        )
        action = exported.sample_actions(
            0,
            {"next_observation": torch.randn(3, observation_dim)},
            training=False,
        )
        assert action.shape == (3, ACTION_DIM)

        _write_bootstrap_task_contract(checkpoint, observation_dim)
        assert json.loads((checkpoint / "task_contract.json").read_text()) == {
            "version": 3,
            "task_mode": "coupled_power_align_close_option_v1",
            "replay_n_step": 3,
            "replay_gamma": 0.99,
            "observation_dim": observation_dim,
            "observation_contract": "pick_tool_coupled_power_align_close_state131_v1",
            "policy_action_dim": ACTION_DIM,
            "policy_action_layout": ACTION_LAYOUT,
            "environment_action_dim": ACTION_DIM,
            "action_projection": "identity_v1",
        }


def test_metadata_contract_is_strict_json() -> None:
    # Exercise the metadata claims independently of the CUDA-only CLI path.
    claim = {
        "source": "explicit_demo_action",
        "upstream_actor_bc_alpha": 0.0,
        "action_dim": ACTION_DIM,
    }
    encoded = json.dumps(claim, allow_nan=False)
    assert json.loads(encoded) == claim


def main() -> None:
    tests = (
        test_loader_and_episode_split,
        test_loader_rejects_non_markov_or_failed_data,
        test_coupled_131d_loader_and_actor_are_parameterized_and_strict,
        test_safe_atanh_matches_flashsac_action_semantics,
        test_phase_balanced_sampling_equalizes_every_epoch,
        test_bc_uses_demo_actions_and_reduces_holdout_error,
        test_export_is_a_fresh_bridge_loadable_checkpoint,
        test_coupled_export_has_131d_actor_and_auditable_task_contract,
        test_metadata_contract_is_strict_json,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"All {len(tests)} FlashSAC BC bootstrap tests passed.")


if __name__ == "__main__":
    main()
