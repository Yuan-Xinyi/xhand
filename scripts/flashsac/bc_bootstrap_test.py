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
    OBSERVATION_DIM,
    PICK_TOOL_NOISE_GROUPS,
    _checkpoint_agent_config,
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
        test_safe_atanh_matches_flashsac_action_semantics,
        test_phase_balanced_sampling_equalizes_every_epoch,
        test_bc_uses_demo_actions_and_reduces_holdout_error,
        test_export_is_a_fresh_bridge_loadable_checkpoint,
        test_metadata_contract_is_strict_json,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"All {len(tests)} FlashSAC BC bootstrap tests passed.")


if __name__ == "__main__":
    main()
