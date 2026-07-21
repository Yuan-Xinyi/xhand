"""Simulation-free regression tests for :mod:`agent_bridge`.

Run with the Isaac Lab Python environment; JAX is intentionally not required::

    TORCHDYNAMO_DISABLE=1 ./isaaclab.sh -p scripts/flashsac/agent_bridge_test.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

from agent_bridge import (
    ActionNoiseGroup,
    BRIDGE_STATE_FILENAME,
    FlashSACTorchBridge,
    assert_transition_tensors,
    build_agent_config,
)


OBSERVATION_DIM = 7
ACTION_DIM = 6
NUM_ENVS = 4


def _spaces() -> tuple[gym.spaces.Box, gym.spaces.Box]:
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(OBSERVATION_DIM,), dtype="float32")
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype="float32")
    return observation_space, action_space


def _config(**overrides: Any):
    defaults = {
        "device_type": "cpu",
        "buffer_device_type": "cpu",
        "buffer_max_length": 64,
        "buffer_min_length": NUM_ENVS,
        "sample_batch_size": NUM_ENVS,
        "actor_noise_zeta_max": 4,
    }
    defaults.update(overrides)
    return build_agent_config(**defaults)


def _groups() -> tuple[ActionNoiseGroup, ...]:
    return (
        ActionNoiseGroup("arm", 0, 2, scale=0.0, zeta_max=1),
        ActionNoiseGroup("token", 2, 5, scale=0.5, zeta_max=3),
        ActionNoiseGroup("residual", 5, 6, scale=0.1, zeta_max=2),
    )


def _agent(**config_overrides: Any) -> FlashSACTorchBridge:
    observation_space, action_space = _spaces()
    return FlashSACTorchBridge(
        observation_space,
        action_space,
        {},
        _config(**config_overrides),
        noise_groups=_groups(),
    )


def _transition(observation: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
    num_envs = observation.shape[0]
    return {
        "observation": observation.clone(),
        "action": action.clone(),
        "reward": torch.arange(num_envs, dtype=torch.float32, device=observation.device),
        "terminated": torch.zeros(num_envs, dtype=torch.bool, device=observation.device),
        "truncated": torch.zeros(num_envs, dtype=torch.bool, device=observation.device),
        "next_observation": observation.add(0.25),
    }


def _expect_error(error_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def test_actions_stay_in_torch_and_group_scales_apply() -> None:
    torch.manual_seed(10)
    agent = _agent()
    observations = torch.randn(NUM_ENVS, OBSERVATION_DIM)
    with torch.no_grad():
        mean, _ = agent._actor.apply(  # noqa: SLF001 - white-box bridge regression
            "get_mean_and_std", observations=observations, training=False
        )
        deterministic = torch.tanh(mean)

    actions = agent.sample_actions(1, {"next_observation": observations}, training=True)
    assert isinstance(actions, torch.Tensor)
    assert actions.device == observations.device == agent.device
    assert actions.shape == (NUM_ENVS, ACTION_DIM)
    # The zero-scale arm slice must be exactly deterministic even in train mode.
    torch.testing.assert_close(actions[:, :2], deterministic[:, :2], rtol=0.0, atol=0.0)
    assert not torch.equal(actions[:, 2:], deterministic[:, 2:])

    cached_noise = agent._cached_noise.clone()  # noqa: SLF001
    repeat_count = agent._cur_noise_repeat_count.clone()  # noqa: SLF001
    _ = agent.sample_actions(2, {"next_observation": observations}, training=False)
    # Deterministic evaluation must not advance or replace exploration state.
    torch.testing.assert_close(agent._cached_noise, cached_noise, rtol=0.0, atol=0.0)  # noqa: SLF001
    torch.testing.assert_close(agent._cur_noise_repeat_count, repeat_count, rtol=0.0, atol=0.0)  # noqa: SLF001

    agent.reset_exploration()
    assert agent._cached_noise.shape == (0, ACTION_DIM)  # noqa: SLF001
    assert torch.count_nonzero(agent._cur_noise_repeat_count) == 0  # noqa: SLF001


def test_partial_reset_refreshes_only_completed_envs() -> None:
    torch.manual_seed(101)
    agent = _agent()
    observations = torch.randn(NUM_ENVS, OBSERVATION_DIM)
    _ = agent.sample_actions(1, {"next_observation": observations}, training=True)
    before = agent._cached_noise.clone()  # noqa: SLF001
    repeat_count = agent._cur_noise_repeat_count.clone()  # noqa: SLF001
    reset_ids = torch.tensor([1, 3], dtype=torch.long)
    agent.reset_exploration(env_ids=reset_ids)
    after = agent._cached_noise  # noqa: SLF001
    torch.testing.assert_close(after[[0, 2]], before[[0, 2]], rtol=0.0, atol=0.0)
    assert not torch.equal(after[reset_ids], before[reset_ids])
    torch.testing.assert_close(  # noqa: SLF001
        agent._cur_noise_repeat_count, repeat_count, rtol=0.0, atol=0.0
    )


def test_transition_contract_and_replay_are_torch_native() -> None:
    torch.manual_seed(11)
    agent = _agent()
    observation = torch.randn(NUM_ENVS, OBSERVATION_DIM)
    action = torch.randn(NUM_ENVS, ACTION_DIM).clamp(-1.0, 1.0)
    transition = _transition(observation, action)
    assert_transition_tensors(
        transition,
        device=agent.device,
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
    )
    agent.process_transition(transition)
    assert agent.can_start_training()
    stored = agent._replay_buffer.get_observations()  # noqa: SLF001
    assert isinstance(stored, torch.Tensor)
    assert stored.device == agent.device
    torch.testing.assert_close(stored, observation)

    invalid = dict(transition)
    invalid["action"] = action.tolist()
    _expect_error(TypeError, agent.process_transition, invalid)


def test_cuda_interaction_has_no_host_round_trip() -> None:
    if not torch.cuda.is_available():
        return
    torch.manual_seed(111)
    observation_space, action_space = _spaces()
    agent = FlashSACTorchBridge(
        observation_space,
        action_space,
        {},
        _config(device_type="cuda:0", buffer_device_type="cuda:0"),
        noise_groups=_groups(),
    )
    observation = torch.randn(NUM_ENVS, OBSERVATION_DIM, device="cuda:0")
    action = agent.sample_actions(1, {"next_observation": observation}, training=True)
    assert action.is_cuda and action.device == observation.device
    transition = _transition(observation, action)
    agent.process_transition(transition)
    stored = agent._replay_buffer.get_observations()  # noqa: SLF001
    assert stored.is_cuda and stored.device == observation.device


def test_checkpoint_exactly_restores_noise_and_rng() -> None:
    torch.manual_seed(12)
    agent = _agent()
    observations = torch.randn(NUM_ENVS, OBSERVATION_DIM)
    previous = {"next_observation": observations}
    _ = agent.sample_actions(1, previous, training=True)
    agent._update_step = 7  # noqa: SLF001 - ensure upstream agent state is restored

    with tempfile.TemporaryDirectory(prefix="flashsac_bridge_test_") as directory:
        checkpoint = Path(directory) / "checkpoint"
        agent.save(str(checkpoint))
        assert (checkpoint / BRIDGE_STATE_FILENAME).is_file()

        expected_next_action = agent.sample_actions(2, previous, training=True)
        expected_noise = agent._cached_noise.clone()  # noqa: SLF001
        expected_count = agent._cur_noise_repeat_count.clone()  # noqa: SLF001

        # Construction consumes RNG and starts from unrelated network weights.
        restored = _agent()
        _ = torch.randn(101)
        restored.load(str(checkpoint))
        actual_next_action = restored.sample_actions(2, previous, training=True)

        torch.testing.assert_close(actual_next_action, expected_next_action, rtol=0.0, atol=0.0)
        torch.testing.assert_close(restored._cached_noise, expected_noise, rtol=0.0, atol=0.0)  # noqa: SLF001
        torch.testing.assert_close(  # noqa: SLF001
            restored._cur_noise_repeat_count, expected_count, rtol=0.0, atol=0.0
        )
        assert restored._update_step == 7  # noqa: SLF001


def test_group_partition_is_validated() -> None:
    observation_space, action_space = _spaces()
    overlapping = (
        ActionNoiseGroup("left", 0, 4),
        ActionNoiseGroup("right", 3, ACTION_DIM),
    )
    _expect_error(
        ValueError,
        FlashSACTorchBridge,
        observation_space,
        action_space,
        {},
        _config(),
        noise_groups=overlapping,
    )


def main() -> None:
    test_actions_stay_in_torch_and_group_scales_apply()
    print("[PASS] torch actions and grouped exploration")
    test_transition_contract_and_replay_are_torch_native()
    print("[PASS] torch transition/replay contract")
    test_partial_reset_refreshes_only_completed_envs()
    print("[PASS] per-environment exploration reset")
    test_cuda_interaction_has_no_host_round_trip()
    print("[PASS] CUDA interaction stays on device" if torch.cuda.is_available() else "[SKIP] CUDA unavailable")
    test_checkpoint_exactly_restores_noise_and_rng()
    print("[PASS] exact checkpoint continuation")
    test_group_partition_is_validated()
    print("[PASS] noise-group validation")
    print("All FlashSAC Torch bridge tests passed.")


if __name__ == "__main__":
    main()
