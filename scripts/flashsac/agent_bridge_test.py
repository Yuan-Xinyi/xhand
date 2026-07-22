"""Simulation-free regression tests for :mod:`agent_bridge`.

Run with the Isaac Lab Python environment; JAX is intentionally not required.
Do not disable TorchDynamo: one test exercises a real ``torch.compile`` wrapper::

    ./isaaclab.sh -p scripts/flashsac/agent_bridge_test.py
"""

from __future__ import annotations

import copy
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
import torch.nn.functional as F

from agent_bridge import (
    ActionAuthorityRule,
    ActionNoiseGroup,
    BRIDGE_STATE_FILENAME,
    FROZEN_LIFT_ACTOR_FILENAME,
    FlashSACTorchBridge,
    PUBLIC_LATCH_FROZEN_ACTOR_ROUTER_NAME,
    PublicLatchFrozenActorRouter,
    assert_transition_tensors,
    build_agent_config,
)


OBSERVATION_DIM = 7
ACTION_DIM = 6
NUM_ENVS = 4
AUTHORITY_OBSERVATION_DIM = 131
AUTHORITY_ACTION_DIM = 14
ALIGN_ACTIVE_OBSERVATION_INDEX = 129
PUBLIC_LATCH_OBSERVATION_INDEX = OBSERVATION_DIM - 1
ROUTER_ACTION_DIM = 21
ROUTER_ARM_STOP = 7


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


def _agent_with_action_dim(
    action_dim: int,
    *,
    use_compile: bool = False,
    unit_normalize_actor_mean_head: bool = True,
) -> FlashSACTorchBridge:
    observation_space = gym.spaces.Box(
        -1.0, 1.0, shape=(OBSERVATION_DIM,), dtype="float32"
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype="float32")
    return FlashSACTorchBridge(
        observation_space,
        action_space,
        {},
        _config(use_compile=use_compile),
        noise_groups=(ActionNoiseGroup("all", 0, action_dim),),
        unit_normalize_actor_mean_head=unit_normalize_actor_mean_head,
    )


def _slice_authority_agent(**config_overrides: Any) -> FlashSACTorchBridge:
    observation_space, action_space = _spaces()
    return FlashSACTorchBridge(
        observation_space,
        action_space,
        {},
        _config(actor_update_period=1, **config_overrides),
        noise_groups=_groups(),
        action_authority_rules=(
            ActionAuthorityRule(
                "arm_after_public_latch",
                0,
                2,
                PUBLIC_LATCH_OBSERVATION_INDEX,
                1.0,
            ),
        ),
    )


def _public_latch_rule() -> ActionAuthorityRule:
    return ActionAuthorityRule(
        "arm_after_public_latch",
        0,
        ROUTER_ARM_STOP,
        PUBLIC_LATCH_OBSERVATION_INDEX,
        1.0,
    )


def _router_noise_groups() -> tuple[ActionNoiseGroup, ...]:
    return (
        ActionNoiseGroup("arm", 0, ROUTER_ARM_STOP, scale=0.0),
        ActionNoiseGroup("hand", ROUTER_ARM_STOP, ROUTER_ACTION_DIM, scale=0.5),
    )


def _public_latch_agent(*, routed: bool) -> FlashSACTorchBridge:
    observation_space = gym.spaces.Box(
        -1.0, 1.0, shape=(OBSERVATION_DIM,), dtype="float32"
    )
    action_space = gym.spaces.Box(
        -1.0, 1.0, shape=(ROUTER_ACTION_DIM,), dtype="float32"
    )
    router = (
        PublicLatchFrozenActorRouter(
            name=PUBLIC_LATCH_FROZEN_ACTOR_ROUTER_NAME,
            observation_index=PUBLIC_LATCH_OBSERVATION_INDEX,
            trainable_start=ROUTER_ARM_STOP,
            trainable_stop=ROUTER_ACTION_DIM,
        )
        if routed
        else None
    )
    return FlashSACTorchBridge(
        observation_space,
        action_space,
        {},
        _config(actor_update_period=1),
        noise_groups=_router_noise_groups(),
        action_authority_rules=(_public_latch_rule(),),
        public_latch_frozen_actor_router=router,
    )


def _load_same_actor_into_router(
    routed: FlashSACTorchBridge,
    source_checkpoint: Path,
) -> None:
    routed.load_actor(str(source_checkpoint))
    routed.load_frozen_lift_actor(str(source_checkpoint))


def _authority_agent(
    *,
    active_index: int | None = ALIGN_ACTIVE_OBSERVATION_INDEX,
    unit_normalize_actor_mean_head: bool = False,
) -> FlashSACTorchBridge:
    observation_space = gym.spaces.Box(
        -100.0,
        100.0,
        shape=(AUTHORITY_OBSERVATION_DIM,),
        dtype="float32",
    )
    action_space = gym.spaces.Box(
        -1.0,
        1.0,
        shape=(AUTHORITY_ACTION_DIM,),
        dtype="float32",
    )
    return FlashSACTorchBridge(
        observation_space,
        action_space,
        {},
        _config(actor_update_period=1),
        noise_groups=(
            ActionNoiseGroup("token", 0, 9, scale=0.5),
            ActionNoiseGroup("residual", 9, 14, scale=0.35),
        ),
        actor_action_active_observation_index=active_index,
        unit_normalize_actor_mean_head=unit_normalize_actor_mean_head,
    )


def _authority_transition(
    observation: torch.Tensor,
    next_observation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    num_envs = observation.shape[0]
    return {
        "observation": observation.clone(),
        "action": torch.zeros(num_envs, AUTHORITY_ACTION_DIM),
        "reward": torch.linspace(-1.0, 1.0, num_envs),
        "terminated": torch.zeros(num_envs, dtype=torch.bool),
        "truncated": torch.zeros(num_envs, dtype=torch.bool),
        "next_observation": next_observation.clone(),
    }


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


def _actor_batch_norm_buffers(
    agent: FlashSACTorchBridge,
) -> dict[str, torch.Tensor]:
    buffers = {
        name.removeprefix("_orig_mod."): value.detach().clone()
        for name, value in agent._actor.network.named_buffers()  # noqa: SLF001
        if name.endswith(("running_mean", "running_var"))
    }
    assert buffers, "FlashSAC actor regression requires stateful BatchNorm buffers"
    return buffers


def _assert_actor_batch_norm_buffers_equal(
    agent: FlashSACTorchBridge,
    expected: dict[str, torch.Tensor],
) -> None:
    actual = _actor_batch_norm_buffers(agent)
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, rtol=0.0, atol=0.0)


def _actor_parameters(agent: FlashSACTorchBridge) -> dict[str, torch.Tensor]:
    return {
        name.removeprefix("_orig_mod."): value.detach().clone()
        for name, value in agent._actor.network.named_parameters()  # noqa: SLF001
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


def test_phase_inactive_collection_actions_are_exact_zero() -> None:
    torch.manual_seed(205)
    agent = _authority_agent()
    assert (
        agent.actor_action_active_observation_index
        == ALIGN_ACTIVE_OBSERVATION_INDEX
    )
    observation = torch.randn(NUM_ENVS, AUTHORITY_OBSERVATION_DIM)
    observation[:, ALIGN_ACTIVE_OBSERVATION_INDEX] = torch.tensor(
        [1.0, 0.0, 0.25, 0.0]
    )
    active = observation[:, ALIGN_ACTIVE_OBSERVATION_INDEX] == 0.0
    with torch.no_grad():
        mean, _ = agent._actor.apply(  # noqa: SLF001
            "get_mean_and_std",
            observations=observation,
            training=False,
        )
        expected_active = torch.tanh(mean)[active]

    deterministic = agent.sample_actions(
        1, {"next_observation": observation}, training=False
    )
    torch.testing.assert_close(
        deterministic[active], expected_active, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        deterministic[~active],
        torch.zeros_like(deterministic[~active]),
        rtol=0.0,
        atol=0.0,
    )
    stochastic = agent.sample_actions(
        2, {"next_observation": observation}, training=True
    )
    torch.testing.assert_close(
        stochastic[~active],
        torch.zeros_like(stochastic[~active]),
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(stochastic[active], deterministic[active])

    _expect_error(ValueError, _authority_agent, active_index=-1)
    _expect_error(
        ValueError,
        _authority_agent,
        active_index=AUTHORITY_OBSERVATION_DIM,
    )
    _expect_error(ValueError, _authority_agent, active_index=True)


def test_public_slice_authority_gates_collection_random_and_replay() -> None:
    torch.manual_seed(2050)
    agent = _slice_authority_agent()
    observation = torch.randn(NUM_ENVS, OBSERVATION_DIM)
    observation[:, PUBLIC_LATCH_OBSERVATION_INDEX] = torch.tensor(
        [0.0, 1.0, 0.0, 1.0]
    )
    with torch.no_grad():
        mean, _ = agent._actor.apply(  # noqa: SLF001
            "get_mean_and_std",
            observations=observation,
            training=False,
        )
        proposal = torch.tanh(mean)

    action = agent.sample_actions(
        1, {"next_observation": observation}, training=False
    )
    expected = proposal.clone()
    expected[observation[:, PUBLIC_LATCH_OBSERVATION_INDEX] == 0.0, :2] = 0.0
    torch.testing.assert_close(action, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(action[:, 2:], proposal[:, 2:], rtol=0.0, atol=0.0)

    random_proposal = torch.linspace(-1.0, 1.0, NUM_ENVS * ACTION_DIM).reshape(
        NUM_ENVS, ACTION_DIM
    )
    canonical = agent.apply_action_authority(random_proposal, observation)
    expected_random = random_proposal.clone()
    expected_random[observation[:, PUBLIC_LATCH_OBSERVATION_INDEX] == 0.0, :2] = 0.0
    torch.testing.assert_close(canonical, expected_random, rtol=0.0, atol=0.0)

    transition = _transition(observation, canonical)
    transition["next_observation"][:, PUBLIC_LATCH_OBSERVATION_INDEX] = 1.0
    assert agent.process_transition(transition) == NUM_ENVS
    invalid_transition = _transition(observation, canonical)
    invalid_transition["action"][0, 0] = 0.25
    _expect_error(ValueError, agent.process_transition, invalid_transition)

    malformed = observation.clone()
    malformed[0, PUBLIC_LATCH_OBSERVATION_INDEX] = 0.25
    _expect_error(
        ValueError,
        agent.apply_action_authority,
        random_proposal,
        malformed,
    )


def test_public_slice_authority_masks_actor_target_entropy_and_checkpoint() -> None:
    torch.manual_seed(2052)
    agent = _slice_authority_agent()
    target_entropy = agent._cfg.temp_target_entropy  # noqa: SLF001
    observation = torch.randn(NUM_ENVS, OBSERVATION_DIM)
    next_observation = torch.randn(NUM_ENVS, OBSERVATION_DIM)
    observation[:, PUBLIC_LATCH_OBSERVATION_INDEX] = torch.tensor(
        [0.0, 1.0, 0.0, 1.0]
    )
    next_observation[:, PUBLIC_LATCH_OBSERVATION_INDEX] = torch.tensor(
        [1.0, 0.0, 1.0, 0.0]
    )
    transition = _transition(
        observation,
        agent.apply_action_authority(torch.zeros(NUM_ENVS, ACTION_DIM), observation),
    )
    transition["next_observation"] = next_observation
    agent.process_transition(transition)
    replay_sample = agent._replay_buffer.sample  # noqa: SLF001
    sample_indices = torch.arange(NUM_ENVS)
    agent._replay_buffer.sample = lambda: replay_sample(  # type: ignore[method-assign]  # noqa: SLF001
        sample_idxs=sample_indices
    )

    actor_records: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    actor_critic_records: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    target_records: list[tuple[torch.Tensor, torch.Tensor]] = []

    def record_actor(_module, _args, kwargs, output):
        actor_records.append(
            (
                kwargs["observations"].detach().clone(),
                output[0].detach().clone(),
                output[1]["log_prob_per_dim"].detach().clone(),
            )
        )

    def record_actor_critic(_module, _args, kwargs, output):
        if not bool(kwargs["training"]):
            actor_critic_records.append(
                (
                    kwargs["observations"].detach().clone(),
                    kwargs["actions"].detach().clone(),
                    output[0].detach().clone(),
                )
            )

    def record_target(_module, _args, kwargs, _output):
        target_records.append(
            (
                kwargs["observations"].detach().clone(),
                kwargs["actions"].detach().clone(),
            )
        )

    handles = (
        agent._actor.network.register_forward_hook(record_actor, with_kwargs=True),  # noqa: SLF001
        agent._critic.network.register_forward_hook(  # noqa: SLF001
            record_actor_critic, with_kwargs=True
        ),
        agent._target_critic.network.register_forward_hook(  # noqa: SLF001
            record_target, with_kwargs=True
        ),
    )
    temperature_before = agent._temperature().detach().clone()  # noqa: SLF001
    try:
        metrics = agent.update(actor_enabled=True)
    finally:
        for handle in handles:
            handle.remove()

    assert len(actor_records) == 2
    assert len(actor_critic_records) == 1
    assert len(target_records) == 1
    actor_obs_all, raw_actions_all, log_prob_per_dim_all = actor_records[0]
    current_observation = actor_obs_all[:NUM_ENVS]
    current_mask = torch.ones(NUM_ENVS, ACTION_DIM, dtype=torch.bool)
    current_mask[
        current_observation[:, PUBLIC_LATCH_OBSERVATION_INDEX] == 0.0, :2
    ] = False
    expected_current_actions = torch.where(
        current_mask,
        raw_actions_all[:NUM_ENVS],
        torch.zeros_like(raw_actions_all[:NUM_ENVS]),
    )
    expected_log_prob = torch.where(
        current_mask,
        log_prob_per_dim_all[:NUM_ENVS],
        torch.zeros_like(log_prob_per_dim_all[:NUM_ENVS]),
    ).sum(dim=-1)
    critic_observation, critic_actions, critic_qs = actor_critic_records[0]
    torch.testing.assert_close(critic_observation, current_observation)
    torch.testing.assert_close(critic_actions, expected_current_actions)
    q = torch.minimum(critic_qs[0], critic_qs[1])
    expected_entropy = -expected_log_prob.mean()
    expected_actor_loss = (expected_log_prob * temperature_before - q).mean()
    torch.testing.assert_close(torch.tensor(metrics["actor/entropy"]), expected_entropy)
    torch.testing.assert_close(torch.tensor(metrics["actor/loss"]), expected_actor_loss)
    mean_active_dimensions = current_mask.sum(dim=-1).float().mean()
    expected_temperature_loss = temperature_before * (
        expected_entropy - target_entropy * mean_active_dimensions / ACTION_DIM
    )
    torch.testing.assert_close(
        torch.tensor(metrics["temperature/loss"]), expected_temperature_loss.squeeze()
    )

    _, raw_next_actions, _ = actor_records[1]
    target_observation, target_actions_all = target_records[0]
    target_next_observation = target_observation[NUM_ENVS:]
    target_next_actions = target_actions_all[NUM_ENVS:]
    next_mask = torch.ones(NUM_ENVS, ACTION_DIM, dtype=torch.bool)
    next_mask[
        target_next_observation[:, PUBLIC_LATCH_OBSERVATION_INDEX] == 0.0, :2
    ] = False
    torch.testing.assert_close(
        target_next_actions,
        torch.where(next_mask, raw_next_actions, torch.zeros_like(raw_next_actions)),
    )

    with tempfile.TemporaryDirectory(prefix="flashsac_slice_authority_") as directory:
        checkpoint = Path(directory) / "checkpoint"
        agent.save(str(checkpoint))
        state = torch.load(
            checkpoint / BRIDGE_STATE_FILENAME,
            map_location="cpu",
            weights_only=True,
        )
        assert state["action_authority_rules"] == [
            {
                "name": "arm_after_public_latch",
                "start": 0,
                "stop": 2,
                "observation_index": PUBLIC_LATCH_OBSERVATION_INDEX,
                "active_value": 1.0,
            }
        ]
        restored = _slice_authority_agent()
        restored.load(str(checkpoint))
        mismatched = _agent()
        _expect_error(ValueError, mismatched.load, str(checkpoint))


def test_public_latch_router_routes_mixed_collection_and_trainable_targets() -> None:
    torch.manual_seed(2054)
    observation = torch.randn(NUM_ENVS, OBSERVATION_DIM)
    observation[:, PUBLIC_LATCH_OBSERVATION_INDEX] = torch.tensor(
        [0.0, 1.0, 0.0, 1.0]
    )
    close_rows = observation[:, PUBLIC_LATCH_OBSERVATION_INDEX] == 0.0
    proposal = torch.linspace(
        -0.9, 0.9, NUM_ENVS * ROUTER_ACTION_DIM
    ).reshape(NUM_ENVS, ROUTER_ACTION_DIM)

    # Demonstration targets can be canonicalized before the frozen actor is
    # loaded: only the close-policy hand suffix remains trainable.
    unloaded = _public_latch_agent(routed=True)
    expected_trainable = torch.zeros_like(proposal)
    expected_trainable[close_rows, ROUTER_ARM_STOP:] = proposal[
        close_rows, ROUTER_ARM_STOP:
    ]
    torch.testing.assert_close(
        unloaded.apply_trainable_action_authority(proposal, observation),
        expected_trainable,
        rtol=0.0,
        atol=0.0,
    )
    _expect_error(
        RuntimeError,
        unloaded.apply_action_authority,
        proposal,
        observation,
    )
    malformed = observation.clone()
    malformed[0, PUBLIC_LATCH_OBSERVATION_INDEX] = 0.25
    _expect_error(
        ValueError,
        unloaded.apply_trainable_action_authority,
        proposal,
        malformed,
    )
    _expect_error(
        ValueError,
        unloaded.apply_action_authority,
        proposal,
        malformed,
    )

    with tempfile.TemporaryDirectory(prefix="flashsac_router_collection_") as directory:
        source_checkpoint = Path(directory) / "source"
        source = _public_latch_agent(routed=False)
        source.save(str(source_checkpoint))
        agent = _public_latch_agent(routed=True)
        _load_same_actor_into_router(agent, source_checkpoint)

        with torch.no_grad():
            mean, _ = agent._actor.apply(  # noqa: SLF001
                "get_mean_and_std",
                observations=observation,
                training=False,
            )
            close_proposal = torch.tanh(mean)
            frozen_action = agent.frozen_lift_actions(observation)

        expected = frozen_action.clone()
        expected[close_rows, :ROUTER_ARM_STOP] = 0.0
        expected[close_rows, ROUTER_ARM_STOP:] = close_proposal[
            close_rows, ROUTER_ARM_STOP:
        ]
        first = agent.sample_actions(
            1, {"next_observation": observation}, training=False
        )
        second = agent.sample_actions(
            2, {"next_observation": observation}, training=False
        )
        torch.testing.assert_close(first, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(second, expected, rtol=0.0, atol=0.0)

        explicit_expected = proposal.clone()
        explicit_expected[close_rows, :ROUTER_ARM_STOP] = 0.0
        explicit_expected[~close_rows] = frozen_action[~close_rows]
        torch.testing.assert_close(
            agent.apply_action_authority(proposal, observation),
            explicit_expected,
            rtol=0.0,
            atol=0.0,
        )

        stochastic = agent.sample_actions(
            3, {"next_observation": observation}, training=True
        )
        torch.testing.assert_close(
            stochastic[~close_rows],
            frozen_action[~close_rows],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            stochastic[close_rows, :ROUTER_ARM_STOP],
            torch.zeros_like(stochastic[close_rows, :ROUTER_ARM_STOP]),
            rtol=0.0,
            atol=0.0,
        )


def test_public_latch_router_checkpoint_sidecar_is_strict_and_bit_exact() -> None:
    torch.manual_seed(2055)
    observation = torch.randn(9, OBSERVATION_DIM)
    observation[:, PUBLIC_LATCH_OBSERVATION_INDEX] = torch.tensor(
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    )

    with tempfile.TemporaryDirectory(prefix="flashsac_router_sidecar_") as directory:
        root = Path(directory)
        source_checkpoint = root / "source"
        source = _public_latch_agent(routed=False)
        source.save(str(source_checkpoint))
        agent = _public_latch_agent(routed=True)
        _load_same_actor_into_router(agent, source_checkpoint)

        checkpoint = root / "routed"
        agent.save(str(checkpoint))
        assert (checkpoint / FROZEN_LIFT_ACTOR_FILENAME).is_file()
        expected_sha256 = agent.frozen_lift_actor_sha256
        expected_action = agent.sample_actions(
            1, {"next_observation": observation}, training=False
        )
        restored = _public_latch_agent(routed=True)
        restored.load(str(checkpoint))
        actual_action = restored.sample_actions(
            1, {"next_observation": observation}, training=False
        )
        torch.testing.assert_close(actual_action, expected_action, rtol=0.0, atol=0.0)
        assert restored.frozen_lift_actor_sha256 == expected_sha256
        assert (
            restored.frozen_lift_actor_source_sha256
            == agent.frozen_lift_actor_source_sha256
        )

        missing_checkpoint = root / "missing_sidecar"
        agent.save(str(missing_checkpoint))
        (missing_checkpoint / FROZEN_LIFT_ACTOR_FILENAME).unlink()
        _expect_error(
            FileNotFoundError,
            _public_latch_agent(routed=True).load,
            str(missing_checkpoint),
        )

        tampered_checkpoint = root / "tampered_sidecar"
        agent.save(str(tampered_checkpoint))
        sidecar_path = tampered_checkpoint / FROZEN_LIFT_ACTOR_FILENAME
        sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=True)
        state = sidecar["network_state_dict"]
        tensor_name = next(
            name
            for name, value in state.items()
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point
        )
        tampered = state[tensor_name].clone()
        tampered.reshape(-1)[0] += 0.125
        state[tensor_name] = tampered
        torch.save(sidecar, sidecar_path)
        _expect_error(
            ValueError,
            _public_latch_agent(routed=True).load,
            str(tampered_checkpoint),
        )
        guarded = _public_latch_agent(routed=True)
        guarded.load_actor(str(checkpoint))
        guarded.load_frozen_lift_actor_sidecar(str(checkpoint))
        guarded_sha256 = guarded.frozen_lift_actor_sha256
        guarded_action = guarded.sample_actions(
            1, {"next_observation": observation}, training=False
        )
        _expect_error(
            ValueError,
            guarded.load_frozen_lift_actor_sidecar,
            str(tampered_checkpoint),
        )
        assert guarded.frozen_lift_actor_sha256 == guarded_sha256
        torch.testing.assert_close(
            guarded.sample_actions(
                1, {"next_observation": observation}, training=False
            ),
            guarded_action,
            rtol=0.0,
            atol=0.0,
        )

        orphan_checkpoint = root / "orphan_sidecar"
        nonrouter = _public_latch_agent(routed=False)
        nonrouter.save(str(orphan_checkpoint))
        shutil.copy2(
            checkpoint / FROZEN_LIFT_ACTOR_FILENAME,
            orphan_checkpoint / FROZEN_LIFT_ACTOR_FILENAME,
        )
        _expect_error(
            ValueError,
            _public_latch_agent(routed=False).load,
            str(orphan_checkpoint),
        )


def test_public_latch_router_zero_update_matches_v5_gate_elementwise() -> None:
    torch.manual_seed(2056)
    observation = torch.randn(23, OBSERVATION_DIM)
    observation[:, PUBLIC_LATCH_OBSERVATION_INDEX] = (
        torch.arange(observation.shape[0]) % 3 == 0
    ).float()

    with tempfile.TemporaryDirectory(prefix="flashsac_router_v5_equivalence_") as directory:
        source_checkpoint = Path(directory) / "source"
        v5 = _public_latch_agent(routed=False)
        v5.save(str(source_checkpoint))
        routed = _public_latch_agent(routed=True)
        _load_same_actor_into_router(routed, source_checkpoint)

        expected = v5.sample_actions(
            1, {"next_observation": observation}, training=False
        )
        actual = routed.sample_actions(
            1, {"next_observation": observation}, training=False
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_public_latch_router_all_frozen_batch_skips_actor_and_routes_target() -> None:
    torch.manual_seed(2057)
    with tempfile.TemporaryDirectory(prefix="flashsac_router_target_") as directory:
        source_checkpoint = Path(directory) / "source"
        source = _public_latch_agent(routed=False)
        source.save(str(source_checkpoint))
        agent = _public_latch_agent(routed=True)
        _load_same_actor_into_router(agent, source_checkpoint)

        observation = torch.randn(NUM_ENVS, OBSERVATION_DIM)
        next_observation = torch.randn(NUM_ENVS, OBSERVATION_DIM)
        observation[:, PUBLIC_LATCH_OBSERVATION_INDEX] = 1.0
        next_observation[:, PUBLIC_LATCH_OBSERVATION_INDEX] = 1.0
        action = agent.sample_actions(
            1, {"next_observation": observation}, training=False
        )
        transition = _transition(observation, action)
        transition["next_observation"] = next_observation
        assert agent.process_transition(transition) == NUM_ENVS

        target_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

        def record_target(_module, _args, kwargs, _output):
            target_calls.append(
                (
                    kwargs["observations"].detach().clone(),
                    kwargs["actions"].detach().clone(),
                )
            )

        handle = agent._target_critic.network.register_forward_hook(  # noqa: SLF001
            record_target,
            with_kwargs=True,
        )
        actor_before = _canonical_network_state(agent._actor)  # noqa: SLF001
        temperature_before = _canonical_network_state(agent._temperature)  # noqa: SLF001
        try:
            metrics = agent.update(actor_enabled=True)
        finally:
            handle.remove()

        assert metrics["actor/updated"] == 0.0
        assert metrics["critic/max_entropy_bonus"] == 0.0
        assert len(target_calls) == 1
        target_observation, target_action = target_calls[0]
        target_next_observation = target_observation[NUM_ENVS:]
        expected_target_action = agent.frozen_lift_actions(target_next_observation)
        torch.testing.assert_close(
            target_action[NUM_ENVS:],
            expected_target_action,
            rtol=0.0,
            atol=0.0,
        )
        _assert_nested_equal(
            _canonical_network_state(agent._actor),  # noqa: SLF001
            actor_before,
        )
        _assert_nested_equal(
            _canonical_network_state(agent._temperature),  # noqa: SLF001
            temperature_before,
        )


def test_demo_rehearsal_respects_public_slice_authority() -> None:
    torch.manual_seed(2053)
    agent = _slice_authority_agent()
    observation = torch.randn(NUM_ENVS, OBSERVATION_DIM)
    observation[:, PUBLIC_LATCH_OBSERVATION_INDEX] = torch.tensor(
        [0.0, 1.0, 0.0, 1.0]
    )
    active = torch.ones(NUM_ENVS, ACTION_DIM, dtype=torch.bool)
    active[observation[:, PUBLIC_LATCH_OBSERVATION_INDEX] == 0.0, :2] = False
    target_std = 0.2
    group_weights = {"arm": 2.0, "token": 0.5, "residual": 1.5}
    action_weights = torch.tensor([2.0, 2.0, 0.5, 0.5, 0.5, 1.5])
    with torch.no_grad():
        predicted_mean, predicted_std = agent._actor.apply(  # noqa: SLF001
            "get_mean_and_std",
            observations=observation,
            training=False,
        )
        target_action = torch.tanh(predicted_mean + 0.2)
        target_action = torch.where(
            active, target_action, torch.zeros_like(target_action)
        )
        target_mean = torch.atanh(target_action.clamp(-0.9999, 0.9999))
        action_elements = F.smooth_l1_loss(
            predicted_mean,
            target_mean,
            reduction="none",
            beta=1.0,
        )
        weighted_active = active.float() * action_weights.unsqueeze(0)
        expected_action_loss = (
            torch.where(active, action_elements, torch.zeros_like(action_elements))
            * action_weights.unsqueeze(0)
        ).sum() / weighted_active.sum()
        std_elements = (
            predicted_std.clamp_min(1.0e-8).log() - torch.tensor(target_std).log()
        ).square()
        expected_std_loss = torch.where(
            active, std_elements, torch.zeros_like(std_elements)
        ).sum() / active.sum()

    metrics = agent.demo_bc_rehearsal(
        {"observation": observation, "action": target_action},
        weight=1.0,
        group_weights=group_weights,
        target_std=target_std,
        std_weight=0.25,
    )
    torch.testing.assert_close(
        torch.tensor(metrics["demo_bc/action_loss"]), expected_action_loss
    )
    torch.testing.assert_close(
        torch.tensor(metrics["demo_bc/std_loss"]), expected_std_loss
    )
    assert metrics["demo_bc/active_action_fraction"] == float(active.float().mean())
    assert metrics["demo_bc/arm_active_elements"] == float(active[:, :2].sum())
    assert metrics["demo_bc/token_active_fraction"] == 1.0
    assert metrics["demo_bc/residual_active_fraction"] == 1.0

    noncanonical = target_action.clone()
    noncanonical[0, 0] = 0.25
    parameters_before = _actor_parameters(agent)
    _expect_error(
        ValueError,
        agent.demo_bc_rehearsal,
        {"observation": observation, "action": noncanonical},
    )
    _assert_nested_equal(_actor_parameters(agent), parameters_before)

    observation_space, action_space = _spaces()
    inactive_agent = FlashSACTorchBridge(
        observation_space,
        action_space,
        {},
        _config(actor_update_period=1),
        noise_groups=_groups(),
        action_authority_rules=(
            ActionAuthorityRule(
                "all_after_public_latch",
                0,
                ACTION_DIM,
                PUBLIC_LATCH_OBSERVATION_INDEX,
                1.0,
            ),
        ),
    )
    inactive_observation = observation.clone()
    inactive_observation[:, PUBLIC_LATCH_OBSERVATION_INDEX] = 0.0
    inactive_before = _actor_parameters(inactive_agent)
    inactive_metrics = inactive_agent.demo_bc_rehearsal(
        {
            "observation": inactive_observation,
            "action": torch.zeros(NUM_ENVS, ACTION_DIM),
        }
    )
    assert inactive_metrics["demo_bc/updated"] == 0.0
    assert inactive_metrics["demo_bc/active_action_fraction"] == 0.0
    _assert_nested_equal(_actor_parameters(inactive_agent), inactive_before)


def test_initialize_zero_actor_mean_is_exact_and_fresh_only() -> None:
    for use_compile in (False, True):
        torch.manual_seed(2051 + int(use_compile))
        agent = _agent_with_action_dim(
            ACTION_DIM,
            use_compile=use_compile,
            unit_normalize_actor_mean_head=False,
        )
        observation = torch.randn(NUM_ENVS, OBSERVATION_DIM)
        before = _canonical_network_state(agent._actor)  # noqa: SLF001
        with torch.no_grad():
            mean_before, std_before = agent._actor.apply(  # noqa: SLF001
                "get_mean_and_std",
                observations=observation,
                training=False,
            )
        assert torch.count_nonzero(mean_before) > 0
        assert agent._actor.optimizer is not None  # noqa: SLF001
        assert not agent._actor.optimizer.state  # noqa: SLF001

        agent.initialize_zero_actor_mean()

        with torch.no_grad():
            mean_after, std_after = agent._actor.apply(  # noqa: SLF001
                "get_mean_and_std",
                observations=observation,
                training=False,
            )
        torch.testing.assert_close(
            torch.tanh(mean_after),
            torch.zeros_like(mean_after),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(std_after, std_before, rtol=0.0, atol=0.0)
        after = _canonical_network_state(agent._actor)  # noqa: SLF001
        mean_keys = {"predictor.mean_w.w.weight", "predictor.mean_bias"}
        assert after.keys() == before.keys()
        for name, value in after.items():
            if name in mean_keys:
                torch.testing.assert_close(
                    value, torch.zeros_like(value), rtol=0.0, atol=0.0
                )
            else:
                # Covers the std head, shared trunk, and every BatchNorm
                # parameter/running buffer.
                torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)

        agent.process_transition(
            _transition(
                observation,
                torch.zeros(NUM_ENVS, ACTION_DIM),
            )
        )
        agent.update(actor_enabled=True)
        optimizer = agent._actor.optimizer  # noqa: SLF001
        assert optimizer.state
        with torch.no_grad():
            mean_after_update, _ = agent._actor.apply(  # noqa: SLF001
                "get_mean_and_std",
                observations=observation,
                training=False,
            )
        # Regression: UnitLinear's ordinary unit projection used to amplify
        # the first tiny gradient into near-saturated residual actions.
        assert float(torch.tanh(mean_after_update).abs().max()) < 0.1
        trained_state = _canonical_network_state(agent._actor)  # noqa: SLF001
        assert bool(
            (
                torch.linalg.vector_norm(
                    trained_state["predictor.mean_w.w.weight"],
                    dim=-1,
                )
                <= 1.0 + 1.0e-6
            ).all()
        )
        _expect_error(RuntimeError, agent.initialize_zero_actor_mean)
        _assert_nested_equal(
            _canonical_network_state(agent._actor),  # noqa: SLF001
            trained_state,
        )


def test_phase_inactive_masks_replay_actor_and_target_updates() -> None:
    torch.manual_seed(206)
    agent = _authority_agent()
    observation = torch.randn(NUM_ENVS, AUTHORITY_OBSERVATION_DIM)
    next_observation = torch.randn(NUM_ENVS, AUTHORITY_OBSERVATION_DIM)
    observation[:, ALIGN_ACTIVE_OBSERVATION_INDEX] = torch.tensor(
        [1.0, 0.0, 1.0, 0.0]
    )
    next_observation[:, ALIGN_ACTIVE_OBSERVATION_INDEX] = torch.tensor(
        [0.0, 1.0, 1.0, 0.0]
    )
    agent.process_transition(_authority_transition(observation, next_observation))
    replay_sample = agent._replay_buffer.sample  # noqa: SLF001
    sample_indices = torch.arange(NUM_ENVS)
    agent._replay_buffer.sample = lambda: replay_sample(  # type: ignore[method-assign]  # noqa: SLF001
        sample_idxs=sample_indices
    )

    actor_records: list[tuple[torch.Tensor, torch.Tensor]] = []
    actor_critic_records: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = []
    target_critic_records: list[tuple[torch.Tensor, torch.Tensor]] = []

    def record_actor(_module, _args, kwargs, output):
        actor_records.append(
            (
                kwargs["observations"].detach().clone(),
                output[1]["log_prob"].detach().clone(),
            )
        )

    def record_actor_critic(_module, _args, kwargs, output):
        if not bool(kwargs["training"]):
            actor_critic_records.append(
                (
                    kwargs["observations"].detach().clone(),
                    kwargs["actions"].detach().clone(),
                    output[0].detach().clone(),
                )
            )

    def record_target_critic(_module, _args, kwargs, _output):
        target_critic_records.append(
            (
                kwargs["observations"].detach().clone(),
                kwargs["actions"].detach().clone(),
            )
        )

    handles = (
        agent._actor.network.register_forward_hook(  # noqa: SLF001
            record_actor, with_kwargs=True
        ),
        agent._critic.network.register_forward_hook(  # noqa: SLF001
            record_actor_critic, with_kwargs=True
        ),
        agent._target_critic.network.register_forward_hook(  # noqa: SLF001
            record_target_critic, with_kwargs=True
        ),
    )
    temperature_before = agent._temperature().detach().clone()  # noqa: SLF001
    try:
        metrics = agent.update(actor_enabled=True)
    finally:
        for handle in handles:
            handle.remove()

    assert len(actor_records) == 2
    assert len(actor_critic_records) == 1
    assert len(target_critic_records) == 1
    actor_observation, actor_log_prob = actor_records[0]
    current_observation = actor_observation[:NUM_ENVS]
    current_log_prob = actor_log_prob[:NUM_ENVS]
    current_active = (
        current_observation[:, ALIGN_ACTIVE_OBSERVATION_INDEX] == 0.0
    )
    critic_observation, critic_actions, critic_qs = actor_critic_records[0]
    torch.testing.assert_close(critic_observation, current_observation)
    torch.testing.assert_close(
        critic_actions[~current_active],
        torch.zeros_like(critic_actions[~current_active]),
        rtol=0.0,
        atol=0.0,
    )
    q = torch.minimum(critic_qs[0], critic_qs[1])
    expected_actor_loss = (
        current_log_prob[current_active] * temperature_before
        - q[current_active]
    ).mean()
    expected_entropy = -current_log_prob[current_active].mean()
    torch.testing.assert_close(
        torch.tensor(metrics["actor/loss"]), expected_actor_loss
    )
    torch.testing.assert_close(
        torch.tensor(metrics["actor/entropy"]), expected_entropy
    )
    assert metrics["actor/updated"] == 1.0

    target_observation, target_actions = target_critic_records[0]
    target_next_observation = target_observation[NUM_ENVS:]
    target_next_actions = target_actions[NUM_ENVS:]
    target_next_active = (
        target_next_observation[:, ALIGN_ACTIVE_OBSERVATION_INDEX] == 0.0
    )
    torch.testing.assert_close(
        target_next_actions[~target_next_active],
        torch.zeros_like(target_next_actions[~target_next_active]),
        rtol=0.0,
        atol=0.0,
    )

    # A fully inactive sampled batch must remain finite and contribute exactly
    # zero actor objective, entropy, and target entropy bonus.
    inactive_agent = _authority_agent()
    inactive_observation = torch.randn(NUM_ENVS, AUTHORITY_OBSERVATION_DIM)
    inactive_next_observation = torch.randn(NUM_ENVS, AUTHORITY_OBSERVATION_DIM)
    inactive_observation[:, ALIGN_ACTIVE_OBSERVATION_INDEX] = 1.0
    inactive_next_observation[:, ALIGN_ACTIVE_OBSERVATION_INDEX] = 1.0
    inactive_agent.process_transition(
        _authority_transition(inactive_observation, inactive_next_observation)
    )
    inactive_sample = inactive_agent._replay_buffer.sample  # noqa: SLF001
    inactive_agent._replay_buffer.sample = lambda: inactive_sample(  # type: ignore[method-assign]  # noqa: SLF001
        sample_idxs=sample_indices
    )
    _seed_optimizer_state(inactive_agent)
    actor_before = _canonical_network_state(inactive_agent._actor)  # noqa: SLF001
    actor_optimizer = inactive_agent._actor.optimizer  # noqa: SLF001
    actor_scheduler = inactive_agent._actor.scheduler  # noqa: SLF001
    assert actor_optimizer is not None and actor_optimizer.state
    assert actor_scheduler is not None
    actor_optimizer_before = copy.deepcopy(actor_optimizer.state_dict())
    actor_scheduler_before = copy.deepcopy(actor_scheduler.state_dict())
    temperature_before = {
        name: value.detach().clone()
        for name, value in inactive_agent._temperature.network.state_dict().items()  # noqa: SLF001
    }
    temperature_optimizer = inactive_agent._temperature.optimizer  # noqa: SLF001
    temperature_scheduler = inactive_agent._temperature.scheduler  # noqa: SLF001
    assert temperature_optimizer is not None and temperature_optimizer.state
    assert temperature_scheduler is not None
    temperature_optimizer_before = copy.deepcopy(temperature_optimizer.state_dict())
    temperature_scheduler_before = copy.deepcopy(temperature_scheduler.state_dict())
    inactive_metrics = inactive_agent.update(actor_enabled=True)
    assert all(torch.isfinite(torch.tensor(value)) for value in inactive_metrics.values())
    assert inactive_metrics["actor/loss"] == 0.0
    assert inactive_metrics["actor/entropy"] == 0.0
    assert inactive_metrics["actor/mean_action"] == 0.0
    assert inactive_metrics["actor/updated"] == 0.0
    assert inactive_metrics["critic/max_entropy_bonus"] == 0.0
    assert not any(name.startswith("temperature/") for name in inactive_metrics)
    _assert_nested_equal(
        _canonical_network_state(inactive_agent._actor),  # noqa: SLF001
        actor_before,
    )
    _assert_nested_equal(actor_optimizer.state_dict(), actor_optimizer_before)
    _assert_nested_equal(actor_scheduler.state_dict(), actor_scheduler_before)
    for name, expected in temperature_before.items():
        torch.testing.assert_close(
            inactive_agent._temperature.network.state_dict()[name],  # noqa: SLF001
            expected,
            rtol=0.0,
            atol=0.0,
        )
    _assert_nested_equal(
        temperature_optimizer.state_dict(), temperature_optimizer_before
    )
    _assert_nested_equal(
        temperature_scheduler.state_dict(), temperature_scheduler_before
    )


def test_locked_policy_forces_zero_critic_targets_until_replay_unlock() -> None:
    torch.manual_seed(207)
    agent = _authority_agent()
    observation = torch.randn(NUM_ENVS, AUTHORITY_OBSERVATION_DIM)
    next_observation = torch.randn(NUM_ENVS, AUTHORITY_OBSERVATION_DIM)
    # CLOSE rows ordinarily grant policy authority. The global exploration
    # bridge must still withhold it until a native strict success is replayed.
    observation[:, ALIGN_ACTIVE_OBSERVATION_INDEX] = 0.0
    next_observation[:, ALIGN_ACTIVE_OBSERVATION_INDEX] = 0.0
    agent.process_transition(_authority_transition(observation, next_observation))
    replay_sample = agent._replay_buffer.sample  # noqa: SLF001
    sample_indices = torch.arange(NUM_ENVS)
    agent._replay_buffer.sample = lambda: replay_sample(  # type: ignore[method-assign]  # noqa: SLF001
        sample_idxs=sample_indices
    )
    target_actions: list[torch.Tensor] = []

    def record_target(_module, _args, kwargs, _output):
        target_actions.append(kwargs["actions"].detach().clone())

    handle = agent._target_critic.network.register_forward_hook(  # noqa: SLF001
        record_target,
        with_kwargs=True,
    )
    actor_before = _canonical_network_state(agent._actor)  # noqa: SLF001
    try:
        metrics = agent.update(
            actor_enabled=False,
            policy_actions_enabled=False,
        )
    finally:
        handle.remove()
    assert len(target_actions) == 1
    torch.testing.assert_close(
        target_actions[0][NUM_ENVS:],
        torch.zeros_like(target_actions[0][NUM_ENVS:]),
        rtol=0.0,
        atol=0.0,
    )
    assert metrics["critic/max_entropy_bonus"] == 0.0
    _assert_nested_equal(
        _canonical_network_state(agent._actor),  # noqa: SLF001
        actor_before,
    )
    _expect_error(
        ValueError,
        _agent().update,
        actor_enabled=False,
        policy_actions_enabled=False,
    )


def test_action_authority_checkpoint_configuration_is_strict() -> None:
    source = _authority_agent()
    with tempfile.TemporaryDirectory(prefix="flashsac_action_authority_") as directory:
        checkpoint = Path(directory) / "checkpoint"
        source.save(str(checkpoint))
        bridge_state = torch.load(
            checkpoint / BRIDGE_STATE_FILENAME,
            map_location="cpu",
            weights_only=True,
        )
        assert (
            bridge_state["actor_action_active_observation_index"]
            == ALIGN_ACTIVE_OBSERVATION_INDEX
        )
        assert bridge_state["unit_normalize_actor_mean_head"] is False
        assert (
            "actor_action_active_observation_index"
            not in _agent()._bridge_checkpoint_state()  # noqa: SLF001
        )

        restored = _authority_agent()
        restored.load(str(checkpoint))
        assert (
            restored.actor_action_active_observation_index
            == ALIGN_ACTIVE_OBSERVATION_INDEX
        )

        mismatched = _authority_agent(active_index=None)
        _expect_error(ValueError, mismatched.load, str(checkpoint))
        mismatched_normalization = _authority_agent(
            unit_normalize_actor_mean_head=True
        )
        _expect_error(
            ValueError,
            mismatched_normalization.load,
            str(checkpoint),
        )


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
    assert agent.process_transition(transition) == NUM_ENVS
    assert agent.can_start_training()
    stored = agent._replay_buffer.get_observations()  # noqa: SLF001
    assert isinstance(stored, torch.Tensor)
    assert stored.device == agent.device
    torch.testing.assert_close(stored, observation)

    invalid = dict(transition)
    invalid["action"] = action.tolist()
    _expect_error(TypeError, agent.process_transition, invalid)

    delayed = _agent(n_step=3)
    assert delayed.process_transition(transition) == 0
    assert delayed.process_transition(transition) == 0
    assert delayed.process_transition(transition) == NUM_ENVS


def test_masked_replay_validates_only_selected_canonical_actions() -> None:
    agent = _slice_authority_agent()
    observation = torch.randn(NUM_ENVS, OBSERVATION_DIM)
    observation[:, PUBLIC_LATCH_OBSERVATION_INDEX] = torch.tensor(
        [0.0, 0.25, 1.0, 0.25]
    )
    action = torch.zeros(NUM_ENVS, ACTION_DIM)
    action[[1, 3], :2] = 0.75
    transition = _transition(observation, action)
    valid = torch.tensor([True, False, True, False])

    # Invalid SEARCH rows deliberately contain malformed authority features and
    # non-canonical option actions.  They must neither fail option validation
    # nor reach replay.
    assert (
        agent.process_transition_masked(
            transition,
            replay_valid_mask=valid,
        )
        == 2
    )
    replay = agent._replay_buffer  # noqa: SLF001
    torch.testing.assert_close(
        replay._observations[:2],  # noqa: SLF001
        observation[valid],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        replay._actions[:2],  # noqa: SLF001
        action[valid],
        rtol=0.0,
        atol=0.0,
    )

    _expect_error(
        TypeError,
        agent.process_transition_masked,
        transition,
        replay_valid_mask=[True] * NUM_ENVS,
    )
    _expect_error(
        TypeError,
        agent.process_transition_masked,
        transition,
        replay_valid_mask=torch.ones(NUM_ENVS),
    )
    _expect_error(
        ValueError,
        agent.process_transition_masked,
        transition,
        replay_valid_mask=torch.ones(NUM_ENVS, 1, dtype=torch.bool),
    )
    _expect_error(
        ValueError,
        agent.process_transition_masked,
        transition,
        replay_valid_mask=torch.ones(NUM_ENVS, dtype=torch.bool, device="meta"),
    )

    noncanonical = _slice_authority_agent()
    bad_transition = _transition(observation, action)
    bad_transition["action"][0, 0] = 0.5
    _expect_error(
        ValueError,
        noncanonical.process_transition_masked,
        bad_transition,
        replay_valid_mask=valid,
    )


def test_masked_replay_uses_oldest_mask_and_preserves_n_step_done() -> None:
    agent = _agent(n_step=3, gamma=0.5)

    def step_transition(step: int) -> dict[str, torch.Tensor]:
        observation = (
            torch.arange(NUM_ENVS * OBSERVATION_DIM, dtype=torch.float32).reshape(
                NUM_ENVS, OBSERVATION_DIM
            )
            + 100.0 * step
        )
        result = _transition(observation, torch.zeros(NUM_ENVS, ACTION_DIM))
        result["reward"] = torch.zeros(NUM_ENVS)
        result["next_observation"] = observation + 10.0
        return result

    transition0 = step_transition(0)
    transition0["reward"][2] = 5.0
    transition1 = step_transition(1)
    transition1["reward"][2] = 7.0
    transition1["terminated"][2] = True
    transition2 = step_transition(2)
    transition2["reward"][0] = 11.0
    transition3 = step_transition(3)
    transition3["reward"][0] = 13.0
    row2 = torch.tensor([False, False, True, False])
    row0 = torch.tensor([True, False, False, False])

    assert agent.process_transition_masked(
        transition0, replay_valid_mask=row2
    ) == 0
    assert agent.process_transition_masked(
        transition1, replay_valid_mask=row2
    ) == 0
    # The current mask selects row 0, but the mature window starts at t0, so
    # the oldest t0 mask materializes row 2.  Its return stops at t1 done.
    assert agent.process_transition_masked(
        transition2, replay_valid_mask=row0
    ) == 1
    replay = agent._replay_buffer  # noqa: SLF001
    torch.testing.assert_close(  # noqa: SLF001
        replay._observations[0], transition0["observation"][2], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(  # noqa: SLF001
        replay._next_observations[0],  # noqa: SLF001
        transition1["next_observation"][2],
        rtol=0.0,
        atol=0.0,
    )
    assert replay._rewards[0].item() == 8.5  # 5 + 0.5 * 7  # noqa: SLF001
    assert replay._terminateds[0].item() == 1.0  # noqa: SLF001

    assert agent.process_transition_masked(
        transition3, replay_valid_mask=row0
    ) == 1
    torch.testing.assert_close(  # noqa: SLF001
        replay._observations[1], transition1["observation"][2], rtol=0.0, atol=0.0
    )
    assert replay._rewards[1].item() == 7.0  # noqa: SLF001
    assert replay.total_materialized_rows == 2

    discontinuous = _agent(n_step=3)
    live = step_transition(0)
    discontinuous.process_transition_masked(live, replay_valid_mask=row2)
    _expect_error(
        ValueError,
        discontinuous.process_transition_masked,
        step_transition(1),
        replay_valid_mask=row0,
    )
    assert len(discontinuous._replay_buffer._n_step_transitions) == 1  # noqa: SLF001

    masked_then_plain = _agent(n_step=3)
    masked_then_plain.process_transition_masked(live, replay_valid_mask=row2)
    _expect_error(
        RuntimeError,
        masked_then_plain.process_transition,
        step_transition(1),
    )
    assert len(masked_then_plain._replay_buffer._n_step_transitions) == 1  # noqa: SLF001

    plain_then_masked = _agent(n_step=3)
    plain_then_masked.process_transition(live)
    _expect_error(
        RuntimeError,
        plain_then_masked.process_transition_masked,
        step_transition(1),
        replay_valid_mask=row2,
    )
    assert len(plain_then_masked._replay_buffer._n_step_transitions) == 1  # noqa: SLF001


def test_masked_replay_excludes_invalid_rows_from_reward_normalizer() -> None:
    agent = _agent(n_step=1, gamma=0.5, normalize_reward=True)
    transition = _transition(
        torch.randn(NUM_ENVS, OBSERVATION_DIM),
        torch.zeros(NUM_ENVS, ACTION_DIM),
    )
    transition["reward"] = torch.tensor([1.0, 1000000.0, 3.0, -1000000.0])
    valid = torch.tensor([True, False, True, False])
    transition["terminated"][valid] = True

    assert agent.process_transition_masked(
        transition, replay_valid_mask=valid
    ) == 2
    assert agent.reward_normalizer is not None
    torch.testing.assert_close(
        agent.reward_normalizer.G_r,
        torch.tensor([1.0, 0.0, 3.0, 0.0]),
        rtol=0.0,
        atol=0.0,
    )
    assert agent.reward_normalizer.G_r_max.item() == 3.0
    assert agent.reward_normalizer.G_rms.count.item() == 2.0
    replay = agent._replay_buffer  # noqa: SLF001
    torch.testing.assert_close(  # noqa: SLF001
        replay._rewards[:2], torch.tensor([1.0, 3.0]), rtol=0.0, atol=0.0
    )

    all_invalid = _transition(
        torch.randn(NUM_ENVS, OBSERVATION_DIM),
        torch.zeros(NUM_ENVS, ACTION_DIM),
    )
    all_invalid["reward"].fill_(1.0e9)
    assert agent.process_transition_masked(
        all_invalid,
        replay_valid_mask=torch.zeros(NUM_ENVS, dtype=torch.bool),
    ) == 0
    assert torch.count_nonzero(agent.reward_normalizer.G_r) == 0
    assert agent.reward_normalizer.G_r_max.item() == 3.0
    assert agent.reward_normalizer.G_rms.count.item() == 2.0
    assert agent.replay_size == 2


def test_fresh_rollout_discards_only_trajectory_local_state() -> None:
    agent = _agent(n_step=3, normalize_reward=True)
    observation = torch.randn(NUM_ENVS, OBSERVATION_DIM)
    action = torch.zeros(NUM_ENVS, ACTION_DIM)
    transition = _transition(observation, action)
    agent.process_transition(transition)
    agent.process_transition(transition)
    assert len(agent._replay_buffer._n_step_transitions) == 2  # noqa: SLF001
    assert agent.reward_normalizer is not None
    previous_max = agent.reward_normalizer.G_r_max.clone()
    previous_rms_count = agent.reward_normalizer.G_rms.count.clone()

    agent.start_fresh_rollout(batch_size=2)

    assert len(agent._replay_buffer._n_step_transitions) == 0  # noqa: SLF001
    assert agent.replay_size == 0
    assert agent.reward_normalizer.G_r.shape == (2,)
    assert torch.count_nonzero(agent.reward_normalizer.G_r) == 0
    torch.testing.assert_close(agent.reward_normalizer.G_r_max, previous_max)
    torch.testing.assert_close(agent.reward_normalizer.G_rms.count, previous_rms_count)
    assert agent._cached_noise.shape == (2, ACTION_DIM)  # noqa: SLF001


def test_critic_burnin_and_demo_only_actor_rehearsal() -> None:
    agent = _agent(actor_update_period=1)
    observation = torch.randn(NUM_ENVS, OBSERVATION_DIM)
    action = torch.zeros(NUM_ENVS, ACTION_DIM)
    agent.process_transition(_transition(observation, action))
    actor_before = {
        key: value.detach().clone()
        for key, value in agent._actor.network.state_dict().items()  # noqa: SLF001
    }
    burnin_metrics = agent.update(actor_enabled=False)
    assert "actor/loss" not in burnin_metrics
    for key, expected in actor_before.items():
        torch.testing.assert_close(
            agent._actor.network.state_dict()[key], expected, rtol=0.0, atol=0.0  # noqa: SLF001
        )

    with torch.no_grad():
        mean, _ = agent._actor.apply(  # noqa: SLF001
            "get_mean_and_std", observations=observation, training=False
        )
    demo_action = torch.tanh(mean + 0.25)
    rehearsal_metrics = agent.demo_bc_rehearsal(
        {"observation": observation, "action": demo_action},
        weight=1.0,
        group_weights={"arm": 0.2, "token": 1.0, "residual": 1.0},
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in rehearsal_metrics.values())
    assert {
        "demo_bc/arm_action_rmse",
        "demo_bc/token_action_rmse",
        "demo_bc/residual_action_rmse",
    }.issubset(rehearsal_metrics)
    _expect_error(
        ValueError,
        agent.demo_bc_rehearsal,
        {"observation": observation, "action": demo_action},
        group_weights={"unknown": 1.0},
    )
    assert any(
        not torch.equal(agent._actor.network.state_dict()[key], expected)  # noqa: SLF001
        for key, expected in actor_before.items()
    )


def test_sac_actor_update_uses_frozen_deployment_batch_norm() -> None:
    """SAC must optimize the deployed actor without adapting actor BN buffers."""

    torch.manual_seed(203)
    agent = _agent(actor_update_period=1)
    # This is deliberately far from the actor's initial running statistics. A
    # training-mode actor forward would visibly overwrite every BN buffer.
    observation = 40.0 + 3.0 * torch.randn(NUM_ENVS, OBSERVATION_DIM)
    action = torch.zeros(NUM_ENVS, ACTION_DIM)
    agent.process_transition(_transition(observation, action))
    buffers_before = _actor_batch_norm_buffers(agent)
    parameters_before = _actor_parameters(agent)

    forward_training_flags: list[bool] = []

    def record_actor_mode(_module, _args, kwargs):
        forward_training_flags.append(bool(kwargs["training"]))

    handle = agent._actor.network.register_forward_pre_hook(  # noqa: SLF001
        record_actor_mode,
        with_kwargs=True,
    )
    try:
        metrics = agent.update(actor_enabled=True)
    finally:
        handle.remove()

    assert "actor/loss" in metrics
    # Both the optimized actor forward and the target-action forward must use
    # the same running-statistics path as deployment.
    assert forward_training_flags and not any(forward_training_flags)
    _assert_actor_batch_norm_buffers_equal(agent, buffers_before)
    parameters_after = _actor_parameters(agent)
    assert any(
        not torch.equal(parameters_after[name], value)
        for name, value in parameters_before.items()
    ), "actor optimizer did not update deployment-path parameters"


def test_demo_rehearsal_optimizes_deployment_path_without_bn_drift() -> None:
    """Demo correction must reduce inference loss while keeping BN fixed."""

    torch.manual_seed(204)
    agent = _agent(actor_update_period=1)
    observation = 35.0 + 4.0 * torch.randn(32, OBSERVATION_DIM)
    with torch.no_grad():
        initial_mean, _ = agent._actor.apply(  # noqa: SLF001
            "get_mean_and_std",
            observations=observation,
            training=False,
        )
        # Stay away from tanh saturation so atanh(action) recovers this target
        # exactly inside demo_bc_rehearsal.
        target_mean = initial_mean.clamp(-1.5, 1.5) + 0.25
        target_action = torch.tanh(target_mean)
    buffers_before = _actor_batch_norm_buffers(agent)
    parameters_before = _actor_parameters(agent)

    def deployment_loss() -> torch.Tensor:
        mean, _ = agent._actor.apply(  # noqa: SLF001
            "get_mean_and_std",
            observations=observation,
            training=False,
        )
        return F.smooth_l1_loss(mean, target_mean, beta=1.0)

    with torch.no_grad():
        loss_before = deployment_loss()
    for _ in range(8):
        metrics = agent.demo_bc_rehearsal(
            {"observation": observation, "action": target_action},
            weight=1.0,
            target_std=0.15,
            std_weight=0.0,
        )
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    with torch.no_grad():
        loss_after = deployment_loss()

    _assert_actor_batch_norm_buffers_equal(agent, buffers_before)
    assert loss_after < loss_before, (
        "demo rehearsal failed to improve the deployed inference function: "
        f"before={float(loss_before)}, after={float(loss_after)}"
    )
    parameters_after = _actor_parameters(agent)
    assert any(
        not torch.equal(parameters_after[name], value)
        for name, value in parameters_before.items()
    )


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


def test_compiled_cuda_demo_rehearsal_preserves_diagnostic_output() -> None:
    """A later CUDA-Graph actor call must not overwrite rehearsal metrics."""

    if not torch.cuda.is_available():
        return
    torch.manual_seed(112)
    observation_space, action_space = _spaces()
    agent = FlashSACTorchBridge(
        observation_space,
        action_space,
        {},
        _config(
            device_type="cuda:0",
            buffer_device_type="cuda:0",
            use_compile=True,
            compile_mode="reduce-overhead",
            use_amp=True,
            actor_update_period=1,
        ),
        noise_groups=_groups(),
    )
    observation = torch.randn(32, OBSERVATION_DIM, device="cuda:0")
    with torch.no_grad():
        mean, _ = agent._actor.apply(  # noqa: SLF001
            "get_mean_and_std",
            observations=observation,
            training=False,
        )
        action = torch.tanh(mean.detach().clone() + 0.1)
    metrics = agent.demo_bc_rehearsal(
        {"observation": observation, "action": action},
        weight=1.0,
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["demo_bc/arm_action_rmse"] >= 0.0
    assert metrics["demo_bc/grad_overflow"] in (0.0, 1.0)

    scaler_state = agent._grad_scaler.state_dict()  # noqa: SLF001
    scaler_state["scale"] = 1.0e30
    agent._grad_scaler.load_state_dict(scaler_state)  # noqa: SLF001
    overflow_metrics = agent.demo_bc_rehearsal(
        {"observation": observation, "action": action},
        weight=2.0,
    )
    assert all(
        torch.isfinite(torch.tensor(value)) for value in overflow_metrics.values()
    )
    assert overflow_metrics["demo_bc/grad_overflow"] == 1.0


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


def test_actor_learning_rate_scale_is_absolute_and_checkpointed() -> None:
    agent = _agent()
    actor_optimizer = agent._actor.optimizer  # noqa: SLF001
    actor_scheduler = agent._actor.scheduler  # noqa: SLF001
    critic_optimizer = agent._critic.optimizer  # noqa: SLF001
    assert actor_optimizer is not None and actor_scheduler is not None
    assert critic_optimizer is not None
    original_critic_lr = float(critic_optimizer.param_groups[0]["lr"])

    agent.set_actor_learning_rate_scale(1.0e-3)
    assert agent.actor_learning_rate_scale == 1.0e-3
    assert math.isclose(float(actor_optimizer.param_groups[0]["lr"]), 3.0e-7)
    assert len(actor_scheduler.base_lrs) == 1
    assert math.isclose(float(actor_scheduler.base_lrs[0]), 3.0e-7)
    assert float(critic_optimizer.param_groups[0]["lr"]) == original_critic_lr

    # The setter is absolute: changing 1e-3 -> 1e-2 must not multiply the
    # already-scaled LR by another 1e-2.
    agent.set_actor_learning_rate_scale(1.0e-2)
    assert agent.actor_learning_rate_scale == 1.0e-2
    assert math.isclose(float(actor_optimizer.param_groups[0]["lr"]), 3.0e-6)
    assert len(actor_scheduler.base_lrs) == 1
    assert math.isclose(float(actor_scheduler.base_lrs[0]), 3.0e-6)
    _expect_error(ValueError, agent.set_actor_learning_rate_scale, 0.0)
    _expect_error(ValueError, agent.set_actor_learning_rate_scale, float("nan"))
    _expect_error(TypeError, agent.set_actor_learning_rate_scale, True)

    with tempfile.TemporaryDirectory(prefix="flashsac_actor_lr_scale_") as directory:
        checkpoint = Path(directory) / "checkpoint"
        agent.save(str(checkpoint))
        bridge_state = torch.load(
            checkpoint / BRIDGE_STATE_FILENAME,
            map_location="cpu",
            weights_only=True,
        )
        assert bridge_state["actor_learning_rate_scale"] == 1.0e-2

        restored = _agent()
        restored.load(str(checkpoint))
        assert restored.actor_learning_rate_scale == 1.0e-2
        assert restored._actor.optimizer is not None  # noqa: SLF001
        assert restored._actor.scheduler is not None  # noqa: SLF001
        assert math.isclose(  # noqa: SLF001
            float(restored._actor.optimizer.param_groups[0]["lr"]), 3.0e-6
        )
        assert len(restored._actor.scheduler.base_lrs) == 1  # noqa: SLF001
        assert math.isclose(  # noqa: SLF001
            float(restored._actor.scheduler.base_lrs[0]), 3.0e-6
        )


def _canonical_network_state(bundle) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    return {
        key.removeprefix(prefix): value.detach().clone()
        for key, value in bundle.network.state_dict().items()
    }


def _seed_optimizer_state(agent: FlashSACTorchBridge) -> None:
    """Populate Adam/scheduler state without compiling or running a forward pass."""

    for bundle in (agent._actor, agent._critic, agent._temperature):  # noqa: SLF001
        assert bundle.optimizer is not None
        for parameter in bundle.optimizer.param_groups[0]["params"]:
            parameter.grad = torch.full_like(parameter, 0.01)
        bundle.optimizer.step()
        bundle.optimizer.zero_grad(set_to_none=True)
        assert bundle.scheduler is not None
        bundle.scheduler.step()
        bundle.update_step = 3


def _assert_nested_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_checkpoint_loads_across_compile_boundary() -> None:
    bundle_names = ("_actor", "_critic", "_target_critic", "_temperature")
    for source_compiled, target_compiled in ((False, True), (True, False)):
        torch.manual_seed(120 + int(source_compiled))
        source = _agent(use_compile=source_compiled)
        _seed_optimizer_state(source)
        source._update_step = 13  # noqa: SLF001

        source_actor_keys = source._actor.network.state_dict().keys()  # noqa: SLF001
        assert any(key.startswith("_orig_mod.") for key in source_actor_keys) is source_compiled

        with tempfile.TemporaryDirectory(prefix="flashsac_compile_boundary_") as directory:
            checkpoint = Path(directory) / "checkpoint"
            source.save(str(checkpoint))

            target = _agent(use_compile=target_compiled)
            target_actor_keys = target._actor.network.state_dict().keys()  # noqa: SLF001
            assert any(key.startswith("_orig_mod.") for key in target_actor_keys) is target_compiled
            target.load(str(checkpoint))

            for name in bundle_names:
                source_bundle = getattr(source, name)
                target_bundle = getattr(target, name)
                _assert_nested_equal(
                    _canonical_network_state(target_bundle),
                    _canonical_network_state(source_bundle),
                )
                if source_bundle.optimizer is not None:
                    _assert_nested_equal(
                        target_bundle.optimizer.state_dict(),
                        source_bundle.optimizer.state_dict(),
                    )
                    _assert_nested_equal(
                        target_bundle.scheduler.state_dict(),
                        source_bundle.scheduler.state_dict(),
                    )
                    assert target_bundle.update_step == source_bundle.update_step == 3
            assert target._update_step == source._update_step == 13  # noqa: SLF001


def test_checkpoint_loads_across_amp_boundary() -> None:
    for source_amp, target_amp in ((False, True), (True, False)):
        source = _agent(use_amp=source_amp)
        source._update_step = 17  # noqa: SLF001
        with tempfile.TemporaryDirectory(prefix="flashsac_amp_boundary_") as directory:
            checkpoint = Path(directory) / "checkpoint"
            source.save(str(checkpoint))
            target = _agent(use_amp=target_amp)
            expected_fresh_state = target._grad_scaler.state_dict()  # noqa: SLF001
            target.load(str(checkpoint))
            assert target._update_step == 17  # noqa: SLF001
            assert target._grad_scaler.is_enabled() is target_amp  # noqa: SLF001
            _assert_nested_equal(  # Changed AMP mode intentionally retains target initialization.
                target._grad_scaler.state_dict(),  # noqa: SLF001
                expected_fresh_state,
            )


def test_actor_only_checkpoint_is_portable_and_leaves_fresh_state() -> None:
    for source_compiled, target_compiled in ((False, True), (True, False)):
        torch.manual_seed(210 + int(source_compiled))
        source = _agent(use_compile=source_compiled, normalize_reward=True)
        _seed_optimizer_state(source)
        source._update_step = 29  # noqa: SLF001

        with tempfile.TemporaryDirectory(prefix="flashsac_actor_only_") as directory:
            checkpoint = Path(directory) / "checkpoint"
            source.save(str(checkpoint))

            target = _agent(use_compile=target_compiled, normalize_reward=True)
            observations = torch.randn(NUM_ENVS, OBSERVATION_DIM)
            _ = target.sample_actions(
                1, {"next_observation": observations}, training=True
            )
            target.process_transition(
                _transition(observations, torch.zeros(NUM_ENVS, ACTION_DIM))
            )
            untouched_networks = {
                name: _canonical_network_state(getattr(target, name))
                for name in ("_critic", "_target_critic", "_temperature")
            }
            untouched_actor_optimizer = target._actor.optimizer.state_dict()  # noqa: SLF001
            untouched_actor_step = target._actor.update_step  # noqa: SLF001
            untouched_agent_step = target._update_step  # noqa: SLF001
            untouched_noise = target._cached_noise.clone()  # noqa: SLF001
            untouched_replay = target._replay_buffer.get_observations().clone()  # noqa: SLF001
            assert target.reward_normalizer is not None
            untouched_normalizer_max = target.reward_normalizer.G_r_max.clone()

            target.load_actor(str(checkpoint))

            _assert_nested_equal(
                _canonical_network_state(target._actor),  # noqa: SLF001
                _canonical_network_state(source._actor),  # noqa: SLF001
            )
            for name, expected in untouched_networks.items():
                _assert_nested_equal(
                    _canonical_network_state(getattr(target, name)), expected
                )
            _assert_nested_equal(
                target._actor.optimizer.state_dict(), untouched_actor_optimizer  # noqa: SLF001
            )
            assert target._actor.update_step == untouched_actor_step == 0  # noqa: SLF001
            assert target._update_step == untouched_agent_step == 0  # noqa: SLF001
            torch.testing.assert_close(
                target._cached_noise, untouched_noise, rtol=0.0, atol=0.0  # noqa: SLF001
            )
            torch.testing.assert_close(
                target._replay_buffer.get_observations(),  # noqa: SLF001
                untouched_replay,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                target.reward_normalizer.G_r_max,
                untouched_normalizer_max,
                rtol=0.0,
                atol=0.0,
            )

            _expect_error(
                FileNotFoundError,
                target.load_actor,
                str(Path(directory) / "missing"),
            )


def test_actor_only_projection_slices_exact_outputs_and_keeps_fresh_state() -> None:
    source_action_dim = 21
    target_action_dim = 14
    hand_indices = tuple(range(7, 21))
    output_keys = {
        "predictor.mean_w.w.weight",
        "predictor.mean_bias",
        "predictor.std_w.w.weight",
        "predictor.std_bias",
    }
    for source_compiled, target_compiled in ((False, True), (True, False)):
        torch.manual_seed(310 + int(source_compiled))
        source = _agent_with_action_dim(source_action_dim, use_compile=source_compiled)
        observations = torch.randn(NUM_ENVS, OBSERVATION_DIM)

        with tempfile.TemporaryDirectory(prefix="flashsac_actor_projection_") as directory:
            checkpoint = Path(directory) / "checkpoint"
            source.save(str(checkpoint))
            source_action = source.sample_actions(
                1, {"next_observation": observations}, training=False
            )

            target = _agent_with_action_dim(target_action_dim, use_compile=target_compiled)
            untouched_networks = {
                name: _canonical_network_state(getattr(target, name))
                for name in ("_critic", "_target_critic", "_temperature")
            }
            untouched_optimizer = target._actor.optimizer.state_dict()  # noqa: SLF001
            target.load_actor(
                str(checkpoint),
                source_action_indices=hand_indices,
                expected_source_action_dim=source_action_dim,
            )

            source_state = _canonical_network_state(source._actor)  # noqa: SLF001
            target_state = _canonical_network_state(target._actor)  # noqa: SLF001
            assert target_state.keys() == source_state.keys()
            index = torch.tensor(hand_indices, dtype=torch.long)
            for key, target_value in target_state.items():
                expected = (
                    source_state[key].index_select(0, index)
                    if key in output_keys
                    else source_state[key]
                )
                torch.testing.assert_close(target_value, expected, rtol=0.0, atol=0.0)

            target_action = target.sample_actions(
                1, {"next_observation": observations}, training=False
            )
            torch.testing.assert_close(
                target_action,
                source_action[:, 7:],
                rtol=1.0e-6,
                atol=1.0e-7,
            )
            for name, expected in untouched_networks.items():
                _assert_nested_equal(
                    _canonical_network_state(getattr(target, name)), expected
                )
            _assert_nested_equal(
                target._actor.optimizer.state_dict(), untouched_optimizer  # noqa: SLF001
            )
            assert target._update_step == 0  # noqa: SLF001

            _expect_error(
                ValueError,
                target.load_actor,
                str(checkpoint),
                source_action_indices=hand_indices,
            )
            _expect_error(
                ValueError,
                target.load_actor,
                str(checkpoint),
                source_action_indices=(7,) * target_action_dim,
                expected_source_action_dim=source_action_dim,
            )
            _expect_error(
                RuntimeError,
                target.load_actor,
                str(checkpoint),
                source_action_indices=tuple(range(6, 20)),
                expected_source_action_dim=20,
            )


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
    test_phase_inactive_collection_actions_are_exact_zero()
    print("[PASS] phase-inactive collection actions are exact zero")
    test_public_slice_authority_gates_collection_random_and_replay()
    print("[PASS] public slice authority gates collection, random warmup, and replay")
    test_public_slice_authority_masks_actor_target_entropy_and_checkpoint()
    print("[PASS] public slice authority masks SAC updates and checkpoints")
    test_public_latch_router_routes_mixed_collection_and_trainable_targets()
    print("[PASS] public-latch router collection and trainable-target authority")
    test_public_latch_router_checkpoint_sidecar_is_strict_and_bit_exact()
    print("[PASS] public-latch router sidecar integrity and exact roundtrip")
    test_public_latch_router_zero_update_matches_v5_gate_elementwise()
    print("[PASS] public-latch router zero-update V5 equivalence")
    test_public_latch_router_all_frozen_batch_skips_actor_and_routes_target()
    print("[PASS] public-latch router frozen critic targets and actor skip")
    test_demo_rehearsal_respects_public_slice_authority()
    print("[PASS] demo rehearsal respects public slice authority")
    test_initialize_zero_actor_mean_is_exact_and_fresh_only()
    print("[PASS] fresh residual actor starts with exact-zero deterministic mean")
    test_phase_inactive_masks_replay_actor_and_target_updates()
    test_locked_policy_forces_zero_critic_targets_until_replay_unlock()
    print("[PASS] phase-inactive actor/target replay updates")
    test_action_authority_checkpoint_configuration_is_strict()
    print("[PASS] action-authority checkpoint configuration")
    test_transition_contract_and_replay_are_torch_native()
    print("[PASS] torch transition/replay contract")
    test_masked_replay_validates_only_selected_canonical_actions()
    test_masked_replay_uses_oldest_mask_and_preserves_n_step_done()
    test_masked_replay_excludes_invalid_rows_from_reward_normalizer()
    print("[PASS] masked per-environment replay contract")
    test_fresh_rollout_discards_only_trajectory_local_state()
    print("[PASS] fresh rollout boundary resets only trajectory-local state")
    test_critic_burnin_and_demo_only_actor_rehearsal()
    print("[PASS] critic burn-in and demo-only actor rehearsal")
    test_sac_actor_update_uses_frozen_deployment_batch_norm()
    print("[PASS] SAC actor update uses frozen deployment BatchNorm")
    test_demo_rehearsal_optimizes_deployment_path_without_bn_drift()
    print("[PASS] demo rehearsal optimizes deployment path without BatchNorm drift")
    test_partial_reset_refreshes_only_completed_envs()
    print("[PASS] per-environment exploration reset")
    test_cuda_interaction_has_no_host_round_trip()
    print("[PASS] CUDA interaction stays on device" if torch.cuda.is_available() else "[SKIP] CUDA unavailable")
    test_compiled_cuda_demo_rehearsal_preserves_diagnostic_output()
    print(
        "[PASS] compiled CUDA demo rehearsal output lifetime"
        if torch.cuda.is_available()
        else "[SKIP] CUDA unavailable"
    )
    test_checkpoint_exactly_restores_noise_and_rng()
    print("[PASS] exact checkpoint continuation")
    test_actor_learning_rate_scale_is_absolute_and_checkpointed()
    print("[PASS] absolute actor-only LR scale and checkpoint continuation")
    test_checkpoint_loads_across_compile_boundary()
    print("[PASS] compiled/uncompiled checkpoint portability (both directions)")
    test_checkpoint_loads_across_amp_boundary()
    print("[PASS] enabled/disabled AMP checkpoint portability (both directions)")
    test_actor_only_checkpoint_is_portable_and_leaves_fresh_state()
    print("[PASS] actor-only portability and fresh critic/replay/normalizer state")
    test_actor_only_projection_slices_exact_outputs_and_keeps_fresh_state()
    print("[PASS] actor-only 21D-to-14D output projection")
    test_group_partition_is_validated()
    print("[PASS] noise-group validation")
    print("All FlashSAC Torch bridge tests passed.")


if __name__ == "__main__":
    main()
