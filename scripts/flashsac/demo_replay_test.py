#!/usr/bin/env python3
"""Simulation-free regression tests for permanent demonstration replay."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from typing import Any

import gymnasium as gym
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Installs the narrow flash_rl.types fallback when this Torch-only environment
# does not have JAX.  No agent or simulator is constructed by the tests.
from agent_bridge import FlashSACTorchBridge, build_agent_config  # noqa: E402
from demo_replay import (  # noqa: E402
    DEMO_MASK_KEY,
    FixedFractionDemoReplay,
    PermanentDemoReservoir,
    TRANSITION_KEYS,
    attach_demo_replay,
    load_and_precompute_n_step,
    precompute_n_step_by_episode,
)
from flash_rl.buffers.torch_buffer import TorchUniformBuffer  # noqa: E402


OBSERVATION_DIM = 5
ACTION_DIM = 3
GAMMA = 0.9


def _batch(
    start: float,
    rows: int,
    *,
    device: torch.device | str,
    terminal_last: bool = False,
) -> dict[str, torch.Tensor]:
    resolved = torch.device(device)
    scalar = torch.arange(rows, dtype=torch.float32, device=resolved).add(start)
    observation = scalar[:, None] + torch.arange(
        OBSERVATION_DIM, dtype=torch.float32, device=resolved
    )[None, :].mul(0.01)
    action = scalar[:, None].mul(0.001) + torch.arange(
        ACTION_DIM, dtype=torch.float32, device=resolved
    )[None, :].mul(0.02)
    terminated = torch.zeros(rows, dtype=torch.bool, device=resolved)
    if terminal_last:
        terminated[-1] = True
    return {
        "observation": observation,
        "action": action,
        "reward": scalar.clone(),
        "terminated": terminated,
        "truncated": torch.zeros(rows, dtype=torch.bool, device=resolved),
        "next_observation": observation.add(0.5),
    }


def _online_buffer(
    *,
    device: torch.device | str,
    n_step: int = 1,
    capacity: int = 32,
    min_length: int = 4,
    sample_batch_size: int = 8,
) -> TorchUniformBuffer:
    observation_space = gym.spaces.Box(
        -float("inf"), float("inf"), shape=(OBSERVATION_DIM,), dtype="float32"
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype="float32")
    return TorchUniformBuffer(
        observation_space=observation_space,
        action_space=action_space,
        n_step=n_step,
        gamma=GAMMA,
        max_length=capacity,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
        device_type=str(device),
    )


def _demos(
    *,
    device: torch.device | str,
    n_step: int = 1,
    rows: int = 6,
    stratum: torch.Tensor | None = None,
) -> PermanentDemoReservoir:
    demos = PermanentDemoReservoir(
        capacity=rows,
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        n_step=n_step,
        gamma=GAMMA,
        device=device,
    )
    demos.add_precomputed(
        _batch(1_000.0, rows, device=device),
        n_step=n_step,
        gamma=GAMMA,
        stratum=stratum,
    )
    demos.seal()
    return demos


def _expect_error(error_type: type[BaseException], function: Any, *args: Any, **kwargs: Any) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _assert_batches_equal(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> None:
    assert set(actual) == set(expected) == {*TRANSITION_KEYS, DEMO_MASK_KEY}
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0)


def test_permanent_storage_and_exact_fraction() -> None:
    online = _online_buffer(device="cpu", capacity=16)
    online.add(_batch(0.0, 8, device="cpu"))
    demos = _demos(device="cpu", rows=4)
    original_demos = {key: value.clone() for key, value in demos.state_dict()["storage"].items()}
    replay = FixedFractionDemoReplay(
        online,
        demos,
        batch_size=8,
        demo_fraction=0.25,
        device="cpu",
        seed=10,
    )

    mixed = replay.sample()
    mask = mixed[DEMO_MASK_KEY]
    assert mask.dtype == torch.bool
    assert int(mask.sum()) == 2
    assert torch.all(mixed["reward"][mask] >= 1_000.0)
    assert torch.all(mixed["reward"][~mask] < 1_000.0)
    assert replay.demo_rows_per_batch == 2
    assert replay.online_rows_per_batch == 6

    # Force multiple online ring-buffer overwrites.  Demo allocation is wholly
    # separate and remains byte-exact.
    for step in range(10):
        replay.add(_batch(100.0 + 4 * step, 4, device="cpu"))
    for key, expected in original_demos.items():
        torch.testing.assert_close(demos.state_dict()["storage"][key], expected, rtol=0.0, atol=0.0)
    _expect_error(
        RuntimeError,
        demos.add_precomputed,
        _batch(2_000.0, 1, device="cpu"),
        n_step=1,
        gamma=GAMMA,
    )

    replay.reset()
    assert len(replay) == 0
    assert replay.demo_size == 4
    assert demos.sealed


def _sampled_strata(batch: dict[str, torch.Tensor], source_labels: torch.Tensor) -> torch.Tensor:
    source_rows = batch["reward"].sub(1_000.0).round().to(dtype=torch.int64)
    return source_labels.to(batch["reward"].device)[source_rows]


def test_stratified_sampling_balances_imbalanced_phases() -> None:
    # The source is intentionally 60% approach.  Uniform row sampling would
    # retain that skew; default labeled sampling gives each present phase an
    # equal integer quota in every demo sub-batch.
    labels = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 2, 2], dtype=torch.uint8)
    demos = _demos(device="cpu", rows=10, stratum=labels)
    generator = torch.Generator(device="cpu").manual_seed(20)
    balanced = demos.sample(6, generator=generator)
    sampled = _sampled_strata(balanced, labels)
    assert [int((sampled == phase).sum()) for phase in range(3)] == [2, 2, 2]

    near_balanced = demos.sample(5, generator=generator)
    counts = [int((_sampled_strata(near_balanced, labels) == phase).sum()) for phase in range(3)]
    assert sum(counts) == 5
    assert max(counts) - min(counts) == 1

    weighted = demos.sample(
        8,
        generator=generator,
        stratum_weights={0: 0.0, 1: 1.0, 2: 3.0},
    )
    sampled = _sampled_strata(weighted, labels)
    assert [int((sampled == phase).sum()) for phase in range(3)] == [0, 2, 6]
    _expect_error(
        ValueError,
        demos.sample,
        8,
        generator=generator,
        stratum_weights={99: 1.0},
    )

    unlabeled = _demos(device="cpu", rows=4)
    uniform = unlabeled.sample(3, generator=generator)
    assert uniform["reward"].shape == (3,)
    _expect_error(
        ValueError,
        unlabeled.sample,
        3,
        generator=generator,
        stratum_weights={0: 1.0},
    )


def test_precomputed_n_step_contract_is_strict() -> None:
    demos = PermanentDemoReservoir(
        capacity=3,
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        n_step=3,
        gamma=GAMMA,
        device="cpu",
    )
    batch = _batch(1_000.0, 3, device="cpu")
    _expect_error(ValueError, demos.add_precomputed, batch, n_step=1, gamma=GAMMA)
    _expect_error(ValueError, demos.add_precomputed, batch, n_step=3, gamma=0.99)

    invalid = dict(batch)
    invalid["truncated"] = invalid["terminated"].clone()
    invalid["terminated"] = torch.ones(3, dtype=torch.bool)
    invalid["truncated"] = torch.ones(3, dtype=torch.bool)
    _expect_error(ValueError, demos.add_precomputed, invalid, n_step=3, gamma=GAMMA)

    demos.add_precomputed(batch, n_step=3, gamma=GAMMA)
    demos.seal()
    online = _online_buffer(device="cpu", n_step=1)
    _expect_error(
        ValueError,
        FixedFractionDemoReplay,
        online,
        demos,
        batch_size=8,
        demo_fraction=0.25,
        device="cpu",
        seed=0,
    )
    _expect_error(
        ValueError,
        FixedFractionDemoReplay,
        _online_buffer(device="cpu", n_step=3),
        demos,
        batch_size=7,
        demo_fraction=0.25,
        device="cpu",
        seed=0,
    )


def _one_step_two_episodes(device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    batch = _batch(0.0, 5, device=device)
    batch["reward"] = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0], device=device)
    batch["terminated"] = torch.tensor([False, False, True, False, False], device=device)
    batch["truncated"] = torch.tensor([False, False, False, False, True], device=device)
    batch["next_observation"] = torch.arange(
        5 * OBSERVATION_DIM, dtype=torch.float32, device=device
    ).reshape(5, OBSERVATION_DIM).add(100.0)
    return batch


def test_episode_bounded_n_step_matches_upstream_and_retains_start_phase() -> None:
    one_step = _one_step_two_episodes()
    phases = torch.tensor([0, 0, 1, 1, 2], dtype=torch.uint8)
    precomputed, retained = precompute_n_step_by_episode(
        one_step,
        torch.tensor([0, 3, 5], dtype=torch.int64),
        n_step=3,
        gamma=GAMMA,
        phase=phases,
    )
    expected_rewards = torch.tensor(
        [
            1.0 + GAMMA * 2.0 + GAMMA**2 * 3.0,
            2.0 + GAMMA * 3.0,
            3.0,
            10.0 + GAMMA * 20.0,
            20.0,
        ]
    )
    torch.testing.assert_close(precomputed["reward"], expected_rewards)
    assert torch.equal(precomputed["terminated"], torch.tensor([True, True, True, False, False]))
    assert torch.equal(precomputed["truncated"], torch.tensor([False, False, False, True, True]))
    torch.testing.assert_close(
        precomputed["next_observation"],
        one_step["next_observation"][torch.tensor([2, 2, 2, 4, 4])],
    )
    assert retained is not None
    assert torch.equal(retained, phases.to(dtype=torch.int64))

    # Feed the same serial stream through the pinned upstream n-step deque.
    # Two rows from a following episode flush the final two starts, just as
    # continued vector-environment collection would after an auto-reset.
    upstream = _online_buffer(
        device="cpu",
        n_step=3,
        capacity=16,
        min_length=1,
        sample_batch_size=1,
    )
    padding = _batch(50.0, 2, device="cpu", terminal_last=True)
    serial = {
        key: torch.cat((one_step[key], padding[key]), dim=0)
        for key in TRANSITION_KEYS
    }
    for row in range(7):
        upstream.add({key: value[row : row + 1] for key, value in serial.items()})
    assert len(upstream) == 5
    upstream_rows = {
        "observation": upstream._observations[:5],  # noqa: SLF001
        "action": upstream._actions[:5],  # noqa: SLF001
        "reward": upstream._rewards[:5],  # noqa: SLF001
        "terminated": upstream._terminateds[:5],  # noqa: SLF001
        "truncated": upstream._truncateds[:5],  # noqa: SLF001
        "next_observation": upstream._next_observations[:5],  # noqa: SLF001
    }
    for key in TRANSITION_KEYS:
        torch.testing.assert_close(
            precomputed[key].to(dtype=upstream_rows[key].dtype),
            upstream_rows[key],
            rtol=0.0,
            atol=0.0,
        )

    malformed = dict(one_step)
    malformed["truncated"] = malformed["truncated"].clone()
    malformed["truncated"][-1] = False
    _expect_error(
        ValueError,
        precompute_n_step_by_episode,
        malformed,
        [0, 3, 5],
        n_step=3,
        gamma=GAMMA,
    )

    with tempfile.TemporaryDirectory(prefix="flashsac_one_step_demo_") as directory:
        path = Path(directory) / "one_step.pt"
        torch.save(
            {
                **one_step,
                "episode_offsets": torch.tensor([0, 3, 5]),
                "phase": phases,
            },
            path,
        )
        loaded, loaded_phase = load_and_precompute_n_step(
            path,
            device="cpu",
            n_step=3,
            gamma=GAMMA,
        )
    for key in TRANSITION_KEYS:
        torch.testing.assert_close(loaded[key], precomputed[key], rtol=0.0, atol=0.0)
    assert loaded_phase is not None
    assert torch.equal(loaded_phase, retained)


def test_checkpoint_exactly_restores_data_pending_n_step_and_rng() -> None:
    online = _online_buffer(device="cpu", n_step=3, capacity=32)
    for step in range(4):
        online.add(_batch(10.0 * step, 4, device="cpu", terminal_last=(step == 2)))
    assert len(online._n_step_transitions) == 3  # noqa: SLF001 - exact-state regression
    labels = torch.tensor([0, 0, 0, 1, 2, 2], dtype=torch.uint8)
    replay = FixedFractionDemoReplay(
        online,
        _demos(device="cpu", n_step=3, stratum=labels),
        batch_size=8,
        demo_fraction=0.25,
        device="cpu",
        seed=123,
        stratum_weights={0: 1.0, 1: 1.0, 2: 2.0},
    )
    _ = replay.sample()  # advance private device RNG before the checkpoint
    continuation = _batch(99.0, 4, device="cpu")

    with tempfile.TemporaryDirectory(prefix="flashsac_demo_replay_") as directory:
        checkpoint = Path(directory) / "replay_buffer.pt"
        replay.save(checkpoint)
        expected_first = replay.sample()
        replay.add(continuation)
        expected_second = replay.sample()
        expected_online = replay.state_dict()["online"]

        restored = FixedFractionDemoReplay(
            _online_buffer(device="cpu", n_step=3, capacity=32),
            _demos(device="cpu", n_step=3, stratum=labels),
            batch_size=8,
            demo_fraction=0.25,
            device="cpu",
            seed=999,
            stratum_weights={0: 1.0, 1: 1.0, 2: 2.0},
        )
        restored.load(checkpoint)
        actual_first = restored.sample()
        restored.add(continuation)
        actual_second = restored.sample()
        actual_online = restored.state_dict()["online"]

    _assert_batches_equal(actual_first, expected_first)
    _assert_batches_equal(actual_second, expected_second)
    assert actual_online["num_in_buffer"] == expected_online["num_in_buffer"]
    assert actual_online["current_idx"] == expected_online["current_idx"]
    assert len(actual_online["n_step_transitions"]) == len(expected_online["n_step_transitions"])
    assert torch.equal(replay.state_dict()["demos"]["stratum"], labels.to(dtype=torch.int64))
    for actual_transition, expected_transition in zip(
        actual_online["n_step_transitions"], expected_online["n_step_transitions"], strict=True
    ):
        for key in TRANSITION_KEYS:
            torch.testing.assert_close(
                actual_transition[key], expected_transition[key], rtol=0.0, atol=0.0
            )


def test_unlabeled_v1_checkpoint_is_backward_compatible() -> None:
    online = _online_buffer(device="cpu")
    online.add(_batch(0.0, 8, device="cpu"))
    replay = FixedFractionDemoReplay(
        online,
        _demos(device="cpu"),
        batch_size=8,
        demo_fraction=0.25,
        device="cpu",
        seed=31,
    )
    state = replay.state_dict()
    state["version"] = 1
    state.pop("stratum_weights")
    state["demos"]["version"] = 1
    state["demos"].pop("stratum")
    expected = replay.sample()

    restored_online = _online_buffer(device="cpu")
    restored = FixedFractionDemoReplay(
        restored_online,
        _demos(device="cpu"),
        batch_size=8,
        demo_fraction=0.25,
        device="cpu",
        seed=999,
    )
    restored.load_state_dict(state)
    _assert_batches_equal(restored.sample(), expected)


def test_demo_fingerprints_prevent_mislabeled_resume() -> None:
    online = _online_buffer(device="cpu")
    online.add(_batch(0.0, 8, device="cpu"))
    replay = FixedFractionDemoReplay(
        online,
        _demos(device="cpu"),
        batch_size=8,
        demo_fraction=0.25,
        device="cpu",
        seed=41,
        demo_fingerprints=("sha256-source-a",),
    )
    state = replay.state_dict()

    matching = FixedFractionDemoReplay(
        _online_buffer(device="cpu"),
        _demos(device="cpu"),
        batch_size=8,
        demo_fraction=0.25,
        device="cpu",
        seed=42,
        demo_fingerprints=("sha256-source-a",),
    )
    matching.load_state_dict(state)
    mismatching = FixedFractionDemoReplay(
        _online_buffer(device="cpu"),
        _demos(device="cpu"),
        batch_size=8,
        demo_fraction=0.25,
        device="cpu",
        seed=42,
        demo_fingerprints=("sha256-source-b",),
    )
    _expect_error(ValueError, mismatching.load_state_dict, state)

    # Starting a new simulator process retains completed online rows while
    # explicitly dropping the trajectory-local n-step deque.
    pending_online = _online_buffer(device="cpu", n_step=3)
    pending_online.add(_batch(10.0, 4, device="cpu"))
    pending_online.add(_batch(20.0, 4, device="cpu"))
    pending = FixedFractionDemoReplay(
        pending_online,
        _demos(device="cpu", n_step=3),
        batch_size=8,
        demo_fraction=0.25,
        device="cpu",
        seed=43,
    )
    assert len(pending_online._n_step_transitions) == 2  # noqa: SLF001
    pending.discard_pending_n_step()
    assert len(pending_online._n_step_transitions) == 0  # noqa: SLF001


def test_attach_helper_uses_unmodified_upstream_seam() -> None:
    online = _online_buffer(device="cpu")
    agent = SimpleNamespace(_replay_buffer=online, device=torch.device("cpu"))
    replay = attach_demo_replay(
        agent,
        _demos(device="cpu"),
        batch_size=8,
        demo_fraction=0.25,
        seed=7,
    )
    assert agent._replay_buffer is replay
    _expect_error(
        RuntimeError,
        attach_demo_replay,
        agent,
        _demos(device="cpu"),
        batch_size=8,
        demo_fraction=0.25,
        seed=7,
    )


def test_unmodified_agent_update_consumes_mixed_batch() -> None:
    observation_space = gym.spaces.Box(
        -float("inf"), float("inf"), shape=(OBSERVATION_DIM,), dtype="float32"
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype="float32")
    config = build_agent_config(
        device_type="cpu",
        buffer_device_type="cpu",
        buffer_max_length=32,
        buffer_min_length=4,
        sample_batch_size=8,
        gamma=GAMMA,
        n_step=1,
        actor_update_period=1,
    )
    agent = FlashSACTorchBridge(observation_space, action_space, {}, config)
    replay = attach_demo_replay(
        agent,
        _demos(device="cpu"),
        batch_size=8,
        demo_fraction=0.25,
        seed=71,
    )
    agent.process_transition(_batch(0.0, 4, device="cpu"))
    assert agent.can_start_training()
    metrics = agent.update()
    assert replay.demo_size == 6
    assert metrics
    assert all(math.isfinite(value) for value in metrics.values())


def test_cuda_sampling_stays_on_device() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:0")
    labels = torch.tensor([0, 0, 0, 1, 2, 2], dtype=torch.uint8)
    online = _online_buffer(device=device)
    online.add(_batch(0.0, 8, device=device))
    replay = FixedFractionDemoReplay(
        online,
        _demos(device=device, stratum=labels),
        batch_size=8,
        demo_fraction=0.25,
        device=device,
        seed=88,
    )
    batch = replay.sample()
    for value in batch.values():
        assert value.device == device
    with tempfile.TemporaryDirectory(prefix="flashsac_demo_replay_cuda_") as directory:
        checkpoint = Path(directory) / "replay_buffer.pt"
        replay.save(checkpoint)
        expected = replay.sample()
        restored = FixedFractionDemoReplay(
            _online_buffer(device=device),
            _demos(device=device, stratum=labels),
            batch_size=8,
            demo_fraction=0.25,
            device=device,
            seed=999,
        )
        restored.load(checkpoint)
        actual = restored.sample()
    _assert_batches_equal(actual, expected)


def main() -> None:
    test_permanent_storage_and_exact_fraction()
    print("[PASS] permanent demos and exact fixed-ratio batches")
    test_stratified_sampling_balances_imbalanced_phases()
    print("[PASS] balanced and explicitly weighted phase sampling")
    test_precomputed_n_step_contract_is_strict()
    print("[PASS] strict precomputed n-step contract")
    test_episode_bounded_n_step_matches_upstream_and_retains_start_phase()
    print("[PASS] episode-bounded n-step preprocessing matches upstream")
    test_checkpoint_exactly_restores_data_pending_n_step_and_rng()
    print("[PASS] exact replay/n-step/RNG checkpoint continuation")
    test_unlabeled_v1_checkpoint_is_backward_compatible()
    print("[PASS] unlabeled v1 replay checkpoint compatibility")
    test_demo_fingerprints_prevent_mislabeled_resume()
    print("[PASS] demo provenance and fresh-rollout replay boundary")
    test_attach_helper_uses_unmodified_upstream_seam()
    print("[PASS] unmodified upstream attachment seam")
    test_unmodified_agent_update_consumes_mixed_batch()
    print("[PASS] unmodified FlashSAC update consumes mixed batch")
    test_cuda_sampling_stays_on_device()
    print("[PASS] CUDA sample path stays on device" if torch.cuda.is_available() else "[SKIP] CUDA unavailable")
    print("All permanent demonstration replay tests passed.")


if __name__ == "__main__":
    main()
