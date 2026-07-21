#!/usr/bin/env python3
"""Pure-Torch logic tests for the PickTool FlashSAC adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from adapter import (
    ACTION_DIM,
    ARM_ACTION_DIM,
    COUPLED_POLICY_OBSERVATION_DIM,
    HAND_ACTION_DIM,
    POLICY_OBSERVATION_DIM,
    PickToolIsaacLabAdapter,
    build_replay_transition,
    classify_done,
    extract_policy_observation,
)


@dataclass
class _Cfg:
    observation_space: int = POLICY_OBSERVATION_DIM
    state_space: int = POLICY_OBSERVATION_DIM
    action_space: int = ACTION_DIM
    observation_noise_model: object | None = None
    close_option_mode: bool = False
    power_close_option_mode: bool = False
    coupled_power_align_close_option_mode: bool = False


class _FakeDirectEnv:
    """Minimal auto-reset environment with the ordering used by DirectRLEnv."""

    def __init__(
        self,
        *,
        call_reset_for_done: bool = True,
        observation_dim: int = POLICY_OBSERVATION_DIM,
    ) -> None:
        self.num_envs = 3
        self.device = "cpu"
        self.cfg = _Cfg(observation_space=observation_dim, state_space=observation_dim)
        self.max_episode_length = 100
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.call_reset_for_done = call_reset_for_done
        self.state = torch.zeros(self.num_envs, observation_dim)
        self.last_action: torch.Tensor | None = None
        self.last_extras: dict[str, object] = {}
        self.closed = False

    @property
    def unwrapped(self) -> "_FakeDirectEnv":
        return self

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # A deliberately different critic group proves that it is ignored.
        return {"policy": self.state, "critic": self.state + 1000.0}

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long)
        self.state[ids] = -100.0 - ids[:, None].float()
        self.episode_length_buf[ids] = 0

    def reset(self, **_: object) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        self.state.copy_(
            torch.arange(self.num_envs, dtype=torch.float32)[:, None].expand_as(self.state)
        )
        payload = {"reset": torch.tensor(1.0)}
        self.last_extras = {"payload": payload}
        return self._get_observations(), self.last_extras

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, object],
    ]:
        self.last_action = action.detach().clone()
        self.state.add_(action[:, :1])
        self.episode_length_buf.add_(1)
        terminated = torch.tensor([False, True, False])
        truncated = torch.tensor([False, False, True])
        reward = torch.tensor([1.0, 2.0, 3.0])
        payload = {"kept": torch.tensor(7.0)}
        self.last_extras = {
            "payload": payload,
            "log": {
                "success_frac": torch.tensor(1.0 / 3.0),
                "clearance_max": torch.tensor(0.2),
                "unrelated": torch.tensor(9.0),
            },
        }
        if self.call_reset_for_done:
            self._reset_idx((terminated | truncated).nonzero().flatten())
        return self._get_observations(), reward, terminated, truncated, self.last_extras

    def close(self) -> None:
        self.closed = True


def _assert_equal(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if not torch.equal(actual, expected):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def test_policy_only_and_spaces() -> None:
    raw = _FakeDirectEnv()
    env = PickToolIsaacLabAdapter(raw, require_cuda=False)
    obs, info = env.reset()
    assert obs.shape == (3, POLICY_OBSERVATION_DIM)
    assert env.observation_space.shape == (3, POLICY_OBSERVATION_DIM)
    assert env.action_space.shape == (3, ACTION_DIM)
    assert info["actor_observation_size"] == (POLICY_OBSERVATION_DIM,)
    assert info["asymmetric_obs"] is False
    assert torch.all(obs[:, 0] < 1000.0), "critic observations were concatenated"
    actions = env.sample_random_actions()
    assert actions.device.type == "cpu"
    assert actions.shape == (3, ACTION_DIM)
    assert bool((actions >= -1.0).all() & (actions <= 1.0).all())


def test_terminal_capture_and_replay_semantics() -> None:
    raw = _FakeDirectEnv()
    env = PickToolIsaacLabAdapter(raw, require_cuda=False)
    observation, _ = env.reset()
    observation = observation.clone()
    action = torch.zeros(3, ACTION_DIM)
    action[:, 0] = 0.5

    next_observation, reward, terminated, truncated, info = env.step(action)
    expected_terminal = observation + 0.5

    # Rollout rows 1 and 2 are the first state of a reset episode.
    _assert_equal(next_observation[1], torch.full((POLICY_OBSERVATION_DIM,), -101.0), "terminated reset obs")
    _assert_equal(next_observation[2], torch.full((POLICY_OBSERVATION_DIM,), -102.0), "timeout reset obs")

    # Replay rows 1 and 2 are the real post-action states captured pre-reset.
    replay_next = info["transition_next_observation"]
    _assert_equal(replay_next, expected_terminal, "transition next observation")
    _assert_equal(info["final_observation_mask"], torch.tensor([False, True, True]), "final mask")
    _assert_equal(info["episode_done"], torch.tensor([False, True, True]), "episode done")
    _assert_equal(info["bootstrap_mask"], torch.tensor([1.0, 0.0, 1.0]), "bootstrap mask")

    transition = build_replay_transition(
        observation,
        action,
        reward,
        terminated,
        truncated,
        info,
    )
    _assert_equal(transition["terminated"], torch.tensor([False, True, False]), "raw termination")
    _assert_equal(transition["truncated"], torch.tensor([False, False, True]), "raw timeout")
    _assert_equal(transition["next_observation"], expected_terminal, "buffer next obs")

    # Original extras survive as the same nested objects; strict logging is only
    # a reference subset of the already-computed task ground truth.
    assert info["payload"] is raw.last_extras["payload"]
    assert set(info["strict_metrics"]) == {"success_frac", "clearance_max"}
    assert info["strict_metrics"]["success_frac"] is raw.last_extras["log"]["success_frac"]


def test_hand_only_actions_expand_at_environment_boundary_and_stay_14d_in_replay() -> None:
    raw = _FakeDirectEnv()
    raw.cfg.close_option_mode = True
    env = PickToolIsaacLabAdapter(raw, require_cuda=False, hand_only_actions=True)
    observation, _ = env.reset()
    observation = observation.clone()
    action = torch.linspace(
        -0.8,
        0.8,
        raw.num_envs * HAND_ACTION_DIM,
        dtype=torch.float32,
    ).reshape(raw.num_envs, HAND_ACTION_DIM)

    next_observation, reward, terminated, truncated, info = env.step(action)
    del next_observation
    assert env.action_dim == HAND_ACTION_DIM
    assert env.environment_action_dim == ACTION_DIM
    assert env.action_space.shape == (raw.num_envs, HAND_ACTION_DIM)
    assert env.sample_random_actions().shape == (raw.num_envs, HAND_ACTION_DIM)
    assert raw.last_action is not None
    assert raw.last_action.shape == (raw.num_envs, ACTION_DIM)
    torch.testing.assert_close(
        raw.last_action[:, :ARM_ACTION_DIM],
        torch.zeros(raw.num_envs, ARM_ACTION_DIM),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        raw.last_action[:, ARM_ACTION_DIM:], action, rtol=0.0, atol=0.0
    )

    transition = build_replay_transition(
        observation,
        action,
        reward,
        terminated,
        truncated,
        info,
        action_dim=HAND_ACTION_DIM,
    )
    assert transition["action"].shape == (raw.num_envs, HAND_ACTION_DIM)
    torch.testing.assert_close(transition["action"], action, rtol=0.0, atol=0.0)

    try:
        build_replay_transition(
            observation,
            action,
            reward,
            terminated,
            truncated,
            info,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("14D replay action was accepted under the default 21D contract")


def test_coupled_actions_and_observations_use_native_identity_boundary() -> None:
    raw = _FakeDirectEnv(observation_dim=COUPLED_POLICY_OBSERVATION_DIM)
    raw.cfg.close_option_mode = True
    raw.cfg.power_close_option_mode = True
    raw.cfg.coupled_power_align_close_option_mode = True
    env = PickToolIsaacLabAdapter(raw, require_cuda=False)
    observation, _ = env.reset()
    assert observation.shape == (raw.num_envs, COUPLED_POLICY_OBSERVATION_DIM)
    assert env.observation_dim == COUPLED_POLICY_OBSERVATION_DIM
    assert env.action_dim == ACTION_DIM

    action = torch.linspace(
        -0.9, 0.9, raw.num_envs * ACTION_DIM, dtype=torch.float32
    ).reshape(raw.num_envs, ACTION_DIM)
    env.step(action)
    assert raw.last_action is not None
    torch.testing.assert_close(raw.last_action, action, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(raw.last_action[:, :ARM_ACTION_DIM]).item() > 0

    try:
        PickToolIsaacLabAdapter(raw, require_cuda=False, hand_only_actions=True)
    except ValueError as exc:
        assert "native 21-D identity" in str(exc)
    else:
        raise AssertionError("coupled environment accepted the hand-only projection")


def test_done_logic() -> None:
    terminated = torch.tensor([False, True, False, True])
    truncated = torch.tensor([False, False, True, True])
    signals = classify_done(terminated, truncated)
    _assert_equal(signals.episode_done, torch.tensor([False, True, True, True]), "done OR")
    _assert_equal(signals.bootstrap_mask, torch.tensor([1.0, 0.0, 1.0, 0.0]), "termination mask")

    try:
        classify_done(terminated.float(), truncated)
    except TypeError:
        pass
    else:
        raise AssertionError("float done tensor was accepted")


def test_contract_failures() -> None:
    try:
        extract_policy_observation({"policy": torch.zeros(2, 230)})
    except ValueError:
        pass
    else:
        raise AssertionError("230D policy+critic tensor was accepted")

    env = PickToolIsaacLabAdapter(_FakeDirectEnv(), require_cuda=False)
    env.reset()
    try:
        env.step(torch.zeros(3, ACTION_DIM - 1))
    except ValueError:
        pass
    else:
        raise AssertionError("20D action was accepted")

    try:
        PickToolIsaacLabAdapter(
            _FakeDirectEnv(), require_cuda=False, hand_only_actions=True
        )
    except ValueError as exc:
        assert "close_option_mode=True" in str(exc)
    else:
        raise AssertionError("hand-only actions were accepted outside close-option mode")

    close_raw = _FakeDirectEnv()
    close_raw.cfg.close_option_mode = True
    close_env = PickToolIsaacLabAdapter(
        close_raw, require_cuda=False, hand_only_actions=True
    )
    close_env.reset()
    try:
        close_env.step(torch.zeros(close_raw.num_envs, HAND_ACTION_DIM + 1))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong-width hand-only action was accepted")

    missing_capture = PickToolIsaacLabAdapter(
        _FakeDirectEnv(call_reset_for_done=False), require_cuda=False, strict=True
    )
    missing_capture.reset()
    try:
        missing_capture.step(torch.zeros(3, ACTION_DIM))
    except RuntimeError as exc:
        assert "capture" in str(exc)
    else:
        raise AssertionError("done rows without captured terminal observations were accepted")

    noisy = _FakeDirectEnv()
    noisy.cfg.observation_noise_model = object()
    try:
        PickToolIsaacLabAdapter(noisy, require_cuda=False, strict=True)
    except ValueError as exc:
        assert "observation_noise_model" in str(exc)
    else:
        raise AssertionError("observation-noise mismatch was accepted")


def test_episode_length_randomization_and_close() -> None:
    raw = _FakeDirectEnv()
    env = PickToolIsaacLabAdapter(raw, require_cuda=False)
    env.reset(randomize_episode_lengths=True)
    assert bool((raw.episode_length_buf >= 0).all())
    assert bool((raw.episode_length_buf < raw.max_episode_length).all())
    env.close()
    assert raw.closed


def main() -> None:
    tests = (
        test_policy_only_and_spaces,
        test_terminal_capture_and_replay_semantics,
        test_hand_only_actions_expand_at_environment_boundary_and_stay_14d_in_replay,
        test_coupled_actions_and_observations_use_native_identity_boundary,
        test_done_logic,
        test_contract_failures,
        test_episode_length_randomization_and_close,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"All {len(tests)} PickTool FlashSAC adapter tests passed.")


if __name__ == "__main__":
    main()
