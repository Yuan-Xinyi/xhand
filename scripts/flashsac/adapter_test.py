#!/usr/bin/env python3
"""Pure-Torch logic tests for the PickTool FlashSAC adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from adapter import (
    ACTION_DIM,
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


class _FakeDirectEnv:
    """Minimal auto-reset environment with the ordering used by DirectRLEnv."""

    def __init__(self, *, call_reset_for_done: bool = True) -> None:
        self.num_envs = 3
        self.device = "cpu"
        self.cfg = _Cfg()
        self.max_episode_length = 100
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.call_reset_for_done = call_reset_for_done
        self.state = torch.zeros(self.num_envs, POLICY_OBSERVATION_DIM)
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
