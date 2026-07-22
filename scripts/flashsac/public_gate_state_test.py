#!/usr/bin/env python3
"""Simulation-free contract tests for PickTool's public safety side channel."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/public_gate_state.py"
)
SPEC = importlib.util.spec_from_file_location("pick_tool_public_gate_state", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _expect_error(
    error_type: type[BaseException], function: Callable[..., Any], *args: Any, **kwargs: Any
) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def test_counter_progress_and_feature_order() -> None:
    assert MODULE.PUBLIC_GATE_STATE_VERSION == 1
    assert MODULE.PUBLIC_GATE_STATE_EXTRAS_KEY == MODULE.PUBLIC_GATE_STATE_CONTRACT
    assert MODULE.PUBLIC_GATE_STATE_CONTRACT == "pick_tool_public_gate_state_v1"
    hard = torch.tensor((0, 1, 9, 10, 11), dtype=torch.long)
    over = torch.tensor((0, 1, 2, 3, 0), dtype=torch.long)
    hard_before, over_before = hard.clone(), over.clone()
    state = MODULE.build_public_gate_state(
        hard,
        over,
        hard_terminate_steps=10,
        overforce_terminate_steps=2,
    )
    assert torch.equal(hard, hard_before) and torch.equal(over, over_before)
    assert torch.allclose(
        state["hard_force_count_progress"],
        torch.tensor((0.0, 0.1, 0.9, 1.0, 1.0)),
    )
    assert torch.allclose(
        state["overforce_count_progress"],
        torch.tensor((0.0, 0.5, 1.0, 1.0, 0.0)),
    )
    features = MODULE.public_gate_feature_tensor(state, num_envs=5, device="cpu")
    assert features.shape == (5, 2)
    assert torch.equal(features[:, 0], state["hard_force_count_progress"])
    assert torch.equal(features[:, 1], state["overforce_count_progress"])
    hard[0] = 10
    over[0] = 2
    assert float(state["hard_force_count_progress"][0]) == 0.0
    assert float(state["overforce_count_progress"][0]) == 0.0


def test_validation_fails_closed() -> None:
    counts = torch.zeros(2, dtype=torch.long)
    _expect_error(
        ValueError,
        MODULE.build_public_gate_state,
        counts,
        counts,
        hard_terminate_steps=0,
        overforce_terminate_steps=2,
    )
    _expect_error(
        TypeError,
        MODULE.build_public_gate_state,
        counts.float(),
        counts,
        hard_terminate_steps=10,
        overforce_terminate_steps=2,
    )
    state = MODULE.build_public_gate_state(
        counts,
        counts,
        hard_terminate_steps=10,
        overforce_terminate_steps=2,
    )
    corrupt = dict(state)
    corrupt["private_clearance"] = torch.zeros(2)
    _expect_error(KeyError, MODULE.public_gate_feature_tensor, corrupt)
    corrupt = dict(state)
    corrupt["hard_force_count_progress"] = torch.tensor((0.0, 1.1))
    corrupt_features = MODULE.public_gate_feature_tensor(corrupt)
    _expect_error(
        ValueError, MODULE.validate_public_gate_feature_values, corrupt_features
    )


def test_environment_publication_order_and_observation_contract() -> None:
    source_path = (
        ROOT
        / "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/pick_tool_token_env.py"
    )
    source = source_path.read_text(encoding="utf-8")
    observation_start = source.index("    def _get_observations(")
    publication = source.index(
        "self.extras[PUBLIC_GATE_STATE_EXTRAS_KEY] = self.get_public_gate_state_v1()",
        observation_start,
    )
    observation_return = source.index('return {"policy": obs, "critic": obs}', publication)
    done_start = source.index("    def _get_dones(", observation_return)
    hard_update = source.index("self._hard_force_steps = torch.where", done_start)
    overforce_update = source.index("self._overforce_steps = torch.where", hard_update)
    unsafe = source.index("unsafe_force = (", overforce_update)
    assert publication < observation_return < hard_update < overforce_update < unsafe
    assert "aligned with the returned observation" in source[publication - 500 : publication]
    assert "reset-before" not in source[overforce_update:unsafe].lower()
    assert "cfg.observation_space != 115 or cfg.state_space != 115" in source


if __name__ == "__main__":
    test_counter_progress_and_feature_order()
    test_validation_fails_closed()
    test_environment_publication_order_and_observation_contract()
