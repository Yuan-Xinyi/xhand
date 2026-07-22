#!/usr/bin/env python3
"""Isaac smoke test for observation-aligned PickTool public safety state."""

from __future__ import annotations

import argparse

import torch


@torch.inference_mode()
def run(*, num_envs: int, seed: int, device: str) -> None:
    from adapter import make_pick_tool_env
    from xhand_inhand.tasks.direct.pick_tool_token.public_gate_state import (
        PUBLIC_GATE_STATE_EXTRAS_KEY,
        public_gate_feature_tensor,
        validate_public_gate_feature_values,
    )

    if num_envs < 2:
        raise ValueError("public gate smoke requires at least two environments")
    env = make_pick_tool_env(
        num_envs=num_envs,
        device=device,
        seed=seed,
        strict=True,
        validate_finite=True,
    )
    try:
        observation, reset_info = env.reset(seed=seed, randomize_episode_lengths=False)
        if observation.shape != (num_envs, 115):
            raise RuntimeError("public side channel changed the frozen obs115 contract")
        reset_features = public_gate_feature_tensor(
            reset_info[PUBLIC_GATE_STATE_EXTRAS_KEY],
            num_envs=num_envs,
            device=env.device,
        )
        validate_public_gate_feature_values(reset_features)
        if not torch.equal(reset_features, torch.zeros_like(reset_features)):
            raise RuntimeError("explicit reset did not publish zero force-counter progress")

        task = env.unwrapped
        cfg = task.cfg
        hard_limit = int(cfg.tactile_hard_terminate_steps)
        overforce_limit = int(cfg.tactile_terminate_steps)
        task._hard_force_steps[1] = hard_limit // 2 - 1
        task._overforce_steps[0] = overforce_limit - 1
        original_signals = task._compute_grasp_signals

        def injected_force_signals() -> dict[str, torch.Tensor]:
            signals = dict(original_signals())
            force = signals["force_magnitude"].clone()
            force[0, 0] = float(cfg.tactile_terminate_force_limit) + 1.0
            force[1, 0] = float(cfg.tactile_hard_force_limit) + 1.0
            signals["force_magnitude"] = force
            return signals

        # Instance assignment intentionally supplies the no-argument callable
        # used by PickTool internals; restore it before closing the simulator.
        vars(task)["_compute_grasp_signals"] = injected_force_signals
        try:
            action = torch.zeros((num_envs, 21), dtype=torch.float32, device=env.device)
            next_observation, _, terminated, truncated, info = env.step(action)
        finally:
            del vars(task)["_compute_grasp_signals"]

        features = public_gate_feature_tensor(
            info[PUBLIC_GATE_STATE_EXTRAS_KEY],
            num_envs=num_envs,
            device=env.device,
        )
        validate_public_gate_feature_values(features)
        if not bool(terminated[0]) or bool(truncated[0]):
            raise RuntimeError("injected overforce did not terminate env slot 0")
        terminal = info.get("pick_tool_terminal", {})
        if not bool(terminal["unsafe_force"][0]):
            raise RuntimeError("terminal truth omitted injected unsafe-force event")
        if not torch.equal(features[0], torch.zeros(2, device=env.device)):
            raise RuntimeError("auto-reset row leaked reset-before force-counter progress")
        expected_hard = float(hard_limit // 2) / float(hard_limit)
        expected_survivor = torch.tensor(
            (expected_hard, 0.0), dtype=torch.float32, device=env.device
        )
        if not torch.equal(
            features[1],
            expected_survivor,
        ):
            raise RuntimeError("survivor row force-counter progress is wrong or misordered")
        old_public_state = info[PUBLIC_GATE_STATE_EXTRAS_KEY]
        current = public_gate_feature_tensor(
            task.get_public_gate_state_v1(), num_envs=num_envs, device=env.device
        )
        validate_public_gate_feature_values(current)
        if not torch.equal(current[0], torch.zeros(2, device=env.device)):
            raise RuntimeError("auto-reset did not clear the live force counters")
        if next_observation.shape != (num_envs, 115):
            raise RuntimeError("step observation violated the frozen obs115 contract")
        env.step(torch.zeros((num_envs, 21), dtype=torch.float32, device=env.device))
        old_features = public_gate_feature_tensor(
            old_public_state, num_envs=num_envs, device=env.device
        )
        if not torch.equal(old_features[1], expected_survivor):
            raise RuntimeError("a later step mutated an earlier adapter info side channel")
        print(
            "[public-gate-state-smoke] PASS "
            f"num_envs={num_envs} reset_row={float(features[0, 1]):.1f} "
            f"survivor_hard={float(features[1, 0]):.3f} reset_live={float(current[0, 1]):.1f}",
            flush=True,
        )
    finally:
        env.close()


def parse_args() -> tuple[argparse.Namespace, object]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--num_envs", "--num-envs", dest="num_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=266)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 2:
        parser.error("--num_envs must be at least 2")
    launcher = AppLauncher(args)
    return args, launcher.app


def main() -> None:
    args, simulation_app = parse_args()
    try:
        run(num_envs=args.num_envs, seed=args.seed, device=str(args.device or "cuda:0"))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
