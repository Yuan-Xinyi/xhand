# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Capture the policy's OWN near-tool states as a curriculum boundary dataset.

The merged nudge+grasp policy latches from oracle-pregrasp spawns but not from home: closure
competence lives around states the ORACLE reached, not the states the POLICY reaches after its
own approach.  This script rolls the current policy from the home spawn and snapshots full
boundary states (the latch-complete schema) whenever the hand is meaningfully engaged with the
tool; training with these as curriculum spawns aligns the closure-learning distribution with the
states the policy actually visits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Self-play boundary capture for nudge+grasp.")
parser.add_argument("--checkpoint", type=Path, required=True, help="FlashSAC checkpoint dir")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument(
    "--engage_dist",
    type=float,
    default=0.08,
    help="posture-agnostic engagement: mean fingertip-to-tool distance below this (m), or any "
    "object contact force.  The gated proximity is deliberately NOT used -- it is ~0 for the "
    "policy's own pushing postures, which is precisely the distribution being captured.",
)
parser.add_argument("--capture_every", type=int, default=15, help="steps between captures per env")
parser.add_argument("--max_states", type=int, default=4096)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "rl_games"))
import agent_bridge  # noqa: E402,F401
from flash_rl.agents.flashSAC.network import FlashSACActor  # noqa: E402

from pick_tool_shared import capture_boundary, sha256  # noqa: E402

_COMPILED_PREFIX = "_orig_mod."


def load_actor(checkpoint: Path, device: torch.device) -> FlashSACActor:
    payload = torch.load(checkpoint / "actor.pt", map_location="cpu", weights_only=True)
    state = payload["network_state_dict"]
    prefixed = [str(k).startswith(_COMPILED_PREFIX) for k in state]
    strip = all(prefixed) and len(prefixed) > 0
    canonical = {
        (k.removeprefix(_COMPILED_PREFIX) if strip else k): v for k, v in state.items()
    }
    actor = FlashSACActor(num_blocks=2, input_dim=115, hidden_dim=128, action_dim=21)
    actor.load_state_dict(canonical)
    return actor.to(device).eval()


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(args_cli.seed)
    n = args_cli.num_envs
    cfg = parse_env_cfg("Pick-Tool-Token-Direct-v0", device=args_cli.device, num_envs=n)
    cfg.seed = args_cli.seed
    cfg.nudge_grasp_mode = True
    cfg.episode_length_s = 10.0
    cfg.nudge_spawn_blend_min = 0.0
    cfg.nudge_spawn_blend_max = 0.0
    env = gym.make("Pick-Tool-Token-Direct-v0", cfg=cfg)
    u = env.unwrapped
    dev = u.device
    actor = load_actor(args_cli.checkpoint, dev)

    obs, _ = env.reset()
    cooldown = torch.zeros(n, dtype=torch.long, device=dev)
    collected: list[dict[str, torch.Tensor]] = []
    kept = 0

    for step in range(args_cli.steps):
        mean, _ = actor.get_mean_and_std(obs["policy"], training=False)
        action = torch.tanh(mean)
        obs, _, terminated, truncated, _ = env.step(action)
        u._compute_intermediate_values()
        signals = u._compute_grasp_signals()
        near = u._curr_fingertip_distances.mean(dim=-1) <= args_cli.engage_dist
        touching = signals["force_magnitude"].max(dim=-1).values >= cfg.contact_force_thr
        engaged = (near | touching) & (~u._is_grasped)
        cooldown = (cooldown - 1).clamp_min(0)
        pick = engaged & (cooldown == 0) & ~(terminated | truncated)
        if bool(pick.any()) and kept < args_cli.max_states:
            snap = capture_boundary(u)
            ids = pick.nonzero(as_tuple=False).squeeze(-1)
            collected.append({key: value[ids].cpu() for key, value in snap.items()})
            kept += int(ids.numel())
            cooldown[pick] = args_cli.capture_every
        if kept >= args_cli.max_states:
            break

    if not collected:
        raise RuntimeError("no engaged states captured; raise --engage_dist")
    boundary = {
        key: torch.cat([part[key] for part in collected])[: args_cli.max_states]
        for key in collected[0]
    }
    count = boundary["joint_pos"].shape[0]
    for key, value in boundary.items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite values in boundary field {key}")
    dataset = {
        "boundaries": {"close_start": boundary},
        "meta": {
            "format_version": 1,
            "kind": "selfplay_nudge_grasp_boundaries",
            "source_checkpoint": str(args_cli.checkpoint.resolve()),
            "num_states": int(count),
            "engage_dist": args_cli.engage_dist,
            "capture_every": args_cli.capture_every,
            "seed": args_cli.seed,
        },
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, args_cli.output)
    dataset["meta"]["dataset_sha256"] = sha256(args_cli.output)
    print(f"captured {count} self-play boundary states -> {args_cli.output}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
