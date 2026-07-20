# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Compare hand action manifolds from one identical, policy-generated hammer pregrasp.

The benchmark first rolls out a compatible 16-action checkpoint and snapshots the best legal
tabletop pregrasp.  It then restores that exact robot/object state before every trial and gives
the same CEM candidate budget to three absolute hand-target parameterizations:

* ``token9``: the existing CrossDex token and RetargetNN;
* ``raw12``: all twelve XHand joints over their runtime limits;
* ``hybrid14``: token9 plus independent residuals on the five distal flexion joints which the
  DexPilot retargeter regularizes at their midpoints.

The arm target is frozen throughout closure.  Ranking uses a dense, object-filtered close score
only to make search possible; the pass/fail result is the environment's shared robust grasp
quality/latch.  True clearance, object drift and per-finger object forces expose crush/launch
solutions rather than silently accepting them.
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Fixed-pregrasp hand-space feasibility benchmark.")
parser.add_argument("--task", type=str, default="Pick-Tool-Token-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--population", type=int, default=256, help="candidates per action space")
parser.add_argument("--iterations", type=int, default=10)
parser.add_argument("--approach_steps", type=int, default=180)
parser.add_argument("--close_steps", type=int, default=24)
parser.add_argument("--eval_steps", type=int, default=8)
parser.add_argument("--elite_frac", type=float, default=0.10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=str, default="/tmp/pick_tool_hand_space_feasibility.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.population < 16:
    parser.error("--population must be at least 16")
if not (0.02 <= args_cli.elite_frac <= 0.5):
    parser.error("--elite_frac must lie in [0.02, 0.5]")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

import xhand_inhand.tasks  # noqa: F401


MODES = ("token9", "raw12", "hybrid14")
DISTAL_JOINT_NAMES = (
    "thumb_joint2",
    "index_joint2",
    "middle_joint1",
    "ring_joint1",
    "pinky_joint1",
)


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().item())


def _quat_angle(reference: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    dot = (reference * current).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def main() -> None:
    torch.manual_seed(args_cli.seed)
    num_envs = len(MODES) * args_cli.population
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = 120.0
    env_cfg.terminate_on_drop = False
    env_cfg.success_hold_steps = 100000

    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    agent_cfg["params"]["seed"] = args_cli.seed
    agent_cfg["params"]["config"]["full_experiment_name"] = "0_hand_space_feasibility"
    agent_cfg["params"]["config"]["num_actors"] = num_envs
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args_cli.checkpoint
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", 5.0)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", 1.0)

    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RlGamesVecEnvWrapper(
        base_env, agent_cfg["params"]["config"]["device"], clip_obs, clip_actions, None, True
    )
    vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env}
    )
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(args_cli.checkpoint)
    agent.reset()

    u = env.unwrapped
    dev = u.device
    all_ids = u.robot._ALL_INDICES
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    hand_names = [u.robot.joint_names[i] for i in u._hand_ids_t.tolist()]
    distal_local = torch.tensor(
        [hand_names.index(name) for name in DISTAL_JOINT_NAMES], dtype=torch.long, device=dev
    )
    hand_lower = u.dof_lower[:, u._hand_ids_t]
    hand_upper = u.dof_upper[:, u._hand_ids_t]
    hand_span = (hand_upper - hand_lower).clamp_min(1.0e-6)

    print("\n=== ACTION MANIFOLD ===", flush=True)
    print(f"articulation hand order: {hand_names}", flush=True)
    print(
        "distal residual joints: "
        + ", ".join(f"{name}[{int(idx)}]" for name, idx in zip(DISTAL_JOINT_NAMES, distal_local)),
        flush=True,
    )
    manifold_generator = torch.Generator(device=dev)
    manifold_generator.manual_seed(12345)
    manifold_actions = 2.0 * torch.rand(
        (65536, u._n_tokens), generator=manifold_generator, device=dev
    ) - 1.0
    with torch.inference_mode():
        manifold_nn = u.retarget.retarget_from_unit_action(manifold_actions)
        manifold_q = manifold_nn[:, u._retarget2isaac]
        manifold_q = torch.maximum(
            torch.minimum(manifold_q, hand_upper[0]), hand_lower[0]
        )
    manifold_std = manifold_q.std(dim=0)
    manifold_range = manifold_q.amax(dim=0) - manifold_q.amin(dim=0)
    for name, std, span in zip(hand_names, manifold_std.tolist(), manifold_range.tolist()):
        suffix = "  <-- distal" if name in DISTAL_JOINT_NAMES else ""
        print(f"token manifold {name:15s}: std={std:.5f} rad range={span:.5f} rad{suffix}", flush=True)

    # ------------------------------------------------------------------ policy pregrasp mining
    best_score = -float("inf")
    snapshot: dict[str, torch.Tensor | float | int] = {}
    print(
        f"\n=== PREGRASP MINING: {num_envs} envs x {args_cli.approach_steps} steps ===",
        flush=True,
    )
    for step in range(args_cli.approach_steps):
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, _, _ = env.step(actions)

        d = u._curr_fingertip_distances
        other_d = d[:, u._other_ee_idx]
        near_d, near_i = torch.topk(other_d, k=2, dim=1, largest=False)
        grasp_dist = (d[:, u._thumb_ee_idx] + near_d.sum(dim=-1)) / 3.0
        other_align = u._finger_align[:, u._other_ee_idx]
        alignment = (
            u._finger_align[:, u._thumb_ee_idx]
            + torch.gather(other_align, 1, near_i).sum(dim=-1)
        ) / 3.0
        to_handle = u.handle_center_w - u.palm_center_w
        to_handle = to_handle / to_handle.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        palm_facing = 0.5 * (1.0 + (u.palm_normal_w * to_handle).sum(dim=-1))
        clearance = u._object_true_min_z() - u._table_surface_z
        # Same geometry that made the old policy a useful approach initializer, while disallowing
        # already-airborne/tipped snapshots.  The exponential provides a sharp preference below 2cm.
        score = torch.exp(-grasp_dist / 0.025) * alignment * palm_facing
        score = torch.where(clearance.abs() <= 0.005, score, torch.full_like(score, -1.0))
        value, idx = score.max(dim=0)
        if _scalar(value) > best_score:
            i = int(idx.item())
            best_score = _scalar(value)
            snapshot = {
                "step": step,
                "source_env": i,
                "score": best_score,
                "grasp_dist": _scalar(grasp_dist[i]),
                "alignment": _scalar(alignment[i]),
                "palm_facing": _scalar(palm_facing[i]),
                "clearance": _scalar(clearance[i]),
                "joint_pos": u.robot.data.joint_pos[i].detach().clone(),
                "object_local_pos": (
                    u.object.data.root_pos_w[i] - u.scene.env_origins[i]
                ).detach().clone(),
                "object_quat": u.object.data.root_quat_w[i].detach().clone(),
                "token": actions[i, u._n_arm :].detach().clone().clamp(-1.0, 1.0),
            }
        if step % 30 == 0 or step == args_cli.approach_steps - 1:
            print(
                f"approach {step:4d}: best={best_score:.4f} "
                f"dist={float(snapshot.get('grasp_dist', float('nan'))):.4f} "
                f"palm={float(snapshot.get('palm_facing', float('nan'))):.3f} "
                f"align={float(snapshot.get('alignment', float('nan'))):.3f}",
                flush=True,
            )

    if not snapshot or best_score <= 0.0:
        raise RuntimeError("No legal tabletop pregrasp was found by the checkpoint rollout.")

    snap_joint = snapshot["joint_pos"]
    snap_hand = snap_joint[u._hand_ids_t]
    snap_arm = snap_joint[u._arm_ids_t]
    snap_obj_local = snapshot["object_local_pos"]
    snap_obj_quat = snapshot["object_quat"]
    seed_token = snapshot["token"]
    print(
        "PREGRASP SELECTED: "
        f"step={snapshot['step']} env={snapshot['source_env']} score={snapshot['score']:.4f} "
        f"dist={snapshot['grasp_dist']:.4f}m palm={snapshot['palm_facing']:.3f} "
        f"align={snapshot['alignment']:.3f} clearance={snapshot['clearance']:+.5f}m",
        flush=True,
    )

    # ------------------------------------------------------------------ fixed-state/direct-target harness
    benchmark_hand_target = snap_hand.unsqueeze(0).repeat(num_envs, 1)

    def benchmark_pre_physics(self, actions: torch.Tensor) -> None:
        self.actions = torch.zeros_like(actions)
        self.dof_targets[:, self._arm_ids_t] = snap_arm
        self.dof_targets[:, self._hand_ids_t] = benchmark_hand_target
        self.dof_targets[:] = torch.clamp(self.dof_targets, self.dof_lower, self.dof_upper)

    u._pre_physics_step = types.MethodType(benchmark_pre_physics, u)
    zero_action = torch.zeros((num_envs, u.cfg.action_space), device=dev)

    @torch.inference_mode()
    def restore_snapshot() -> None:
        joint = snap_joint.unsqueeze(0).repeat(num_envs, 1)
        u.robot.write_joint_state_to_sim(joint, torch.zeros_like(joint), env_ids=all_ids)
        u.robot.set_joint_position_target(joint, env_ids=all_ids)
        u.dof_targets[:] = joint
        benchmark_hand_target[:] = snap_hand
        pose = torch.zeros((num_envs, 7), device=dev)
        pose[:, :3] = snap_obj_local + u.scene.env_origins
        pose[:, 3:7] = snap_obj_quat
        u.object.write_root_pose_to_sim(pose, env_ids=all_ids)
        u.object.write_root_velocity_to_sim(torch.zeros((num_envs, 6), device=dev), env_ids=all_ids)
        u.episode_length_buf.zero_()
        u._contact_steps.zero_()
        u._lost_contact_steps.zero_()
        u._is_grasped.zero_()
        u._grasp_bonus_given.zero_()
        u._success_paid.zero_()
        u._success_steps.zero_()
        u._is_success.zero_()
        u.actions.zero_()
        u.prev_actions.zero_()
        u._compute_intermediate_values()

    def decode(mode: str, latent: torch.Tensor, row_slice: slice) -> torch.Tensor:
        lower = hand_lower[row_slice]
        upper = hand_upper[row_slice]
        if mode == "raw12":
            return lower + 0.5 * (latent + 1.0) * (upper - lower)
        token = latent[:, :9]
        target_nn = u.retarget.retarget_from_unit_action(token)
        target = target_nn[:, u._retarget2isaac]
        target = torch.maximum(torch.minimum(target, upper), lower)
        if mode == "hybrid14":
            residual = latent[:, 9:]
            base_distal = target[:, distal_local]
            lo_distal = lower[:, distal_local]
            hi_distal = upper[:, distal_local]
            # Feasibility mapping: -1/0/+1 exactly reach lower/base/upper without an arbitrary
            # symmetric scale.  The later PPO action can use a narrower regularized residual.
            delta = torch.where(
                residual >= 0.0,
                residual * (hi_distal - base_distal),
                residual * (base_distal - lo_distal),
            )
            target[:, distal_local] = base_distal + delta
        return target

    def dense_close_score() -> torch.Tensor:
        # Dense search-only bridge: pad proximity and real object force are additive per finger,
        # while alignment/opposition/palm gates retain the semantics of the strict wrap truth.
        signals = u._compute_grasp_signals()
        region = u._handle_contact_region.float()
        near = torch.exp(-u._handle_side_distances / 0.012) * region
        force = signals["contact_strength"]
        contact_or_near = torch.maximum(force, near)
        align = torch.clamp(
            (u._finger_align - u.cfg.grasp_align_min)
            / max(1.0 - u.cfg.grasp_align_min, 1.0e-6),
            0.0,
            1.0,
        )
        ti = u._contact_thumb_idx
        oi = u._contact_other_ids
        thumb_normal = u._handle_surface_normal_w[:, ti].unsqueeze(1)
        opposition = 0.5 * (
            1.0 - (thumb_normal * u._handle_surface_normal_w[:, oi]).sum(dim=-1)
        )
        opposition = torch.clamp(
            (opposition - u.cfg.grasp_opposition_min)
            / max(1.0 - u.cfg.grasp_opposition_min, 1.0e-6),
            0.0,
            1.0,
        )
        thumb = contact_or_near[:, ti] * align[:, ti]
        others = contact_or_near[:, oi] * align[:, oi] * opposition
        best_two = torch.topk(others, k=2, dim=1).values
        coverage = (thumb + best_two.sum(dim=-1)) / 3.0
        return coverage * signals["palm_score"]

    dims = {"token9": 9, "raw12": 12, "hybrid14": 14}
    slices = {
        mode: slice(i * args_cli.population, (i + 1) * args_cli.population)
        for i, mode in enumerate(MODES)
    }
    raw_seed = (2.0 * (snap_hand - hand_lower[0]) / hand_span[0] - 1.0).clamp(-1.0, 1.0)
    means = {
        "token9": seed_token.clone(),
        "raw12": raw_seed.clone(),
        "hybrid14": torch.cat((seed_token.clone(), torch.zeros(5, device=dev))),
    }
    stds = {
        "token9": torch.full((9,), 0.35, device=dev),
        "raw12": torch.full((12,), 0.35, device=dev),
        "hybrid14": torch.cat(
            (torch.full((9,), 0.35, device=dev), torch.full((5,), 0.55, device=dev))
        ),
    }
    best: dict[str, dict] = {
        mode: {"score": -float("inf"), "latent": means[mode].clone()} for mode in MODES
    }
    elite_count = max(8, int(round(args_cli.population * args_cli.elite_frac)))

    print(
        f"\n=== CEM: {args_cli.population} candidates/mode x {args_cli.iterations} iterations; "
        f"close={args_cli.close_steps}, eval={args_cli.eval_steps}, elites={elite_count} ===",
        flush=True,
    )
    for iteration in range(args_cli.iterations):
        latent_by_mode: dict[str, torch.Tensor] = {}
        target_all = torch.empty((num_envs, len(hand_names)), device=dev)
        for mode in MODES:
            latent = means[mode] + stds[mode] * torch.randn(
                (args_cli.population, dims[mode]), device=dev
            )
            latent = latent.clamp(-1.0, 1.0)
            latent[0] = means[mode].clamp(-1.0, 1.0)
            if iteration == 0:
                if mode == "token9":
                    latent[1] = seed_token
                elif mode == "raw12":
                    latent[1] = raw_seed
                else:
                    latent[1, :9] = seed_token
                    latent[1, 9:] = 0.0
            latent_by_mode[mode] = latent
            target_all[slices[mode]] = decode(mode, latent, slices[mode])

        restore_snapshot()
        q_wrap_sum = torch.zeros(num_envs, device=dev)
        q_grasp_sum = torch.zeros(num_envs, device=dev)
        dense_sum = torch.zeros(num_envs, device=dev)
        latch_sum = torch.zeros(num_envs, device=dev)
        force_peak = torch.zeros((num_envs, len(u.ee_names)), device=dev)
        clearance_peak = torch.full((num_envs,), -float("inf"), device=dev)
        eval_count = 0
        total_steps = args_cli.close_steps + args_cli.eval_steps
        for step in range(total_steps):
            if step < args_cli.close_steps:
                x = float(step + 1) / float(args_cli.close_steps)
                blend = x * x * (3.0 - 2.0 * x)
            else:
                blend = 1.0
            benchmark_hand_target[:] = snap_hand + blend * (target_all - snap_hand)
            with torch.inference_mode():
                env.step(zero_action)
            force_peak = torch.maximum(force_peak, u._finger_object_force_magnitudes())
            clearance = u._object_true_min_z() - u._table_surface_z
            clearance_peak = torch.maximum(clearance_peak, clearance)
            if step >= args_cli.close_steps:
                signals = u._compute_grasp_signals()
                q_wrap_sum += signals["quality"]
                q_grasp_sum += signals["grasp_quality"]
                dense_sum += dense_close_score()
                latch_sum += u._is_grasped.float()
                eval_count += 1

        q_wrap_mean = q_wrap_sum / eval_count
        q_grasp_mean = q_grasp_sum / eval_count
        dense_mean = dense_sum / eval_count
        latch_frac = latch_sum / eval_count
        object_local = u.object.data.root_pos_w - u.scene.env_origins
        xy_drift = (object_local[:, :2] - snap_obj_local[:2]).norm(dim=-1)
        rot_drift = _quat_angle(snap_obj_quat.unsqueeze(0), u.object.data.root_quat_w)
        launch = (clearance_peak > 0.02) & (latch_frac < 0.5)
        score_all = (
            2.0 * latch_frac
            + 2.0 * q_grasp_mean
            + q_wrap_mean
            + dense_mean
            - 4.0 * torch.relu(xy_drift - 0.02)
            - 0.25 * rot_drift
            - 2.0 * launch.float()
        )

        line = []
        for mode in MODES:
            sl = slices[mode]
            scores = score_all[sl]
            top = torch.topk(scores, k=elite_count, largest=True)
            elite = latent_by_mode[mode][top.indices]
            means[mode] = elite.mean(dim=0)
            stds[mode] = elite.std(dim=0, unbiased=False).clamp(0.04, 0.75)
            local_best = int(torch.argmax(scores).item())
            global_idx = sl.start + local_best
            value = _scalar(scores[local_best])
            if value > best[mode]["score"]:
                best[mode] = {
                    "score": value,
                    "latent": latent_by_mode[mode][local_best].detach().clone(),
                    "target": target_all[global_idx].detach().clone(),
                    "q_wrap": _scalar(q_wrap_mean[global_idx]),
                    "q_grasp": _scalar(q_grasp_mean[global_idx]),
                    "dense_close": _scalar(dense_mean[global_idx]),
                    "latch_fraction": _scalar(latch_frac[global_idx]),
                    "clearance_peak": _scalar(clearance_peak[global_idx]),
                    "xy_drift": _scalar(xy_drift[global_idx]),
                    "rotation_drift_rad": _scalar(rot_drift[global_idx]),
                    "force_peak": force_peak[global_idx].detach().clone(),
                }
            b = best[mode]
            line.append(
                f"{mode}: score={b['score']:.3f} latch={b.get('latch_fraction', 0):.2f} "
                f"q={b.get('q_grasp', 0):.3f} wrap={b.get('q_wrap', 0):.3f}"
            )
        print(f"iter {iteration + 1:02d}: " + " | ".join(line), flush=True)

    # ------------------------------------------------------------------ report
    output = {
        "checkpoint": str(Path(args_cli.checkpoint).resolve()),
        "seed": args_cli.seed,
        "population_per_mode": args_cli.population,
        "iterations": args_cli.iterations,
        "pregrasp": {
            key: (value.detach().cpu().tolist() if torch.is_tensor(value) else value)
            for key, value in snapshot.items()
        },
        "hand_joint_names": hand_names,
        "fingertip_force_order": list(u.ee_names),
        "distal_joint_names": list(DISTAL_JOINT_NAMES),
        "token_manifold": {
            name: {"std_rad": std, "range_rad": span}
            for name, std, span in zip(
                hand_names, manifold_std.detach().cpu().tolist(), manifold_range.detach().cpu().tolist()
            )
        },
        "results": {},
    }
    print("\n=== FINAL FIXED-PREGRASP COMPARISON ===", flush=True)
    for mode in MODES:
        result = best[mode]
        passed = bool(
            result.get("latch_fraction", 0.0) >= 0.5
            and result.get("q_grasp", 0.0) >= u.cfg.grasp_quality_high
            and not (
                result.get("clearance_peak", 0.0) > 0.02
                and result.get("latch_fraction", 0.0) < 0.5
            )
        )
        serial = {
            key: (value.detach().cpu().tolist() if torch.is_tensor(value) else value)
            for key, value in result.items()
        }
        serial["robust_grasp_pass"] = passed
        output["results"][mode] = serial
        print(
            f"{mode:8s} pass={str(passed):5s} latch={result.get('latch_fraction', 0):.3f} "
            f"q_grasp={result.get('q_grasp', 0):.3f} q_wrap={result.get('q_wrap', 0):.3f} "
            f"dense={result.get('dense_close', 0):.3f} "
            f"clear_peak={result.get('clearance_peak', 0):+.4f}m "
            f"xy={result.get('xy_drift', 0):.4f}m rot={result.get('rotation_drift_rad', 0):.3f}rad "
            f"force_peak={[round(x, 2) for x in result.get('force_peak', torch.zeros(5)).tolist()]}",
            flush=True,
        )

    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {output_path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
