#!/usr/bin/env python3
"""Collect physically validated 21-action hammer demonstrations for BC and curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Formal-action oracle demonstration collector.")
parser.add_argument("--feasibility_json", required=True)
parser.add_argument(
    "--pregrasp_source", choices=("fixed", "policy", "ik", "curriculum"), default="fixed"
)
parser.add_argument("--checkpoint", default=None, help="old 87/16 checkpoint for policy pregrasps")
parser.add_argument("--curriculum_dataset", default=None, help="dataset providing close_start states")
parser.add_argument("--rollout_checkpoint", default=None, help="learner actor used to visit DAgger states")
parser.add_argument("--teacher_probability", type=float, default=1.0)
parser.add_argument("--retain_all_episodes", action="store_true")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--approach_steps", type=int, default=180)
parser.add_argument("--approach_hover_fraction", type=float, default=0.65)
parser.add_argument("--approach_hover_height", type=float, default=0.10)
parser.add_argument("--approach_settle_steps", type=int, default=30)
parser.add_argument("--close_steps", type=int, default=48)
parser.add_argument("--close_hold_steps", type=int, default=12)
parser.add_argument("--micro_steps", type=int, default=30)
parser.add_argument("--micro_hold_steps", type=int, default=40)
parser.add_argument("--lift_steps", type=int, default=160)
parser.add_argument("--settle_steps", type=int, default=20)
parser.add_argument("--micro_height", type=float, default=0.04)
parser.add_argument("--target_height", type=float, default=0.24)
parser.add_argument("--damping", type=float, default=0.05)
parser.add_argument("--max_cart_step", type=float, default=0.004)
parser.add_argument("--max_rot_step", type=float, default=0.05)
parser.add_argument("--max_joint_step", type=float, default=0.04)
parser.add_argument("--grip_force_target", type=float, default=3.0)
parser.add_argument("--grip_force_limit", type=float, default=20.0)
parser.add_argument("--grip_servo_step", type=float, default=0.006)
parser.add_argument("--grip_servo_range", type=float, default=0.60)
parser.add_argument("--min_success_fraction", type=float, default=0.75)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", default="/tmp/pick_tool_oracle_demos.pt")
parser.add_argument("--metrics", default="/tmp/pick_tool_oracle_demos.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.pregrasp_source == "policy" and not args_cli.checkpoint:
    parser.error("--checkpoint is required with --pregrasp_source policy")
if args_cli.pregrasp_source == "curriculum" and not args_cli.curriculum_dataset:
    parser.error("--curriculum_dataset is required with --pregrasp_source curriculum")
if not 0.0 <= args_cli.teacher_probability <= 1.0:
    parser.error("--teacher_probability must be in [0, 1]")
if args_cli.rollout_checkpoint is None and args_cli.teacher_probability != 1.0:
    parser.error("--teacher_probability differs from 1 but no --rollout_checkpoint was supplied")
if not 0.0 < args_cli.approach_hover_fraction < 1.0:
    parser.error("--approach_hover_fraction must be in (0, 1)")
if args_cli.approach_hover_height <= 0.0:
    parser.error("--approach_hover_height must be positive")
if args_cli.approach_settle_steps < 1:
    parser.error("--approach_settle_steps must be positive")
if args_cli.target_height < 0.20:
    parser.error("--target_height must be at least 0.20m")
if not 0.0 < args_cli.min_success_fraction <= 1.0:
    parser.error("--min_success_fraction must be in (0, 1]")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn.functional as F
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401
from bc_pick_tool import MigratedActor, clone_state, load_torch

# Single source of truth for the boundary schema shared with the env's curriculum loader.
from pick_tool_shared import capture_boundary, limit_norm, sha256
from xhand_inhand.tasks.direct.pick_tool_token.hybrid_action import (
    apply_asymmetric_joint_residual,
    invert_asymmetric_joint_residual,
)

_sha256 = sha256
_limit_norm = limit_norm
_capture_boundary = capture_boundary


PHASE_APPROACH = 0
PHASE_CLOSE = 1
PHASE_MICRO = 2
PHASE_LIFT = 3
PHASE_SETTLE = 4


def _summary(value: torch.Tensor) -> str:
    flat = value.detach().float().flatten()
    if flat.numel() <= 16:
        return str([round(x, 3) for x in flat.tolist()])
    quantiles = torch.quantile(flat, torch.tensor((0.0, 0.1, 0.5, 0.9, 1.0), device=flat.device))
    return "min/p10/median/p90/max=" + "/".join(f"{float(x):.3f}" for x in quantiles)


class LegacyReachActor:
    """Minimal deterministic forward pass for the old shared-MLP 87/16 RL-Games policy."""

    def __init__(self, checkpoint: str, device: torch.device):
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        payload = payload[0] if 0 in payload else payload
        state = payload["model"]
        self.mean = state["running_mean_std.running_mean"].float()
        self.var = state["running_mean_std.running_var"].float()
        self.layers = [
            (state[f"a2c_network.actor_mlp.{i}.weight"], state[f"a2c_network.actor_mlp.{i}.bias"])
            for i in (0, 2, 4)
        ]
        self.mu_weight = state["a2c_network.mu.weight"]
        self.mu_bias = state["a2c_network.mu.bias"]
        if self.mean.numel() != 87 or self.mu_bias.numel() != 16:
            raise RuntimeError("legacy reach checkpoint is not an 87-observation/16-action policy")

    @torch.inference_mode()
    def __call__(self, observation_prefix: torch.Tensor) -> torch.Tensor:
        x = observation_prefix.clamp(-5.0, 5.0)
        x = (x - self.mean) / torch.sqrt(self.var + 1.0e-5)
        x = x.clamp(-5.0, 5.0)
        for weight, bias in self.layers:
            x = F.elu(F.linear(x, weight, bias))
        return F.linear(x, self.mu_weight, self.mu_bias).clamp(-1.0, 1.0)


def _write_boundary(u, state: dict[str, torch.Tensor]) -> None:
    """Restore one batched physical state without advancing simulation time."""

    all_ids = u.robot._ALL_INDICES
    joint_pos = state["joint_pos"]
    joint_vel = state["joint_vel"]
    u.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=all_ids)
    u.robot.set_joint_position_target(state["dof_targets"], env_ids=all_ids)
    u.dof_targets.copy_(state["dof_targets"])
    object_pose = torch.zeros((u.num_envs, 7), device=u.device)
    object_pose[:, :3] = state["object_local_pos"] + u.scene.env_origins
    object_pose[:, 3:7] = state["object_quat"]
    u.object.write_root_pose_to_sim(object_pose, env_ids=all_ids)
    u.object.write_root_velocity_to_sim(state["object_velocity"], env_ids=all_ids)
    u.actions.copy_(state["last_action"])
    u.prev_actions.copy_(state["last_action"])
    # Restore the latch counters/flag when present so an internal snapshot round-trip matches
    # what the env's curriculum reset would load.
    if "contact_steps" in state:
        u._contact_steps.copy_(state["contact_steps"].to(dtype=u._contact_steps.dtype))
        u._lost_contact_steps.copy_(state["lost_contact_steps"].to(dtype=u._lost_contact_steps.dtype))
        u._is_grasped.copy_(state["is_grasped"].to(dtype=u._is_grasped.dtype))
    # Joint writes update generalized coordinates immediately, but rigid-body poses (including the
    # palm frame used below) are refreshed only after a forward-kinematics pass.
    u.scene.write_data_to_sim()
    u.sim.forward()
    u.scene.update(dt=u.physics_dt)


def _checkpoint_model(path: Path) -> dict[str, torch.Tensor]:
    raw = load_torch(path)
    if not isinstance(raw, dict):
        raise TypeError("rollout checkpoint root is not a dictionary")
    payload = raw[0] if 0 in raw else raw.get("0")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise KeyError("rollout checkpoint must contain {0: {'model': state_dict}}")
    return clone_state(payload["model"])


@torch.inference_mode()
def main() -> None:
    print(
        "[deprecation] oracle_demo_dataset produces two label-quality artifacts that "
        "option_oracle_dataset.py fixes: (1) the CLOSE-phase arm label is identically zero, "
        "so under --teacher_probability<1 it teaches 'do not correct' on drifted states; "
        "(2) the lift target height is scheduled by the teacher's loop counter, which is not in "
        "the 115-D observation, making those arm labels non-Markovian. Prefer "
        "option_oracle_dataset.py for state-driven, Markovian labels; use this collector only "
        "for the approach+teleport pregrasp sourcing it uniquely provides.",
        flush=True,
    )
    torch.manual_seed(args_cli.seed)
    artifact_path = Path(args_cli.feasibility_json)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    hybrid = artifact["results"]["hybrid14"]
    if not hybrid.get("robust_grasp_pass", False):
        raise RuntimeError("the selected hybrid14 feasibility result is not a robust grasp")

    cfg = parse_env_cfg("Pick-Tool-Token-Direct-v0", device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    cfg.episode_length_s = 120.0
    cfg.terminate_on_drop = False
    cfg.success_hold_steps = 100000
    # Dataset rows must remain one physical episode.  Disable the environment's safety reset here;
    # force violations remain labels/metrics and failed rollouts are either retained explicitly or
    # discarded at the end, never silently continued from an auto-reset state.
    cfg.tactile_hard_terminate_steps = 100000
    cfg.tactile_terminate_steps = 100000
    if args_cli.pregrasp_source == "curriculum":
        cfg.curriculum_dataset = args_cli.curriculum_dataset
        cfg.curriculum_boundary = "close_start"
        cfg.curriculum_reset_probability = 1.0
        cfg.curriculum_joint_noise = 0.0
    env = gym.make("Pick-Tool-Token-Direct-v0", cfg=cfg)
    u = env.unwrapped
    dev = u.device
    n = u.num_envs
    all_ids = u.robot._ALL_INDICES
    obs, _ = env.reset()

    def step_without_reset(action: torch.Tensor):
        result = env.step(action)
        terminated, truncated = result[2], result[3]
        if bool(terminated.any()) or bool(truncated.any()):
            raise RuntimeError(
                "oracle collector encountered an unexpected environment reset; "
                "no cross-episode rows were written"
            )
        return result

    rollout_actor = None
    if args_cli.rollout_checkpoint:
        rollout_actor = MigratedActor(_checkpoint_model(Path(args_cli.rollout_checkpoint))).to(dev).eval()
        if rollout_actor.observation_dim != 115 or rollout_actor.action_dim != 21:
            raise RuntimeError("DAgger rollout actor must use 115 observations and 21 actions")
        print(
            f"DAgger rollout learner={Path(args_cli.rollout_checkpoint).resolve()} "
            f"teacher_probability={args_cli.teacher_probability:.3f}",
            flush=True,
        )

    eye6 = torch.eye(6, device=dev).unsqueeze(0)

    def dls_pose_action(desired_pos: torch.Tensor, desired_quat: torch.Tensor) -> torch.Tensor:
        """Convert a desired palm pose into the environment's normalized arm delta action."""

        u._compute_intermediate_values()
        pos_error, rot_error = compute_pose_error(
            u.robot.data.body_pos_w[:, u.palm_idx],
            u.robot.data.body_quat_w[:, u.palm_idx],
            desired_pos,
            desired_quat,
            rot_error_type="axis_angle",
        )
        delta = torch.cat(
            (_limit_norm(pos_error, args_cli.max_cart_step), _limit_norm(rot_error, args_cli.max_rot_step)),
            dim=-1,
        )
        jacobian = u.robot.root_physx_view.get_jacobians()
        jacobian = jacobian[:, u._palm_jac_idx, :, :][:, :, u._arm_ids_t]
        jt = jacobian.transpose(1, 2)
        system = jacobian @ jt + (args_cli.damping**2) * eye6
        delta_q = (jt @ torch.linalg.solve(system, delta.unsqueeze(-1))).squeeze(-1)
        delta_q = delta_q.clamp(-args_cli.max_joint_step, args_cli.max_joint_step)
        return (delta_q / (cfg.act_moving_average * cfg.action_scale)).clamp(-1.0, 1.0)

    hand_names = [u.robot.joint_names[i] for i in u._hand_ids_t.tolist()]
    if hand_names != artifact["hand_joint_names"]:
        raise RuntimeError("artifact hand-joint order differs from the runtime articulation")

    def fit_formal_pregrasp_action(target_hand: torch.Tensor) -> torch.Tensor:
        """Fit token9+residual5 to a physical pregrasp hand configuration.

        A policy token is an instantaneous action, not necessarily the steady hand target that
        produced its snapshot after EMA filtering.  Holding that token throughout an IK approach
        changes the hand shape and moves the close initiation set.  A small CEM fit keeps every
        command on the formal 14-D hand-action manifold while reproducing the reference joints.
        """

        population = 4096
        elite_count = 64
        center = torch.tensor(artifact["pregrasp"]["token"], device=dev).unsqueeze(0)
        sigma = torch.full_like(center, 0.35)
        hand_lower_one = u.dof_lower[:1, u._hand_ids_t]
        hand_upper_one = u.dof_upper[:1, u._hand_ids_t]
        non_distal = torch.ones(len(hand_names), dtype=torch.bool, device=dev)
        non_distal[u._distal_hand_ids] = False
        desired = target_hand[:1]
        best_token = center.clone()
        best_loss = torch.full((), float("inf"), device=dev)
        for _ in range(8):
            candidates = (center + sigma * torch.randn((population, u._n_tokens), device=dev)).clamp(
                -1.0, 1.0
            )
            candidates[0] = best_token[0]
            decoded = u.retarget.retarget_from_unit_action(candidates)[:, u._retarget2isaac]
            loss = (decoded[:, non_distal] - desired[:, non_distal]).square().mean(dim=-1)
            iteration_best = torch.argmin(loss)
            if loss[iteration_best] < best_loss:
                best_loss = loss[iteration_best]
                best_token = candidates[iteration_best : iteration_best + 1].clone()
            elite = candidates[torch.topk(loss, k=elite_count, largest=False).indices]
            center = elite.mean(dim=0, keepdim=True)
            sigma = elite.std(dim=0, keepdim=True).clamp_min(0.01)

        token_target = u.retarget.retarget_from_unit_action(best_token)[:, u._retarget2isaac]
        token_target = torch.maximum(torch.minimum(token_target, hand_upper_one), hand_lower_one)
        residual = invert_asymmetric_joint_residual(
            desired, token_target, hand_lower_one, hand_upper_one, u._distal_hand_ids
        )
        decoded, _ = apply_asymmetric_joint_residual(
            token_target,
            hand_lower_one,
            hand_upper_one,
            residual,
            u._distal_hand_ids,
        )
        error = decoded - desired
        print(
            f"formal pregrasp fit rmse={float(error.square().mean().sqrt()):.4f}rad "
            f"max={float(error.abs().max()):.4f}rad",
            flush=True,
        )
        return torch.cat((best_token, residual), dim=-1).repeat(n, 1)

    approach_obs: list[torch.Tensor] = []
    approach_action: list[torch.Tensor] = []
    best_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    approach_teacher_rows = 0
    approach_total_rows = 0

    if args_cli.pregrasp_source == "fixed":
        snap_joint = torch.tensor(artifact["pregrasp"]["joint_pos"], device=dev).repeat(n, 1)
        snap_obj_local = torch.tensor(artifact["pregrasp"]["object_local_pos"], device=dev).repeat(n, 1)
        snap_obj_quat = torch.tensor(artifact["pregrasp"]["object_quat"], device=dev).repeat(n, 1)
        snap_token = torch.tensor(artifact["pregrasp"]["token"], device=dev).repeat(n, 1)
        snap_hand_action = torch.cat(
            (snap_token, torch.zeros((n, u._n_distal_residuals), device=dev)), dim=-1
        )
        pregrasp_score = torch.full((n,), float(artifact["pregrasp"]["score"]), device=dev)
    elif args_cli.pregrasp_source == "policy":
        actor = LegacyReachActor(args_cli.checkpoint, dev)
        pregrasp_score = torch.full((n,), -float("inf"), device=dev)
        snap_joint = u.robot.data.joint_pos.detach().clone()
        snap_obj_local = (u.object.data.root_pos_w - u.scene.env_origins).detach().clone()
        snap_obj_quat = u.object.data.root_quat_w.detach().clone()
        snap_token = torch.zeros((n, u._n_tokens), device=dev)
        snap_hand_action = torch.zeros(
            (n, u._n_tokens + u._n_distal_residuals), device=dev
        )
        print(f"mining {n} per-env policy pregrasps", flush=True)
        for step in range(args_cli.approach_steps):
            old_action = actor(obs["policy"][:, :87])
            action = torch.zeros((n, cfg.action_space), device=dev)
            action[:, :16] = old_action
            approach_obs.append(obs["policy"].clamp(-5.0, 5.0).detach().clone())
            approach_action.append(action.detach().clone())
            obs, _, _, _, _ = step_without_reset(action)

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
            score = torch.exp(-grasp_dist / 0.025) * alignment * palm_facing
            score = torch.where(clearance.abs() <= 0.005, score, torch.full_like(score, -1.0))
            better = score > pregrasp_score
            pregrasp_score = torch.where(better, score, pregrasp_score)
            best_step = torch.where(better, torch.full_like(best_step, step), best_step)
            snap_joint[better] = u.robot.data.joint_pos[better]
            snap_obj_local[better] = u.object.data.root_pos_w[better] - u.scene.env_origins[better]
            snap_obj_quat[better] = u.object.data.root_quat_w[better]
            snap_token[better] = old_action[better, u._n_arm :]
            snap_hand_action[better, : u._n_tokens] = snap_token[better]
        print(
            f"pregrasp score min/median/max={float(pregrasp_score.min()):.3f}/"
            f"{float(pregrasp_score.median()):.3f}/{float(pregrasp_score.max()):.3f}",
            flush=True,
        )
    elif args_cli.pregrasp_source == "ik":
        # Build a collision-free analytic approach to the one palm/object transform already proven
        # feasible by the hand-space and lift oracles.  The random reset is saved while the fixed
        # reference snapshot is used only as a forward-kinematics calibration sample.
        initial_state = _capture_boundary(u)
        reference_state = {
            "joint_pos": torch.tensor(artifact["pregrasp"]["joint_pos"], device=dev).repeat(n, 1),
            "joint_vel": torch.zeros_like(initial_state["joint_vel"]),
            "dof_targets": torch.tensor(artifact["pregrasp"]["joint_pos"], device=dev).repeat(n, 1),
            "object_local_pos": torch.tensor(
                artifact["pregrasp"]["object_local_pos"], device=dev
            ).repeat(n, 1),
            "object_quat": torch.tensor(artifact["pregrasp"]["object_quat"], device=dev).repeat(n, 1),
            "object_velocity": torch.zeros_like(initial_state["object_velocity"]),
            "last_action": torch.zeros_like(initial_state["last_action"]),
        }
        _write_boundary(u, reference_state)
        u._compute_intermediate_values()
        reference_d = u._curr_fingertip_distances
        reference_other = torch.topk(
            reference_d[:, u._other_ee_idx], k=2, dim=1, largest=False
        ).values
        reference_grasp_dist = (
            reference_d[:, u._thumb_ee_idx] + reference_other.sum(dim=-1)
        ) / 3.0
        print(
            f"IK reference FK grasp_dist={float(reference_grasp_dist.median()):.4f}m "
            f"(artifact={float(artifact['pregrasp']['grasp_dist']):.4f}m)",
            flush=True,
        )
        reference_object_pos = reference_state["object_local_pos"] + u.scene.env_origins
        palm_in_object_pos, palm_in_object_quat = subtract_frame_transforms(
            reference_object_pos,
            reference_state["object_quat"],
            u.robot.data.body_pos_w[:, u.palm_idx],
            u.robot.data.body_quat_w[:, u.palm_idx],
        )
        palm_in_object_pos = palm_in_object_pos.detach().clone()
        palm_in_object_quat = palm_in_object_quat.detach().clone()

        _write_boundary(u, initial_state)
        u._contact_steps.zero_()
        u._lost_contact_steps.zero_()
        u._is_grasped.zero_()
        u._grasp_bonus_given.zero_()
        u._safe_grasp_steps.zero_()
        u._success_paid.zero_()
        u._success_steps.zero_()
        u._is_success.zero_()
        u._potential_initialized.zero_()
        u._compute_intermediate_values()
        obs = u._get_observations()

        target_reference_hand = reference_state["joint_pos"][:, u._hand_ids_t]
        snap_hand_action = fit_formal_pregrasp_action(target_reference_hand)
        snap_token = snap_hand_action[:, : u._n_tokens]
        hover_steps = max(1, int(round(args_cli.approach_steps * args_cli.approach_hover_fraction)))
        descent_steps = args_cli.approach_steps - hover_steps - args_cli.approach_settle_steps
        if descent_steps < 1:
            raise RuntimeError(
                "approach schedule leaves no descent steps; reduce hover fraction or settle steps"
            )
        for step in range(args_cli.approach_steps):
            target_pos, target_quat = combine_frame_transforms(
                u.object.data.root_pos_w,
                u.object.data.root_quat_w,
                palm_in_object_pos,
                palm_in_object_quat,
            )
            if step < hover_steps:
                desired_pos = target_pos.clone()
                desired_pos[:, 2] += args_cli.approach_hover_height
            elif step < hover_steps + descent_steps:
                descent = float(step - hover_steps + 1) / float(descent_steps)
                descent = descent * descent * (3.0 - 2.0 * descent)
                desired_pos = target_pos.clone()
                desired_pos[:, 2] += args_cli.approach_hover_height * (1.0 - descent)
            else:
                desired_pos = target_pos
            action = torch.zeros((n, cfg.action_space), device=dev)
            action[:, : u._n_arm] = dls_pose_action(desired_pos, target_quat)
            action[:, u._n_arm :] = snap_hand_action
            approach_obs.append(obs["policy"].clamp(-5.0, 5.0).detach().clone())
            approach_action.append(action.detach().clone())
            executed = action
            if rollout_actor is not None:
                learner_action = rollout_actor(obs["policy"]).clamp(-1.0, 1.0)
                teacher_mask = torch.rand((n,), device=dev) < args_cli.teacher_probability
                executed = torch.where(teacher_mask.unsqueeze(-1), action, learner_action)
                approach_teacher_rows += int(teacher_mask.sum())
                approach_total_rows += n
            obs, _, _, _, _ = step_without_reset(executed)

        u._compute_intermediate_values()
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
        pregrasp_score = torch.exp(-grasp_dist / 0.025) * alignment * palm_facing
        final_target_pos, final_target_quat = combine_frame_transforms(
            u.object.data.root_pos_w,
            u.object.data.root_quat_w,
            palm_in_object_pos,
            palm_in_object_quat,
        )
        final_pos_error, final_rot_error = compute_pose_error(
            u.robot.data.body_pos_w[:, u.palm_idx],
            u.robot.data.body_quat_w[:, u.palm_idx],
            final_target_pos,
            final_target_quat,
            rot_error_type="axis_angle",
        )
        best_step.fill_(args_cli.approach_steps - 1)
        snap_joint = u.robot.data.joint_pos.detach().clone()
        snap_obj_local = (u.object.data.root_pos_w - u.scene.env_origins).detach().clone()
        snap_obj_quat = u.object.data.root_quat_w.detach().clone()
        print(
            f"IK pregrasp score min/median/max={float(pregrasp_score.min()):.3f}/"
            f"{float(pregrasp_score.median()):.3f}/{float(pregrasp_score.max()):.3f}; "
            f"grasp_dist median={float(grasp_dist.median()):.4f}m; "
            f"pose_error median={float(final_pos_error.norm(dim=-1).median()):.4f}m/"
            f"{float(final_rot_error.norm(dim=-1).median()):.3f}rad; "
            f"align/palm median={float(alignment.median()):.3f}/{float(palm_facing.median()):.3f}",
            flush=True,
        )
    else:
        # The environment reset has already sampled physically captured, successful close-start
        # states.  Preserve their per-state formal hand action: IK datasets from different seeds
        # can use slightly different fitted pregrasp tokens/residuals.
        snap_joint = u.robot.data.joint_pos.detach().clone()
        snap_obj_local = (u.object.data.root_pos_w - u.scene.env_origins).detach().clone()
        snap_obj_quat = u.object.data.root_quat_w.detach().clone()
        snap_hand_action = u.actions[:, u._n_arm :].detach().clone()
        snap_token = snap_hand_action[:, : u._n_tokens]
        pregrasp_score = torch.zeros(n, device=dev)
        print(f"using {n} sampled curriculum close-start states", flush=True)

    # Restore each selected pregrasp with zero velocity. Sensor history is deliberately not read until
    # the first formal step; reward potentials are invalidated to prevent a reset-state jump.
    u.robot.write_joint_state_to_sim(snap_joint, torch.zeros_like(snap_joint), env_ids=all_ids)
    u.robot.set_joint_position_target(snap_joint, env_ids=all_ids)
    u.dof_targets.copy_(snap_joint)
    object_pose = torch.zeros((n, 7), device=dev)
    object_pose[:, :3] = snap_obj_local + u.scene.env_origins
    object_pose[:, 3:7] = snap_obj_quat
    u.object.write_root_pose_to_sim(object_pose, env_ids=all_ids)
    u.object.write_root_velocity_to_sim(torch.zeros((n, 6), device=dev), env_ids=all_ids)
    u.episode_length_buf.zero_()
    u._contact_steps.zero_()
    u._lost_contact_steps.zero_()
    u._is_grasped.zero_()
    u._grasp_bonus_given.zero_()
    u._safe_grasp_steps.zero_()
    u._success_paid.zero_()
    u._success_steps.zero_()
    u._is_success.zero_()
    u._potential_initialized.zero_()
    u.actions.zero_()
    u.actions[:, u._n_arm :] = snap_hand_action
    u.prev_actions.copy_(u.actions)
    u._compute_intermediate_values()
    obs = u._get_observations()
    close_start = _capture_boundary(u)

    latent = torch.tensor(hybrid["latent"], device=dev).repeat(n, 1)
    token = latent[:, : u._n_tokens]
    pregrasp_latent = snap_hand_action
    hand_lower = u.dof_lower[:, u._hand_ids_t]
    hand_upper = u.dof_upper[:, u._hand_ids_t]
    token_base = u.retarget.retarget_from_unit_action(token)[:, u._retarget2isaac]
    token_base = torch.maximum(torch.minimum(token_base, hand_upper), hand_lower)
    expected_target = torch.tensor(hybrid["target"], device=dev).repeat(n, 1)
    decoded_target, _ = apply_asymmetric_joint_residual(
        token_base,
        hand_lower,
        hand_upper,
        latent[:, u._n_tokens :],
        u._distal_hand_ids,
    )
    if float((decoded_target - expected_target).abs().max()) >= 1.0e-6:
        raise RuntimeError("hybrid artifact no longer decodes to its saved hand target")

    desired_distal = expected_target.index_select(1, u._distal_hand_ids).clone()
    servo_lower = desired_distal.clone()
    servo_upper = torch.minimum(
        servo_lower + args_cli.grip_servo_range,
        hand_upper.index_select(1, u._distal_hand_ids),
    )
    fingertip_to_distal = {
        "thumb_rota_link2": "thumb_joint2",
        "index_rota_link2": "index_joint2",
        "mid_link2": "middle_joint1",
        "ring_link2": "ring_joint1",
        "pinky_link2": "pinky_joint1",
    }
    distal_names = list(cfg.distal_residual_joint_names)
    force_to_residual = [distal_names.index(fingertip_to_distal[name]) for name in u.ee_names]

    oracle_obs: list[torch.Tensor] = []
    oracle_action: list[torch.Tensor] = []
    oracle_phase: list[int] = []
    force_peak = torch.zeros((n, len(u.ee_names)), device=dev)
    # Track the env's exact sustained-force safety contract across the whole episode so a demo
    # that the deployment/eval env would terminate (30 N for 10 frames or 60 N for 2 frames) is
    # rejected as a success, not merely reported via force_peak.  Force terminations are disabled
    # during collection, so this must be reconstructed here.
    hard_force_steps = torch.zeros(n, dtype=torch.long, device=dev)
    overforce_steps = torch.zeros(n, dtype=torch.long, device=dev)
    force_unsafe = torch.zeros(n, dtype=torch.bool, device=dev)
    executed_teacher_rows = approach_teacher_rows
    executed_total_rows = approach_total_rows

    def hand_action() -> torch.Tensor:
        desired = token_base.clone()
        desired.index_copy_(1, u._distal_hand_ids, desired_distal)
        residual = invert_asymmetric_joint_residual(
            desired, token_base, hand_lower, hand_upper, u._distal_hand_ids
        )
        return torch.cat((token, residual), dim=-1)

    def update_grip(force: torch.Tensor) -> None:
        delta = torch.zeros_like(desired_distal)
        for force_index, residual_index in enumerate(force_to_residual):
            delta[:, residual_index] = args_cli.grip_servo_step * (
                (force[:, force_index] < args_cli.grip_force_target).float()
                - (force[:, force_index] > args_cli.grip_force_limit).float()
            )
        # Integrate from the *observable previous action*, not from hidden teacher state.  Decoding
        # u.actions exactly recovers the old oracle servo on teacher rollouts while learner-executed
        # actions naturally become the feedback starting point in DAgger states.
        previous_hand_action = u.actions[:, u._n_arm :]
        previous_token_target = u.retarget.retarget_from_unit_action(
            previous_hand_action[:, : u._n_tokens]
        )[:, u._retarget2isaac]
        previous_target, _ = apply_asymmetric_joint_residual(
            previous_token_target,
            hand_lower,
            hand_upper,
            previous_hand_action[:, u._n_tokens :],
            u._distal_hand_ids,
        )
        previous_distal = previous_target.index_select(1, u._distal_hand_ids)
        desired_distal.copy_(
            torch.maximum(torch.minimum(previous_distal + delta, servo_upper), servo_lower)
        )

    def step_record(action: torch.Tensor, phase: int) -> dict[str, torch.Tensor]:
        nonlocal obs, executed_teacher_rows, executed_total_rows
        oracle_obs.append(obs["policy"].clamp(-5.0, 5.0).detach().clone())
        # The stored action is always the corrective oracle label.  With a rollout actor, only the
        # executed action is mixed; this is standard DAgger and deliberately visits learner-induced
        # off-expert states without teaching the learner to repeat its own error.
        oracle_action.append(action.detach().clone())
        oracle_phase.append(phase)
        executed = action
        if rollout_actor is not None:
            learner_action = rollout_actor(obs["policy"]).clamp(-1.0, 1.0)
            teacher_mask = torch.rand((n,), device=dev) < args_cli.teacher_probability
            executed = torch.where(teacher_mask.unsqueeze(-1), action, learner_action)
            executed_teacher_rows += int(teacher_mask.sum())
            executed_total_rows += n
        obs, _, _, _, _ = step_without_reset(executed)
        signals = u._compute_grasp_signals()
        force = signals["force_magnitude"]
        force_peak.copy_(torch.maximum(force_peak, force))
        force_max = force.max(dim=-1).values
        hard_force_steps.copy_(
            torch.where(
                force_max > cfg.tactile_hard_force_limit,
                hard_force_steps + 1,
                torch.zeros_like(hard_force_steps),
            )
        )
        overforce_steps.copy_(
            torch.where(
                force_max > cfg.tactile_terminate_force_limit,
                overforce_steps + 1,
                torch.zeros_like(overforce_steps),
            )
        )
        force_unsafe.logical_or_(
            (hard_force_steps >= cfg.tactile_hard_terminate_steps)
            | (overforce_steps >= cfg.tactile_terminate_steps)
        )
        return {
            "signals": signals,
            "force": force,
            "clearance": u._object_true_min_z() - u._table_surface_z,
        }

    print(f"formal close from {args_cli.pregrasp_source} pregrasps", flush=True)
    close_state = None
    for step in range(args_cli.close_steps + args_cli.close_hold_steps):
        action = torch.zeros((n, cfg.action_space), device=dev)
        if step < args_cli.close_steps:
            x = float(step + 1) / float(args_cli.close_steps)
            blend = x * x * (3.0 - 2.0 * x)
            action[:, u._n_arm :] = pregrasp_latent + blend * (latent - pregrasp_latent)
        else:
            action[:, u._n_arm :] = hand_action()
        close_state = step_record(action, PHASE_CLOSE)
        if step >= args_cli.close_steps:
            update_grip(close_state["force"])
    close_pass = u._is_grasped & (close_state["signals"]["grasp_quality"] >= cfg.grasp_quality_high)
    print(f"close pass={int(close_pass.sum())}/{n}", flush=True)

    lift_start = _capture_boundary(u)
    palm_body_start = u.robot.data.body_pos_w[:, u.palm_idx].detach().clone()
    palm_quat_target = u.robot.data.body_quat_w[:, u.palm_idx].detach().clone()
    palm_center_start_z = u.palm_center_w[:, 2].detach().clone()
    clearance_start = (u._object_true_min_z() - u._table_surface_z).detach().clone()
    def dls_action(height: float) -> torch.Tensor:
        desired_pos = palm_body_start.clone()
        desired_pos[:, 2] += height
        action = torch.zeros((n, cfg.action_space), device=dev)
        action[:, : u._n_arm] = dls_pose_action(desired_pos, palm_quat_target)
        action[:, u._n_arm :] = hand_action()
        return action

    micro_state = close_state
    for step in range(args_cli.micro_steps):
        height = args_cli.micro_height * float(step + 1) / args_cli.micro_steps
        micro_state = step_record(dls_action(height), PHASE_MICRO)
        update_grip(micro_state["force"])
    for _ in range(args_cli.micro_hold_steps):
        micro_state = step_record(dls_action(args_cli.micro_height), PHASE_MICRO)
        update_grip(micro_state["force"])
    micro_end = _capture_boundary(u)
    micro_pass = (
        (micro_state["clearance"] >= 0.015)
        & u._is_grasped
        & (micro_state["signals"]["grasp_quality"] >= cfg.grasp_quality_low)
    )
    print(
        f"micro pass={int(micro_pass.sum())}/{n}; clearance="
        f"{_summary(micro_state['clearance'])}",
        flush=True,
    )

    lift_state = micro_state
    mid_lift = None
    for step in range(args_cli.lift_steps):
        height = args_cli.micro_height + (args_cli.target_height - args_cli.micro_height) * (
            float(step + 1) / args_cli.lift_steps
        )
        lift_state = step_record(dls_action(height), PHASE_LIFT)
        update_grip(lift_state["force"])
        if step + 1 == max(1, args_cli.lift_steps // 2):
            mid_lift = _capture_boundary(u)
    if mid_lift is None:
        raise RuntimeError("mid-lift boundary was not captured")
    settle_start = _capture_boundary(u)

    stable_count = torch.zeros(n, dtype=torch.long, device=dev)
    for _ in range(args_cli.settle_steps):
        lift_state = step_record(dls_action(args_cli.target_height), PHASE_SETTLE)
        update_grip(lift_state["force"])
        signals = lift_state["signals"]
        slow = (
            (u.object.data.root_com_lin_vel_w.norm(dim=-1) < cfg.success_max_obj_lin_speed)
            & (u.object.data.root_com_ang_vel_w.norm(dim=-1) < cfg.success_max_obj_ang_speed)
        )
        stable = (
            (lift_state["clearance"] >= cfg.lift_success_height)
            & u._is_grasped
            & (signals["grasp_quality"] >= cfg.grasp_quality_high)
            & (signals["force_magnitude"].max(dim=-1).values <= cfg.grasp_bonus_max_force)
            & slow
        )
        stable_count = torch.where(stable, stable_count + 1, torch.zeros_like(stable_count))

    # A marginal close may cross the strict Schmitt threshold during the deliberately slow
    # micro-lift.  Judge demonstrations by the task's final fifteen-step criterion, and also
    # reject any episode that violated the env's sustained-force safety at any point (the
    # deployment/eval env would have terminated it).
    success = (stable_count >= 15) & ~force_unsafe
    if int(force_unsafe.sum()):
        print(
            f"[safety] {int(force_unsafe.sum())}/{n} episodes hit the env's sustained-force cutoff "
            f"and are excluded from successes",
            flush=True,
        )
    success_ids = success.nonzero(as_tuple=False).squeeze(-1)
    selected_ids = (
        torch.arange(n, device=dev, dtype=torch.long)
        if args_cli.retain_all_episodes
        else success_ids
    )
    final_clearance = lift_state["clearance"]
    final_signals = lift_state["signals"]
    palm_rise = u.palm_center_w[:, 2] - palm_center_start_z
    clearance_gain = final_clearance - clearance_start
    print(
        f"20cm stable success={int(success.sum())}/{n}; clearance min/median/max="
        f"{float(final_clearance.min()):.3f}/{float(final_clearance.median()):.3f}/"
        f"{float(final_clearance.max()):.3f}m",
        flush=True,
    )
    print(
        f"final latch={int(u._is_grasped.sum())}/{n} q={_summary(final_signals['grasp_quality'])} "
        f"force_max={_summary(final_signals['force_magnitude'].max(dim=-1).values)} "
        f"stable_count={_summary(stable_count)}",
        flush=True,
    )
    if success_ids.numel() == 0 and not args_cli.retain_all_episodes:
        env.close()
        raise RuntimeError("formal-action oracle produced no successful demonstrations")
    required_successes = max(1, int(args_cli.min_success_fraction * n + 0.999999))
    if success_ids.numel() < required_successes and not args_cli.retain_all_episodes:
        env.close()
        raise RuntimeError(
            f"formal-action oracle passed only {success_ids.numel()}/{n}; "
            f"required at least {required_successes}/{n}"
        )

    oracle_obs_t = torch.stack(oracle_obs)
    oracle_action_t = torch.stack(oracle_action)
    oracle_phase_t = torch.tensor(oracle_phase, dtype=torch.uint8, device=dev)
    approach_obs_t = torch.stack(approach_obs) if approach_obs else None
    approach_action_t = torch.stack(approach_action) if approach_action else None

    episode_obs = []
    episode_action = []
    episode_phase = []
    episode_id = []
    episode_step = []
    # Per-episode approach length marks the teleport splice: rows [0:approach_length] come from
    # the policy/IK approach, then the state is re-teleported (velocities zeroed, object rewound)
    # before the close rows.  That single boundary is not a physical transition; sequence/history
    # consumers must not cross it.  0 means no approach segment (already-in-hand pregrasp).
    episode_approach_length = []
    offsets = [0]
    for episode, env_id in enumerate(selected_ids.tolist()):
        obs_parts = []
        action_parts = []
        phase_parts = []
        approach_length = 0
        if approach_obs_t is not None:
            end = int(best_step[env_id]) + 1
            approach_length = end
            obs_parts.append(approach_obs_t[:end, env_id])
            action_parts.append(approach_action_t[:end, env_id])
            phase_parts.append(torch.full((end,), PHASE_APPROACH, dtype=torch.uint8, device=dev))
        obs_parts.append(oracle_obs_t[:, env_id])
        action_parts.append(oracle_action_t[:, env_id])
        phase_parts.append(oracle_phase_t)
        ep_obs = torch.cat(obs_parts)
        ep_action = torch.cat(action_parts)
        ep_phase = torch.cat(phase_parts)
        length = ep_obs.shape[0]
        episode_obs.append(ep_obs)
        episode_action.append(ep_action)
        episode_phase.append(ep_phase)
        episode_id.append(torch.full((length,), episode, dtype=torch.int32, device=dev))
        episode_step.append(torch.arange(length, dtype=torch.int16, device=dev))
        episode_approach_length.append(approach_length)
        offsets.append(offsets[-1] + length)

    checkpoint_path = Path(args_cli.checkpoint) if args_cli.checkpoint else None
    dataset = {
        "obs": torch.cat(episode_obs).float().cpu(),
        "action": torch.cat(episode_action).float().cpu(),
        "phase": torch.cat(episode_phase).cpu(),
        "episode_id": torch.cat(episode_id).cpu(),
        "step": torch.cat(episode_step).cpu(),
        "episode_offsets": torch.tensor(offsets, dtype=torch.int64),
        # Per-episode outcome, aligned with episode_id order.  With --retain_all_episodes the
        # failed episodes are otherwise indistinguishable inside the .pt.
        "episode_success": success[selected_ids].cpu(),
        "episode_force_unsafe": force_unsafe[selected_ids].cpu(),
        "episode_stable_count": stable_count[selected_ids].cpu(),
        "episode_approach_length": torch.tensor(episode_approach_length, dtype=torch.int64),
        "boundaries": {
            "close_start": {key: value[selected_ids].cpu() for key, value in close_start.items()},
            "lift_start": {key: value[selected_ids].cpu() for key, value in lift_start.items()},
            "micro_end": {key: value[selected_ids].cpu() for key, value in micro_end.items()},
            "mid_lift": {key: value[selected_ids].cpu() for key, value in mid_lift.items()},
            "settle_start": {key: value[selected_ids].cpu() for key, value in settle_start.items()},
        },
        "meta": {
            "format_version": 1,
            "action_layout": "arm_delta7|crossdex_token9|distal_residual5",
            "observation_layout": "legacy_prefix87|distal_action5|grasp_transport23",
            "phase_names": ["approach", "close", "micro", "lift", "settle"],
            "pregrasp_source": args_cli.pregrasp_source,
            "seed": args_cli.seed,
            "feasibility_json": str(artifact_path.resolve()),
            "feasibility_sha256": _sha256(artifact_path),
            "checkpoint": str(checkpoint_path.resolve()) if checkpoint_path else None,
            "checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path else None,
            "rollout_checkpoint": str(Path(args_cli.rollout_checkpoint).resolve())
            if args_cli.rollout_checkpoint
            else None,
            "teacher_probability": args_cli.teacher_probability,
            "retain_all_episodes": args_cli.retain_all_episodes,
            "oracle_steps_without_approach": len(oracle_phase),
            "oracle_schedule": {
                "close_steps": args_cli.close_steps,
                "close_hold_steps": args_cli.close_hold_steps,
                "micro_steps": args_cli.micro_steps,
                "micro_hold_steps": args_cli.micro_hold_steps,
                "lift_steps": args_cli.lift_steps,
                "settle_steps": args_cli.settle_steps,
                "micro_height": args_cli.micro_height,
                "target_height": args_cli.target_height,
            },
        },
    }
    if not torch.isfinite(dataset["obs"]).all() or not torch.isfinite(dataset["action"]).all():
        env.close()
        raise RuntimeError("refusing to save: dataset contains non-finite observations or actions")
    for name, boundary in dataset["boundaries"].items():
        for key, value in boundary.items():
            if torch.is_floating_point(value) and not torch.isfinite(value).all():
                env.close()
                raise RuntimeError(f"refusing to save: boundary {name}.{key} contains non-finite values")
    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)

    metrics = {
        "num_envs": n,
        "pregrasp_source": args_cli.pregrasp_source,
        "close_pass_count": int(close_pass.sum()),
        "micro_pass_count": int(micro_pass.sum()),
        "success_count": int(success.sum()),
        "retained_episode_count": int(selected_ids.numel()),
        "executed_teacher_fraction": (
            executed_teacher_rows / executed_total_rows if executed_total_rows else 1.0
        ),
        "transitions": int(dataset["obs"].shape[0]),
        "final_clearance": final_clearance.cpu().tolist(),
        "palm_rise_minus_clearance_gain": (palm_rise - clearance_gain).cpu().tolist(),
        "force_order": list(u.ee_names),
        "force_peak_per_finger": force_peak.max(dim=0).values.cpu().tolist(),
        "stable_count": stable_count.cpu().tolist(),
        "final_grasp_quality": final_signals["grasp_quality"].cpu().tolist(),
        "dataset": str(output_path.resolve()),
    }
    metrics_path = Path(args_cli.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {output_path} ({dataset['obs'].shape[0]} transitions)", flush=True)
    print(f"wrote {metrics_path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
