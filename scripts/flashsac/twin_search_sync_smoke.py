#!/usr/bin/env python3
"""Run a fail-closed full-task SEARCH cross-slot A/A diagnostic.

Only the even-slot observations are evaluated by the frozen rl_games SEARCH
actor.  Each resulting 21-D action is copied to its adjacent odd-slot twin.  The run
fails immediately on an observation, action, controller-target, reward, done,
or task-authored terminal-outcome mismatch.  Passing is required before any
cross-slot result can be treated as paired evidence; failure is an expected
falsification outcome because the hammer-table manifold cannot be copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import traceback
from typing import Any, Mapping

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
parser.add_argument("--num_envs", "--num-envs", dest="num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--env_spacing",
    type=float,
    default=2.5,
    help="world-space clone spacing; local task geometry is unchanged",
)
parser.add_argument(
    "--enhanced_determinism",
    action="store_true",
    help="enable PhysX enhanced determinism for paired-evidence diagnostics",
)
parser.add_argument(
    "--record_continuous_errors",
    action="store_true",
    help=(
        "diagnostic-only mode: record continuous obs/reward/target threshold "
        "violations instead of stopping; discrete done/outcome mismatches still fail"
    ),
)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 2 or args_cli.num_envs % 2 != 0:
    parser.error("--num_envs must be at least two and even")
if args_cli.steps < 1:
    parser.error("--steps must be positive")
if not math.isfinite(args_cli.env_spacing) or args_cli.env_spacing <= 0.0:
    parser.error("--env_spacing must be finite and positive")
if not args_cli.checkpoint.is_file() or args_cli.checkpoint.is_symlink():
    parser.error("--checkpoint must be a regular, non-symlink file")
if args_cli.output.exists() or args_cli.output.is_symlink():
    parser.error("--output already exists; twin evidence is immutable")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from adapter import make_pick_tool_env
from evaluate import (
    FULL_TASK_MODE,
    _load_diagnostic_approach_actor,
    validate_terminal_events,
)
from pick_tool_twin_state import (
    ACTION_DIM,
    OBSERVATION_DIM,
    READINESS_STRATUM_NAMES,
    copy_fresh_reset_to_twins,
    make_twin_pair_layout,
    publish_json_no_clobber,
    replicate_source_actions,
    update_public_readiness,
)


IMMEDIATE_OBSERVATION_ATOL = 1.0e-5
ACTION_ATOL = 1.0e-7
TARGET_ATOL = 1.0e-6
STEP_OBSERVATION_ATOL = 1.0e-4
REWARD_ATOL = 1.0e-4
CONTACT_RESET_ATOL = 1.0e-6
TIMESTAMP_ATOL = 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair_differences(
    value: torch.Tensor,
    source: torch.Tensor,
    destination: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    selected_source = value.index_select(0, source)[active]
    selected_destination = value.index_select(0, destination)[active]
    if value.dtype is torch.bool or not (
        value.dtype.is_floating_point or value.dtype.is_complex
    ):
        mismatch = selected_source != selected_destination
        if mismatch.ndim > 1:
            mismatch = mismatch.flatten(1).any(dim=1)
        return mismatch.to(dtype=torch.float32)
    difference = (selected_source - selected_destination).abs()
    if difference.ndim > 1:
        difference = difference.flatten(1).max(dim=1).values
    return difference


def _pair_max(
    value: torch.Tensor,
    source: torch.Tensor,
    destination: torch.Tensor,
    active: torch.Tensor,
) -> float:
    difference = _pair_differences(value, source, destination, active)
    return float(difference.max()) if difference.numel() else 0.0


def _pair_max_detail(
    value: torch.Tensor,
    source: torch.Tensor,
    destination: torch.Tensor,
    active: torch.Tensor,
) -> dict[str, Any]:
    """Locate the exact active pair/component responsible for a maximum error."""

    active_ranks = active.nonzero(as_tuple=False).flatten()
    selected_source = value.index_select(0, source)[active]
    selected_destination = value.index_select(0, destination)[active]
    if selected_source.numel() == 0:
        return {"abs_error": 0.0}
    difference = (selected_source - selected_destination).abs()
    flat = difference.reshape(difference.shape[0], -1)
    flat_index = int(flat.argmax())
    active_index = flat_index // flat.shape[1]
    component = flat_index % flat.shape[1]
    pair_rank = int(active_ranks[active_index])
    source_flat = selected_source.reshape(selected_source.shape[0], -1)
    destination_flat = selected_destination.reshape(selected_destination.shape[0], -1)
    return {
        "abs_error": float(flat[active_index, component]),
        "pair_rank": pair_rank,
        "source_env": int(source[pair_rank]),
        "destination_env": int(destination[pair_rank]),
        "flat_component": component,
        "source_value": float(source_flat[active_index, component]),
        "destination_value": float(destination_flat[active_index, component]),
    }


def _pair_mismatch_mask(
    value: torch.Tensor,
    source: torch.Tensor,
    destination: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    selected_source = value.index_select(0, source)[active]
    selected_destination = value.index_select(0, destination)[active]
    mismatch = selected_source != selected_destination
    if mismatch.ndim > 1:
        mismatch = mismatch.flatten(1).any(dim=1)
    return mismatch


def _fail_if_above(name: str, value: float, threshold: float, step: int) -> None:
    if not math.isfinite(value) or value > threshold:
        raise RuntimeError(
            f"step {step}: {name}={value:g} exceeded fixed threshold {threshold:g}"
        )


def _terminal_pair_mismatches(
    events: Mapping[str, torch.Tensor],
    source: torch.Tensor,
    destination: torch.Tensor,
    active: torch.Tensor,
) -> tuple[int, dict[str, int]]:
    active_pairs = int(active.sum())
    union = torch.zeros(active_pairs, dtype=torch.bool, device=source.device)
    per_outcome: dict[str, int] = {}
    for name, value in sorted(events.items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"validated terminal outcome {name!r} is not a tensor")
        mismatch = _pair_mismatch_mask(value, source, destination, active)
        count = int(mismatch.sum())
        per_outcome[name] = count
        union |= mismatch
    return int(union.sum()), per_outcome


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    device_string = str(args.device or "cuda:0")
    device = torch.device(device_string)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"PickTool twin smoke requires CUDA Isaac physics, got {device_string}")

    checkpoint = args.checkpoint.resolve()
    actor = _load_diagnostic_approach_actor(checkpoint, device=device)
    actor.eval()
    cfg_overrides = {
        "scene.env_spacing": float(args.env_spacing),
        "sim.physx.enable_enhanced_determinism": bool(args.enhanced_determinism),
    }
    env = make_pick_tool_env(
        num_envs=args.num_envs,
        device=device_string,
        seed=args.seed,
        cfg_overrides=cfg_overrides,
        strict=True,
        validate_finite=True,
    )
    layout = make_twin_pair_layout(args.num_envs, env.device)
    task = env.unwrapped
    steps: list[dict[str, Any]] = []
    readiness_counts = {name: 0 for name in READINESS_STRATUM_NAMES}
    ready_count = torch.zeros(layout.pair_count, dtype=torch.long, device=env.device)
    fork_used = torch.zeros(layout.pair_count, dtype=torch.bool, device=env.device)
    active = torch.ones(layout.pair_count, dtype=torch.bool, device=env.device)
    summary = {
        "immediate_observation_abs_error": 0.0,
        "immediate_action_abs_error": 0.0,
        "immediate_dof_target_abs_error": 0.0,
        "action_abs_error_max": 0.0,
        "dof_target_abs_error_max": 0.0,
        "step_observation_abs_error_max": 0.0,
        "reward_abs_error_max": 0.0,
        "terminated_mismatch_pairs_total": 0,
        "truncated_mismatch_pairs_total": 0,
        "terminal_outcome_mismatch_pairs_total": 0,
        "terminal_outcome_mismatch_by_name": {},
        "observation_threshold_violation_steps": 0,
        "reward_threshold_violation_steps": 0,
        "dof_target_threshold_violation_steps": 0,
    }
    try:
        reset_observation, _ = env.reset(
            seed=args.seed, randomize_episode_lengths=False
        )
        if reset_observation.shape != (args.num_envs, OBSERVATION_DIM):
            raise RuntimeError("adapter reset violated the 115-D observation contract")
        observation, copy_report = copy_fresh_reset_to_twins(
            task,
            layout,
            reset_zero_atol=TARGET_ATOL,
            contact_force_atol=CONTACT_RESET_ATOL,
            timestamp_atol=TIMESTAMP_ATOL,
        )
        # The adapter owns the public observation passed to a state-conditioned
        # action boundary.  Synchronize it to the zero-time rebuilt observation.
        env._current_policy_observation = observation.detach().clone()
        env._last_executed_policy_action = None

        immediate_obs = _pair_max(
            observation, layout.source, layout.destination, active
        )
        immediate_action = _pair_max(
            task.actions, layout.source, layout.destination, active
        )
        immediate_dof = max(
            _pair_max(task.dof_targets, layout.source, layout.destination, active),
            _pair_max(
                task.robot.data.joint_pos_target,
                layout.source,
                layout.destination,
                active,
            ),
        )
        summary["immediate_observation_abs_error"] = immediate_obs
        summary["immediate_action_abs_error"] = immediate_action
        summary["immediate_dof_target_abs_error"] = immediate_dof
        _fail_if_above(
            "immediate observation pair error",
            immediate_obs,
            IMMEDIATE_OBSERVATION_ATOL,
            0,
        )
        _fail_if_above("immediate action pair error", immediate_action, ACTION_ATOL, 0)
        _fail_if_above("immediate target pair error", immediate_dof, TARGET_ATOL, 0)

        cfg = task.cfg
        for step_index in range(1, args.steps + 1):
            if not bool(active.any()):
                break
            active_before = active.clone()
            active_source_ids = layout.source[active_before]
            public_source = observation.index_select(0, layout.source)
            readiness = update_public_readiness(public_source, ready_count, fork_used)
            ready_count = torch.where(
                active_before, readiness.ready_count, ready_count
            )
            fork_used = torch.where(active_before, readiness.fork_used, fork_used)
            for stratum_index, name in enumerate(READINESS_STRATUM_NAMES):
                readiness_counts[name] += int(
                    (
                        active_before
                        & readiness.trigger
                        & (readiness.stratum == stratum_index)
                    ).sum()
                )

            source_actions = observation.new_zeros((layout.pair_count, ACTION_DIM))
            actor_input = observation.index_select(0, active_source_ids)
            actor_output = actor(actor_input)
            if actor_output.shape != (int(active_before.sum()), ACTION_DIM):
                raise RuntimeError("SEARCH actor violated the 115/21 contract")
            if not bool(torch.isfinite(actor_output).all()):
                raise FloatingPointError("SEARCH actor produced NaN or infinity")
            source_actions[active_before] = actor_output.clamp(-1.0, 1.0)
            action = replicate_source_actions(source_actions, layout)
            action_error = _pair_max(
                action, layout.source, layout.destination, active_before
            )
            _fail_if_above("action pair error", action_error, ACTION_ATOL, step_index)

            next_observation, reward, terminated, truncated, info = env.step(action)
            events = validate_terminal_events(
                info,
                terminated,
                truncated,
                task_mode=FULL_TASK_MODE,
                close_option_confirm_steps=int(cfg.close_option_confirm_steps),
                power_required_other_contacts=int(cfg.power_grasp_required_other_contacts),
                power_grasp_quality_threshold=float(cfg.power_grasp_quality_high),
                close_option_min_hold_quality=float(cfg.close_option_min_hold_quality),
                close_option_safe_force_limit=float(cfg.grasp_bonus_max_force),
            )
            transition_observation = info.get("transition_next_observation")
            if not isinstance(transition_observation, torch.Tensor) or transition_observation.shape != (
                args.num_envs,
                OBSERVATION_DIM,
            ):
                raise RuntimeError("adapter omitted the terminal-safe transition observation")

            observation_error = _pair_max(
                transition_observation,
                layout.source,
                layout.destination,
                active_before,
            )
            observation_error_detail = _pair_max_detail(
                transition_observation,
                layout.source,
                layout.destination,
                active_before,
            )
            reward_error = _pair_max(
                reward, layout.source, layout.destination, active_before
            )
            dof_target_error = max(
                _pair_max(
                    task.dof_targets,
                    layout.source,
                    layout.destination,
                    active_before,
                ),
                _pair_max(
                    task.robot.data.joint_pos_target,
                    layout.source,
                    layout.destination,
                    active_before,
                ),
            )
            terminated_mismatch = _pair_mismatch_mask(
                terminated,
                layout.source,
                layout.destination,
                active_before,
            )
            truncated_mismatch = _pair_mismatch_mask(
                truncated,
                layout.source,
                layout.destination,
                active_before,
            )
            terminal_mismatch_count, terminal_mismatch_by_name = (
                _terminal_pair_mismatches(
                    events,
                    layout.source,
                    layout.destination,
                    active_before,
                )
            )
            terminated_mismatch_count = int(terminated_mismatch.sum())
            truncated_mismatch_count = int(truncated_mismatch.sum())

            observation_threshold_violated = (
                not math.isfinite(observation_error)
                or observation_error > STEP_OBSERVATION_ATOL
            )
            if observation_threshold_violated and not args.record_continuous_errors:
                physical_error_detail = {
                    "joint_pos": _pair_max_detail(
                        task.robot.data.joint_pos,
                        layout.source,
                        layout.destination,
                        active_before,
                    ),
                    "joint_vel": _pair_max_detail(
                        task.robot.data.joint_vel,
                        layout.source,
                        layout.destination,
                        active_before,
                    ),
                    "object_local_position": _pair_max_detail(
                        task.object.data.root_link_pos_w - task.scene.env_origins,
                        layout.source,
                        layout.destination,
                        active_before,
                    ),
                    "object_quaternion": _pair_max_detail(
                        task.object.data.root_link_quat_w,
                        layout.source,
                        layout.destination,
                        active_before,
                    ),
                    "object_com_linear_velocity": _pair_max_detail(
                        task.object.data.root_com_lin_vel_w,
                        layout.source,
                        layout.destination,
                        active_before,
                    ),
                    "object_com_angular_velocity": _pair_max_detail(
                        task.object.data.root_com_ang_vel_w,
                        layout.source,
                        layout.destination,
                        active_before,
                    ),
                }
                raise RuntimeError(
                    f"step {step_index}: step observation pair error="
                    f"{observation_error:g} exceeded fixed threshold "
                    f"{STEP_OBSERVATION_ATOL:g}; observation_detail="
                    f"{observation_error_detail}; physical_detail="
                    f"{physical_error_detail}"
                )
            reward_threshold_violated = (
                not math.isfinite(reward_error) or reward_error > REWARD_ATOL
            )
            target_threshold_violated = (
                not math.isfinite(dof_target_error)
                or dof_target_error > TARGET_ATOL
            )
            if reward_threshold_violated and not args.record_continuous_errors:
                _fail_if_above(
                    "reward pair error", reward_error, REWARD_ATOL, step_index
                )
            if target_threshold_violated and not args.record_continuous_errors:
                _fail_if_above(
                    "dof target pair error", dof_target_error, TARGET_ATOL, step_index
                )
            if terminated_mismatch_count or truncated_mismatch_count:
                raise RuntimeError(
                    f"step {step_index}: done mismatch: terminated="
                    f"{terminated_mismatch_count}, truncated={truncated_mismatch_count}"
                )
            if terminal_mismatch_count:
                raise RuntimeError(
                    f"step {step_index}: task-authored terminal outcome mismatch: "
                    f"{terminal_mismatch_by_name}"
                )

            done_source = (
                terminated.index_select(0, layout.source)
                | truncated.index_select(0, layout.source)
            )
            completed_now = active_before & done_source
            active = active_before & (~done_source)
            row = {
                "step": step_index,
                "active_pairs_before": int(active_before.sum()),
                "active_pairs_after": int(active.sum()),
                "completed_pairs": int(completed_now.sum()),
                "action_abs_error": action_error,
                "dof_target_abs_error": dof_target_error,
                "observation_abs_error": observation_error,
                "reward_abs_error": reward_error,
                "terminated_mismatch_pairs": terminated_mismatch_count,
                "truncated_mismatch_pairs": truncated_mismatch_count,
                "terminal_outcome_mismatch_pairs": terminal_mismatch_count,
                "terminal_outcome_mismatch_by_name": terminal_mismatch_by_name,
                "observation_threshold_violated": observation_threshold_violated,
                "reward_threshold_violated": reward_threshold_violated,
                "dof_target_threshold_violated": target_threshold_violated,
                "readiness_trigger_count": int(
                    (active_before & readiness.trigger).sum()
                ),
            }
            steps.append(row)
            summary["action_abs_error_max"] = max(
                float(summary["action_abs_error_max"]), action_error
            )
            summary["dof_target_abs_error_max"] = max(
                float(summary["dof_target_abs_error_max"]), dof_target_error
            )
            summary["step_observation_abs_error_max"] = max(
                float(summary["step_observation_abs_error_max"]), observation_error
            )
            summary["reward_abs_error_max"] = max(
                float(summary["reward_abs_error_max"]), reward_error
            )
            summary["terminated_mismatch_pairs_total"] += terminated_mismatch_count
            summary["truncated_mismatch_pairs_total"] += truncated_mismatch_count
            summary["terminal_outcome_mismatch_pairs_total"] += terminal_mismatch_count
            summary["observation_threshold_violation_steps"] += int(
                observation_threshold_violated
            )
            summary["reward_threshold_violation_steps"] += int(
                reward_threshold_violated
            )
            summary["dof_target_threshold_violation_steps"] += int(
                target_threshold_violated
            )
            aggregate_outcomes = summary["terminal_outcome_mismatch_by_name"]
            for name, count in terminal_mismatch_by_name.items():
                aggregate_outcomes[name] = int(aggregate_outcomes.get(name, 0)) + count
            observation = next_observation

        full_rollout_requested = args.steps >= int(env.max_episode_steps)
        if full_rollout_requested and bool(active.any()):
            raise RuntimeError(
                "full null rollout exhausted the native horizon with active twin pairs"
            )
        evidence_eligible = not any(
            int(summary[name])
            for name in (
                "observation_threshold_violation_steps",
                "reward_threshold_violation_steps",
                "dof_target_threshold_violation_steps",
                "terminated_mismatch_pairs_total",
                "truncated_mismatch_pairs_total",
                "terminal_outcome_mismatch_pairs_total",
            )
        )
        return {
            "status": (
                "passed"
                if evidence_eligible
                else "diagnostic_completed_with_continuous_mismatch"
            ),
            "evidence_eligible": evidence_eligible,
            "record_continuous_errors": bool(args.record_continuous_errors),
            "kind": "pick_tool_twin_search_sync_smoke_v1",
            "task_mode": FULL_TASK_MODE,
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "seed": int(args.seed),
            "num_envs": int(args.num_envs),
            "pair_count": layout.pair_count,
            "env_spacing": float(args.env_spacing),
            "enhanced_determinism": bool(args.enhanced_determinism),
            "pairing": "adjacent_even_source_to_odd_destination_v1",
            "steps_requested": int(args.steps),
            "steps_executed": len(steps),
            "native_max_episode_steps": int(env.max_episode_steps),
            "full_rollout_requested": full_rollout_requested,
            "full_rollout_completed": not bool(active.any()),
            "active_pairs_final": int(active.sum()),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "policy": "frozen_rlgames_search_even_source_only_clamped_mean_v1",
            "thresholds": {
                "fresh_reset_contact_force_abs_max": CONTACT_RESET_ATOL,
                "contact_timestamp_pair_abs_error": TIMESTAMP_ATOL,
                "immediate_observation_abs_error": IMMEDIATE_OBSERVATION_ATOL,
                "action_abs_error": ACTION_ATOL,
                "dof_target_abs_error": TARGET_ATOL,
                "step_observation_abs_error": STEP_OBSERVATION_ATOL,
                "reward_abs_error": REWARD_ATOL,
                "done_mismatch_pairs": 0,
                "terminal_outcome_mismatch_pairs": 0,
            },
            "reset_copy": copy_report,
            "public_readiness": {
                "contract": (
                    "g=min(obs[96],second_largest(obs[92:96])); "
                    "eligible=(obs[106]==0 and g>=0.10); consecutive_cap4; sticky_once"
                ),
                "trigger_count_by_stratum": readiness_counts,
                "trigger_count": sum(readiness_counts.values()),
                "fork_used_final": int(fork_used.sum()),
            },
            "summary": summary,
            "per_step": steps,
        }
    finally:
        env.close()


def main() -> None:
    output = Path(os.path.abspath(os.fspath(args_cli.output)))
    base = {
        "kind": "pick_tool_twin_search_sync_smoke_v1",
        "seed": int(args_cli.seed),
        "num_envs": int(args_cli.num_envs),
        "env_spacing": float(args_cli.env_spacing),
        "enhanced_determinism": bool(args_cli.enhanced_determinism),
        "record_continuous_errors": bool(args_cli.record_continuous_errors),
        "steps_requested": int(args_cli.steps),
        "checkpoint": str(args_cli.checkpoint.absolute()),
    }
    try:
        report = run(args_cli)
    except Exception as error:
        failure = {
            **base,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        publish_json_no_clobber(failure, output)
        raise
    else:
        # Strict serialization before publication also rejects hidden NaN/Inf.
        json.dumps(report, sort_keys=True, allow_nan=False)
        publish_json_no_clobber(report, output)
        print(
            "[twin-search-sync] "
            f"status={report['status']} pairs={report['pair_count']} "
            f"steps={report['steps_executed']} "
            f"active_final={report['active_pairs_final']} "
            f"obs_max={report['summary']['step_observation_abs_error_max']:.3g} "
            f"reward_max={report['summary']['reward_abs_error_max']:.3g}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
