#!/usr/bin/env python3
"""Collect safe, successful executed V6 CLOSE actions for self-imitation.

Every environment executes the frozen rl_games SEARCH actor until the public
q=0.30/four-frame handoff.  A pre-launch balanced assignment gives half the
slots deterministic V6 CLOSE and half the same V6 actor with hand-only
exploration while unlatched.  The checkpoint-native router supplies the exact
common frozen LIFT actor after latch.  Only each slot's first episode is
audited, and only trigger-through-first-latch actions from strict 20 cm,
force-safe successes are serialized as actor rehearsal rows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import traceback
from typing import Any, Mapping

import torch

from collect_online_close_ab import (
    _checkpoint_directory,
    _checkpoint_hashes,
    _load_routed_agent,
    _regular_file,
    _validate_live_task,
    git_provenance,
    runtime_asset_fingerprints,
    runtime_provenance,
    source_fingerprints as ab_source_fingerprints,
)
from online_handoff import update_online_handoff
import v6_success_self_imitation as contract


EXTRA_SOURCE_FILES = (
    "scripts/flashsac/collect_v6_success_self_imitation.py",
    "scripts/flashsac/v6_success_self_imitation.py",
    "scripts/flashsac/actor_rehearsal.py",
)


@dataclass(frozen=True)
class CollectionSpec:
    repository_root: Path
    v6_checkpoint: Path
    search_checkpoint: Path
    seed: int
    replicate: str
    num_envs: int
    min_retained: int
    exploratory_hand_noise_scale: float
    kit_args: str
    output_stem: Path
    dataset_output: Path
    report_output: Path
    checkpoint_sha256: Mapping[str, Mapping[str, str] | str]
    exploratory_assignment: torch.Tensor


def _owned(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def build_spec(args: argparse.Namespace) -> CollectionSpec:
    root = Path(__file__).resolve().parents[2]
    if isinstance(args.seed, bool):
        raise TypeError("seed must be an integer")
    if args.num_envs < 2 or args.num_envs % 2:
        raise ValueError("num_envs must be an even integer of at least two")
    if args.min_retained < 2:
        raise ValueError("min_retained must be at least two (one per cohort)")
    if args.min_retained > args.num_envs:
        raise ValueError("min_retained cannot exceed num_envs")
    if args.exploratory_hand_noise_scale != contract.EXPLORATORY_HAND_NOISE_SCALE:
        raise ValueError(
            "the v1 dataset contract requires exact "
            f"exploratory_hand_noise_scale={contract.EXPLORATORY_HAND_NOISE_SCALE}"
        )
    if getattr(args, "kit_args", None) != contract.KIT_ARGS:
        raise ValueError(
            f"self-imitation collection requires exact --kit_args={contract.KIT_ARGS!r}"
        )
    v6 = _checkpoint_directory(args.v6_checkpoint, label="V6 checkpoint")
    search = _regular_file(args.search_checkpoint, label="SEARCH checkpoint")
    output_stem = Path(os.path.abspath(os.fspath(args.output_stem)))
    dataset_output = Path(f"{output_stem}.pt")
    report_output = Path(f"{output_stem}.json")
    for output in (dataset_output, report_output):
        if _owned(output):
            raise FileExistsError(f"self-imitation output already exists: {output}")
    return CollectionSpec(
        repository_root=root,
        v6_checkpoint=v6,
        search_checkpoint=search,
        seed=int(args.seed),
        replicate=str(args.replicate),
        num_envs=int(args.num_envs),
        min_retained=int(args.min_retained),
        exploratory_hand_noise_scale=float(args.exploratory_hand_noise_scale),
        kit_args=contract.KIT_ARGS,
        output_stem=output_stem,
        dataset_output=dataset_output,
        report_output=report_output,
        checkpoint_sha256={
            "v6": _checkpoint_hashes(v6),
            "search": contract.sha256_file(search),
        },
        exploratory_assignment=contract.exploratory_cohort_mask(
            seed=int(args.seed),
            num_envs=int(args.num_envs),
            replicate=str(args.replicate),
        ),
    )


def source_fingerprints(root: Path) -> dict[str, str]:
    result = ab_source_fingerprints(root)
    for relative in EXTRA_SOURCE_FILES:
        result[relative] = contract.sha256_file(
            _regular_file(root / relative, label=f"source {relative}")
        )
    return dict(sorted(result.items()))


def _current_hashes(spec: CollectionSpec) -> dict[str, Mapping[str, str] | str]:
    return {
        "v6": _checkpoint_hashes(spec.v6_checkpoint),
        "search": contract.sha256_file(spec.search_checkpoint),
    }


def _strict_max_force(
    info: Mapping[str, Any], *, num_envs: int, device: torch.device
) -> torch.Tensor:
    """Read per-step force from the task's reset-before-clone truth, fail closed."""

    from evaluate import _require_vector, _terminal_mapping

    raw = _terminal_mapping(info)
    value = _require_vector(
        "pick_tool_terminal['max_force']",
        raw.get("max_force"),
        num_envs=num_envs,
        device=device,
        dtype=torch.float32,
    )
    if not bool(torch.isfinite(value).all()) or bool((value < 0.0).any()):
        raise FloatingPointError("reset-before max_force must be finite and non-negative")
    return value


def _assemble_dataset(
    *,
    spec: CollectionSpec,
    trial_tensors: Mapping[str, torch.Tensor],
    recorded_env: list[torch.Tensor],
    recorded_observation: list[torch.Tensor],
    recorded_action: list[torch.Tensor],
    recorded_source_step: list[torch.Tensor],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    retained_ids = trial_tensors["trial_retained"].nonzero(as_tuple=False).flatten()
    if retained_ids.numel() < spec.min_retained:
        raise RuntimeError(
            f"only {retained_ids.numel()} strict safe successes, "
            f"below min_retained={spec.min_retained}"
        )
    retained_cohort = trial_tensors["trial_cohort"][retained_ids]
    counts = [int((retained_cohort == value).sum()) for value in (0, 1)]
    if min(counts) < 1:
        raise RuntimeError(
            "self-imitation dataset requires at least one deterministic and one "
            f"exploratory safe success; got {counts}"
        )
    if not recorded_env:
        raise RuntimeError("safe successful trajectories have no recorded CLOSE rows")
    all_env = torch.cat(recorded_env).to(device="cpu", dtype=torch.int64)
    all_observation = torch.cat(recorded_observation).to(
        device="cpu", dtype=torch.float32
    )
    all_action = torch.cat(recorded_action).to(device="cpu", dtype=torch.float32)
    all_source_step = torch.cat(recorded_source_step).to(
        device="cpu", dtype=torch.int64
    )
    episode_observation: list[torch.Tensor] = []
    episode_action: list[torch.Tensor] = []
    episode_id: list[torch.Tensor] = []
    episode_step: list[torch.Tensor] = []
    episode_source_step: list[torch.Tensor] = []
    offsets = [0]
    for dataset_episode, env_id_value in enumerate(retained_ids.tolist()):
        env_id = int(env_id_value)
        selected = all_env == env_id
        observation = all_observation[selected]
        action = all_action[selected]
        source_step = all_source_step[selected]
        trigger_step = int(trial_tensors["trial_trigger_step"][env_id])
        first_latch_step = int(trial_tensors["trial_first_latch_step"][env_id])
        expected_source_step = torch.arange(
            trigger_step, first_latch_step + 1, dtype=torch.int64
        )
        if observation.shape[0] < 1 or not torch.equal(
            source_step, expected_source_step
        ):
            raise RuntimeError(
                f"env {env_id} CLOSE recording is not contiguous from trigger "
                "through the first latch transition"
            )
        if not bool((observation[:, contract.PUBLIC_LATCH_INDEX] == 0.0).all()):
            raise RuntimeError(f"env {env_id} retained a post-latch observation")
        if not torch.equal(
            action[:, : contract.ARM_ACTION_DIM],
            torch.zeros_like(action[:, : contract.ARM_ACTION_DIM]),
        ):
            raise RuntimeError(f"env {env_id} retained a non-zero CLOSE arm action")
        rows = observation.shape[0]
        episode_observation.append(observation)
        episode_action.append(action)
        episode_id.append(torch.full((rows,), dataset_episode, dtype=torch.int64))
        episode_step.append(torch.arange(rows, dtype=torch.int64))
        episode_source_step.append(source_step)
        offsets.append(offsets[-1] + rows)

    selected = retained_ids
    payload: dict[str, Any] = {
        "obs": torch.cat(episode_observation),
        "action": torch.cat(episode_action),
        "phase": torch.full((offsets[-1],), 1, dtype=torch.uint8),
        "episode_id": torch.cat(episode_id),
        "step": torch.cat(episode_step),
        "source_step": torch.cat(episode_source_step),
        "episode_offsets": torch.tensor(offsets, dtype=torch.int64),
        "episode_success": torch.ones(selected.numel(), dtype=torch.bool),
        "episode_native_success": trial_tensors["trial_native_success"][selected],
        "episode_dropped": trial_tensors["trial_dropped"][selected],
        "episode_unsafe_force": trial_tensors["trial_unsafe_force"][selected],
        "episode_unlatched_clearance_ge_5cm": trial_tensors[
            "trial_unlatched_clearance_ge_5cm"
        ][selected],
        "episode_ever_grasped": trial_tensors["trial_ever_grasped"][selected],
        "episode_triggered": trial_tensors["trial_triggered"][selected],
        "episode_ever_latched": trial_tensors["trial_ever_latched"][selected],
        "episode_latch_released_after_first": trial_tensors[
            "trial_latch_released_after_first"
        ][selected],
        "episode_source_env_id": selected.to(torch.int64),
        "episode_cohort": trial_tensors["trial_cohort"][selected],
        "episode_trigger_step": trial_tensors["trial_trigger_step"][selected],
        "episode_first_latch_step": trial_tensors["trial_first_latch_step"][selected],
        "episode_terminal_step": trial_tensors["trial_terminal_step"][selected],
        "episode_hand_noise_scale": trial_tensors["trial_hand_noise_scale"][selected],
        "episode_trigger_score": trial_tensors["trial_trigger_score"][selected],
        "episode_max_true_clearance_m": trial_tensors[
            "trial_max_true_clearance_m"
        ][selected],
        "episode_trajectory_max_force": trial_tensors[
            "trial_trajectory_max_force"
        ][selected],
        "meta": dict(metadata),
    }
    for name in (*contract.TRIAL_BOOL_NAMES, *contract.TRIAL_LONG_NAMES, *contract.TRIAL_FLOAT_NAMES):
        payload[name] = trial_tensors[name]
    return contract.build_dataset(payload)


@torch.inference_mode()
def run_collection(
    spec: CollectionSpec, *, device_string: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    device = torch.device(device_string)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"self-imitation collection requires CUDA Isaac physics, got {device}")
    assignment_cpu = contract.exploratory_cohort_mask(
        seed=spec.seed, num_envs=spec.num_envs, replicate=spec.replicate
    )
    if not torch.equal(assignment_cpu, spec.exploratory_assignment):
        raise RuntimeError("pre-launch cohort assignment changed")

    from adapter import make_pick_tool_env
    from evaluate import (
        FULL_TASK_MODE,
        StrictEpisodeTracker,
        _load_diagnostic_approach_actor,
        _read_physical_truth,
        _seed_everything,
        physical_truth_from_terminal_info,
        validate_terminal_events,
    )

    _seed_everything(spec.seed)
    importlib.import_module("xhand_inhand.tasks")
    source_before = source_fingerprints(spec.repository_root)
    runtime_assets_before = runtime_asset_fingerprints(spec.repository_root)
    for path, digest in runtime_assets_before.items():
        if not Path(path).is_absolute() and source_before.get(path) != digest:
            raise RuntimeError(f"source/runtime asset fingerprints disagree for {path}")
    git_paths = tuple(
        sorted(
            set(source_before)
            | {
                path
                for path in runtime_assets_before
                if not Path(path).is_absolute()
            }
        )
    )
    git_before = git_provenance(spec.repository_root, git_paths)
    if git_before["source_files_dirty"] or git_before["flashsac_dirty"]:
        raise RuntimeError("self-imitation source and FlashSAC trees must be committed and clean")

    env = make_pick_tool_env(
        num_envs=spec.num_envs,
        device=device_string,
        seed=spec.seed,
        strict=True,
        validate_finite=True,
    )
    try:
        _validate_live_task(env)
        agent, load = _load_routed_agent(env, spec.v6_checkpoint, seed=spec.seed)
        if git_before["flashsac_commit"] != load["fork_commit"]:
            raise RuntimeError("loaded FlashSAC fork differs from submodule HEAD")
        search_actor = _load_diagnostic_approach_actor(
            spec.search_checkpoint, device=device
        ).eval()
        if _current_hashes(spec) != spec.checkpoint_sha256:
            raise RuntimeError("a source checkpoint changed while loading")

        observation, _ = env.reset(seed=spec.seed, randomize_episode_lengths=False)
        if observation.shape != (spec.num_envs, contract.OBSERVATION_DIM):
            raise RuntimeError("reset violated the obs115 contract")
        initial_truth = _read_physical_truth(env.unwrapped, task_mode=FULL_TASK_MODE)
        if bool(initial_truth.grasped.any()):
            raise RuntimeError("self-imitation reset unexpectedly begins with a grasp latch")
        tracker = StrictEpisodeTracker(
            episodes=spec.num_envs,
            num_envs=spec.num_envs,
            device=env.device,
            initial_truth=initial_truth,
            task_mode=FULL_TASK_MODE,
        )
        exploratory = assignment_cpu.to(device=env.device)
        ready_count = torch.zeros(spec.num_envs, dtype=torch.long, device=env.device)
        option_active = torch.zeros(spec.num_envs, dtype=torch.bool, device=env.device)
        triggered = torch.zeros_like(option_active)
        trigger_step = torch.full(
            (spec.num_envs,), -1, dtype=torch.long, device=env.device
        )
        trigger_score = torch.zeros(
            spec.num_envs, dtype=torch.float32, device=env.device
        )
        ever_latched = torch.zeros_like(option_active)
        latch_released_after_first = torch.zeros_like(option_active)
        first_latch_step = torch.full_like(trigger_step, -1)
        episode_step = torch.zeros_like(trigger_step)
        trajectory_max_force = torch.zeros(
            spec.num_envs, dtype=torch.float32, device=env.device
        )
        recorded_env: list[torch.Tensor] = []
        recorded_observation: list[torch.Tensor] = []
        recorded_action: list[torch.Tensor] = []
        recorded_source_step: list[torch.Tensor] = []
        cfg = env.unwrapped.cfg
        vector_steps = 0

        while not tracker.complete and vector_steps < contract.MAX_EPISODE_ACTIONS:
            active_before = tracker.active.clone()
            handoff = update_online_handoff(
                observation,
                ready_count_before=ready_count,
                option_active_before=option_active,
                min_score=contract.HANDOFF_MIN_SCORE,
                hold_steps=contract.HANDOFF_HOLD_STEPS,
            )
            trigger = active_before & handoff["trigger"]
            if bool((trigger & triggered).any()):
                raise RuntimeError("an environment triggered the sticky handoff twice")
            triggered |= trigger
            trigger_step = torch.where(trigger, episode_step, trigger_step)
            trigger_score = torch.where(
                trigger, handoff["score"].to(torch.float32), trigger_score
            )
            ready_count = handoff["ready_count_after"]
            option_active = handoff["option_active_after"]

            latch = observation[:, contract.PUBLIC_LATCH_INDEX]
            if not bool(((latch == 0.0) | (latch == 1.0)).all()):
                raise RuntimeError("public grasp latch is not binary")
            if bool((active_before & (~ever_latched) & (latch == 1.0)).any()):
                raise RuntimeError(
                    "public latch edge was not audited on the preceding action"
                )
            noise_scale = contract.build_cohort_noise_scale(
                observation,
                exploratory,
                exploratory_hand_scale=spec.exploratory_hand_noise_scale,
            )
            routed_action = agent.sample_actions(
                vector_steps + 1,
                {"next_observation": observation},
                training=True,
                noise_scale=noise_scale,
            )
            deterministic_rows = active_before & (~exploratory)
            if bool(deterministic_rows.any()):
                deterministic_action = agent.sample_actions(
                    vector_steps + 1,
                    {"next_observation": observation},
                    training=False,
                )
                if not torch.equal(
                    routed_action[deterministic_rows],
                    deterministic_action[deterministic_rows],
                ):
                    raise RuntimeError(
                        "zero-noise cohort differs from deterministic V6 execution"
                    )
            close_rows = option_active & (latch == 0.0)
            if bool(close_rows.any()) and not torch.equal(
                routed_action[close_rows, : contract.ARM_ACTION_DIM],
                torch.zeros_like(routed_action[close_rows, : contract.ARM_ACTION_DIM]),
            ):
                raise RuntimeError("checkpoint authority emitted a non-zero CLOSE arm action")
            lift_rows = option_active & (latch == 1.0)
            if bool(lift_rows.any()):
                expected_lift = agent.frozen_lift_actions(observation)
                if not torch.equal(routed_action[lift_rows], expected_lift[lift_rows]):
                    raise RuntimeError("post-latch action differs from frozen LIFT sidecar")
            with torch.no_grad():
                search_action = search_actor(observation).clamp(-1.0, 1.0)
            action = contract.select_executed_action(
                search_action=search_action,
                routed_v6_action=routed_action,
                option_active=option_active,
            )
            action = torch.where(
                active_before.unsqueeze(-1), action, torch.zeros_like(action)
            )
            record_mask = (
                active_before
                & option_active
                & (~ever_latched)
                & (latch == 0.0)
            )
            pending_env = record_mask.nonzero(as_tuple=False).flatten()
            pending_observation = observation[record_mask].detach().clone()
            pending_action = action[record_mask].detach().clone()
            pending_source_step = episode_step[record_mask].detach().clone()

            next_observation, reward, terminated, truncated, info = env.step(action)
            vector_steps += 1
            executed = env.last_executed_policy_action
            if executed is None or not torch.equal(executed, action):
                raise RuntimeError("adapter executed an action different from the saved policy action")
            if pending_env.numel():
                recorded_env.append(pending_env.cpu())
                recorded_observation.append(pending_observation.cpu())
                recorded_action.append(pending_action.cpu())
                recorded_source_step.append(pending_source_step.cpu())

            per_step_force = _strict_max_force(
                info, num_envs=spec.num_envs, device=env.device
            )
            trajectory_max_force = torch.where(
                active_before,
                torch.maximum(trajectory_max_force, per_step_force),
                trajectory_max_force,
            )
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
            transition_truth = physical_truth_from_terminal_info(
                info,
                num_envs=spec.num_envs,
                device=env.device,
                task_mode=FULL_TASK_MODE,
                close_option_confirm_steps=int(cfg.close_option_confirm_steps),
                close_option_min_hold_quality=float(cfg.close_option_min_hold_quality),
                grasp_quality_threshold=float(cfg.grasp_quality_high),
                safe_force_limit=float(cfg.grasp_bonus_max_force),
            )
            transition_observation = info.get("transition_next_observation")
            if not isinstance(transition_observation, torch.Tensor):
                raise KeyError("adapter omitted transition_next_observation")
            newly_latched, newly_released = (
                contract.public_latch_transition_masks(
                    transition_observation=transition_observation,
                    active_before=active_before,
                    ever_latched_before=ever_latched,
                )
            )
            latch_released_after_first |= newly_released
            first_latch_step = torch.where(
                newly_latched, episode_step, first_latch_step
            )
            ever_latched |= newly_latched
            post_reset_truth = _read_physical_truth(
                env.unwrapped, task_mode=FULL_TASK_MODE
            )
            tracker.step(
                reward=reward.to(torch.float32),
                terminated=terminated,
                truncated=truncated,
                events=events,
                transition_truth=transition_truth,
                post_reset_truth=post_reset_truth,
            )
            episode_step += active_before.long()
            observation = next_observation

        if not tracker.complete or len(tracker.records) != spec.num_envs:
            raise RuntimeError(
                f"collection completed {len(tracker.records)}/{spec.num_envs} first episodes"
            )
        records_by_slot = {int(record["env_slot"]): record for record in tracker.records}
        if set(records_by_slot) != set(range(spec.num_envs)):
            raise RuntimeError("tracker did not produce one record per environment slot")
        ordered = [records_by_slot[index] for index in range(spec.num_envs)]
        if any(int(record["slot_episode_index"]) != 0 for record in ordered):
            raise RuntimeError("self-imitation audit contains a post-reset episode")

        cpu_assignment = assignment_cpu.to(torch.int64)
        max_clearance = torch.tensor(
            [float(record["max_true_clearance_m"]) for record in ordered],
            dtype=torch.float32,
        )
        native_success = torch.tensor(
            [bool(record["success"]) for record in ordered], dtype=torch.bool
        )
        dropped = torch.tensor(
            [bool(record["dropped"]) for record in ordered], dtype=torch.bool
        )
        unsafe = torch.tensor(
            [bool(record["unsafe_force"]) for record in ordered], dtype=torch.bool
        )
        unlatched = torch.tensor(
            [bool(record["ever_unlatched_clearance_ge_5cm"]) for record in ordered],
            dtype=torch.bool,
        )
        ever_grasped = torch.tensor(
            [bool(record["ever_grasped"]) for record in ordered], dtype=torch.bool
        )
        ever_20cm = torch.tensor(
            [bool(record["ever_clearance_ge_20cm"]) for record in ordered],
            dtype=torch.bool,
        )
        force_cpu = trajectory_max_force.cpu()
        ever_latched_cpu = ever_latched.cpu()
        triggered_cpu = triggered.cpu()
        retained = (
            triggered_cpu
            & native_success
            & ever_latched_cpu
            & ever_grasped
            & ever_20cm
            & ~dropped
            & ~unsafe
            & ~unlatched
            & ~latch_released_after_first.cpu()
            & (force_cpu <= float(cfg.grasp_bonus_max_force))
            & (first_latch_step.cpu() >= trigger_step.cpu())
        )
        trial_tensors: dict[str, torch.Tensor] = {
            "trial_env_slot": torch.arange(spec.num_envs, dtype=torch.int64),
            "trial_cohort": cpu_assignment,
            "trial_episode_length": torch.tensor(
                [int(record["length"]) for record in ordered], dtype=torch.int64
            ),
            "trial_trigger_step": trigger_step.cpu(),
            "trial_first_latch_step": first_latch_step.cpu(),
            "trial_terminal_step": torch.tensor(
                [int(record["length"]) - 1 for record in ordered], dtype=torch.int64
            ),
            "trial_hand_noise_scale": cpu_assignment.to(torch.float32)
            * spec.exploratory_hand_noise_scale,
            "trial_trigger_score": trigger_score.cpu(),
            "trial_max_true_clearance_m": max_clearance,
            "trial_trajectory_max_force": force_cpu,
            "trial_triggered": triggered_cpu,
            "trial_ever_latched": ever_latched_cpu,
            "trial_latch_released_after_first": latch_released_after_first.cpu(),
            "trial_native_success": native_success,
            "trial_failure": torch.tensor(
                [bool(record["failure"]) for record in ordered], dtype=torch.bool
            ),
            "trial_time_out": torch.tensor(
                [bool(record["time_out"]) for record in ordered], dtype=torch.bool
            ),
            "trial_dropped": dropped,
            "trial_unsafe_force": unsafe,
            "trial_unlatched_clearance_ge_5cm": unlatched,
            "trial_ever_grasped": ever_grasped,
            "trial_ever_clearance_ge_20cm": ever_20cm,
            "trial_retained": retained,
        }

        source_after = source_fingerprints(spec.repository_root)
        runtime_assets_after = runtime_asset_fingerprints(spec.repository_root)
        git_after = git_provenance(spec.repository_root, git_paths)
        if (
            source_after != source_before
            or runtime_assets_after != runtime_assets_before
            or git_after != git_before
            or _current_hashes(spec) != spec.checkpoint_sha256
        ):
            raise RuntimeError("source, runtime asset, or checkpoint changed during collection")
        v6_hashes = spec.checkpoint_sha256["v6"]
        if not isinstance(v6_hashes, Mapping):
            raise TypeError("V6 checkpoint hash manifest is not a mapping")
        metadata = {
            **contract.REQUIRED_METADATA,
            "seed": spec.seed,
            "num_envs": spec.num_envs,
            "min_retained": spec.min_retained,
            "cohort_assignment_salt": contract.COHORT_ASSIGNMENT_SALT,
            "cohort_replicate": spec.replicate,
            "v6_checkpoint": str(spec.v6_checkpoint),
            "search_checkpoint": str(spec.search_checkpoint),
            "search_checkpoint_sha256": spec.checkpoint_sha256["search"],
            "v6_actor_sha256": v6_hashes["actor.pt"],
            "v6_task_contract_sha256": v6_hashes["task_contract.json"],
            "v6_bridge_state_sha256": v6_hashes["torch_bridge_state.pt"],
            "frozen_lift_actor_sha256": v6_hashes["frozen_lift_actor.pt"],
            "frozen_lift_semantic_sha256": load["frozen_semantic_sha256"],
            "frozen_lift_source_actor_sha256": load[
                "frozen_source_actor_sha256"
            ],
            "source_manifest_sha256": contract.manifest_sha256(source_before),
            "runtime_asset_manifest_sha256": contract.manifest_sha256(
                runtime_assets_before
            ),
            "source_sha256": source_before,
            "runtime_asset_sha256": runtime_assets_before,
            "flashsac_upstream_commit": load["upstream_commit"],
            "flashsac_fork_commit": load["fork_commit"],
            "git": git_before,
            "runtime": runtime_provenance(seed=spec.seed, device=env.device),
        }
        dataset = _assemble_dataset(
            spec=spec,
            trial_tensors=trial_tensors,
            recorded_env=recorded_env,
            recorded_observation=recorded_observation,
            recorded_action=recorded_action,
            recorded_source_step=recorded_source_step,
            metadata=metadata,
        )
        report = {
            "kind": contract.REPORT_KIND,
            "status": "complete",
            "collector": contract.COLLECTOR,
            "seed": spec.seed,
            "cohort_replicate": spec.replicate,
            "num_envs": spec.num_envs,
            "vector_steps": int(trial_tensors["trial_episode_length"].max()),
            "retained_episodes": int(retained.sum()),
            "transitions": int(dataset["obs"].shape[0]),
            "v6_actor_sha256": metadata["v6_actor_sha256"],
            "search_checkpoint_sha256": metadata["search_checkpoint_sha256"],
            "summary": contract.summarize_dataset(dataset),
        }
        contract.validate_report(report, dataset)
        return dataset, report
    finally:
        env.close()


def parse_args() -> tuple[argparse.Namespace, CollectionSpec, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--v6_checkpoint", type=Path, required=True)
    parser.add_argument("--search_checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--replicate", choices=("a", "b"), required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--min_retained", type=int, default=2)
    parser.add_argument(
        "--exploratory_hand_noise_scale",
        type=float,
        default=contract.EXPLORATORY_HAND_NOISE_SCALE,
    )
    parser.add_argument(
        "--output_stem",
        type=Path,
        required=True,
        help="immutable output stem; .pt and .json are published together",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    try:
        spec = build_spec(args)
    except (TypeError, ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
    launcher = AppLauncher(args)
    return args, spec, launcher.app


def publish_failure_attempt(spec: CollectionSpec, error: BaseException) -> Path:
    payload = {
        "kind": contract.REPORT_KIND,
        "status": "failed",
        "seed": spec.seed,
        "cohort_replicate": spec.replicate,
        "num_envs": spec.num_envs,
        "canonical_dataset_output": str(spec.dataset_output),
        "canonical_report_output": str(spec.report_output),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    for attempt in range(1, 10_000):
        output = Path(f"{spec.output_stem}.failed_attempt_{attempt:03d}.json")
        try:
            contract.publish_json_no_clobber(payload, output)
        except FileExistsError:
            continue
        return output
    raise RuntimeError("self-imitation failure-attempt namespace is exhausted")


def main() -> None:
    args, spec, simulation_app = parse_args()
    try:
        try:
            dataset, report = run_collection(
                spec, device_string=str(args.device or "cuda:0")
            )
            digest = contract.publish_dataset_and_report_no_clobber(
                dataset,
                report,
                dataset_output=spec.dataset_output,
                report_output=spec.report_output,
            )
            print(
                "[v6-success-self-imitation] "
                f"seed={spec.seed} replicate={spec.replicate} "
                f"envs={spec.num_envs} retained={report['retained_episodes']} "
                f"rows={report['transitions']} sha256={digest}",
                flush=True,
            )
        except BaseException as error:
            failure = publish_failure_attempt(spec, error)
            print(
                f"[v6-success-self-imitation] failure recorded at {failure}",
                flush=True,
            )
            raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
