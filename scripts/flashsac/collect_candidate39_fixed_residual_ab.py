#!/usr/bin/env python3
"""Collect reserved-seed Candidate 39 fixed-direction A/B evidence.

SEARCH, the deterministic V6 router, arm authority, and frozen LIFT are common
to both arms.  During the sticky 32-action pre-latch CLOSE window, treated
slots receive the one sealed fixed hand14 pre-tanh residual; controls receive
an exact zero residual.  Every slot's first episode is retained without outcome
selection, including legal SEARCH-side latch/release cases with zero rows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import math
import os
from pathlib import Path
import traceback
from typing import Any, Mapping

import torch

import candidate39_fixed_episode_residual as artifact_contract
from collect_candidate39_episode_residual_ab import (
    _append_rows,
    _cat_rows,
    _public_latch_transition_masks,
    _read_transition_telemetry,
)
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
import option_residual_screen


WINDOW_STEPS = 32
TOKEN_SCALE = 0.05
DISTAL_SCALE = 0.025
RAW_Z_ABS_CAP = 2.0
TOKEN_COMPONENT_CAP = 0.10
DISTAL_COMPONENT_CAP = 0.05
PRE_TANH_L2_CAP = 0.20
NUM_ENVS = 64
ALLOWED_NUM_ENVS = (8, NUM_ENVS)
RESERVED_EVIDENCE_SEEDS = (329, 330)
PROHIBITED_FORMAL_SEEDS = (331, 332, 333)
HANDOFF_MIN_SCORE = 0.30
HANDOFF_HOLD_STEPS = 4
KIT_ARGS = "--/app/extensions/fsWatcherEnabled=false"
OBSERVATION_DIM = 115
ACTION_DIM = 21
ARM_ACTION_DIM = 7
HAND_ACTION_DIM = 14
PUBLIC_LATCH_INDEX = 106
MAX_EPISODE_ACTIONS = 999
EXPECTED_V6_SHA256 = {
    "actor.pt": "d9aacbd48891c192c0d1491514137262b82990fa71440787076403de06606288",
    "task_contract.json": "8a2a135fa1cd5bb01965fc7dc6d6a73978dffc1bf55941ae392146de1d17f61d",
    "torch_bridge_state.pt": "78176bb99eb5dac80a2789b231d54db6cf2b86132bc7835a129dd7f5a7df978c",
    "frozen_lift_actor.pt": "117f1b0ae3641bd24b6b9f3d585576a79b5aff968139c0dde86e6346f60ebaa0",
}
EXPECTED_SEARCH_SHA256 = (
    "b91555d8227cf4e87e41ae858d7f8b1ee42ae920a9d829f2f5b1453b529f1f02"
)
EXPECTED_FROZEN_SEMANTIC_SHA256 = (
    "8af0fdbc2572346b7519fbb4b29354fb5bc882a0c0020c8c3ace8376696ab324"
)
EXPECTED_FROZEN_SOURCE_SHA256 = (
    "7757869eaa1df02f5f52c2dcd1353fb4a486b7512019650a8234d76702b9a1fb"
)
EXPECTED_FLASHSAC_FORK_COMMIT = "5ecf331fa11cd457dd39018b3d68af571b257666"

EXTRA_SOURCE_FILES = (
    "scripts/flashsac/collect_candidate39_fixed_residual_ab.py",
    "scripts/flashsac/candidate39_fixed_episode_residual.py",
    "scripts/flashsac/collect_candidate39_episode_residual_ab.py",
    "scripts/flashsac/candidate39_episode_residual.py",
    "scripts/flashsac/option_residual_screen.py",
    "scripts/flashsac/candidate39_fixed_episode_residual_test.py",
    "scripts/flashsac/analyze_candidate39_fixed_validation.py",
    "scripts/flashsac/analyze_candidate39_fixed_validation_test.py",
    artifact_contract.VALIDATION_PLAN,
    artifact_contract.FIXED_DIRECTION_MANIFEST,
)


@dataclass(frozen=True)
class CollectionSpec:
    repository_root: Path
    v6_checkpoint: Path
    search_checkpoint: Path
    fixed_direction: Path
    validation_plan: Path
    fixed_direction_manifest: Path
    seed: int
    replicate: str
    num_envs: int
    window_steps: int
    token_scale: float
    distal_scale: float
    raw_z_abs_cap: float
    token_component_cap: float
    distal_component_cap: float
    pre_tanh_l2_cap: float
    kit_args: str
    output_stem: Path
    artifact_output: Path
    report_output: Path
    immutable_sha256: Mapping[str, Mapping[str, str] | str]
    treatment: torch.Tensor
    assignment_rank: torch.Tensor
    fixed_z: torch.Tensor


def _owned(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _require_exact(name: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, float):
        valid = (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        )
    else:
        valid = actual == expected and type(actual) is type(expected)
    if not valid:
        raise ValueError(
            f"Candidate 39 fixed validation requires exact {name}={expected!r}"
        )


def _exact_registered_file(root: Path, supplied: Path, relative: str, label: str) -> Path:
    expected = (root / relative).resolve(strict=True)
    actual = _regular_file(supplied, label=label).resolve(strict=True)
    if actual != expected:
        raise ValueError(f"{label} must be exact registered path {expected}")
    return actual


def build_spec(args: argparse.Namespace) -> CollectionSpec:
    """Resolve policies, fixed direction, and assignment before AppLauncher."""

    root = Path(__file__).resolve().parents[2]
    artifact_contract.validate_sealed_validation_plan()
    if not isinstance(args.seed, int) or isinstance(args.seed, bool) or args.seed < 0:
        raise TypeError("seed must be a non-negative integer")
    if (
        not isinstance(args.num_envs, int)
        or isinstance(args.num_envs, bool)
        or args.num_envs not in ALLOWED_NUM_ENVS
    ):
        raise ValueError(
            f"fixed validation num_envs must be one of {ALLOWED_NUM_ENVS}; "
            "8 is smoke-only and evidence uses 64"
        )
    if args.num_envs == NUM_ENVS and args.seed not in RESERVED_EVIDENCE_SEEDS:
        raise ValueError("64-env fixed validation is reserved to seeds 329 and 330")
    if args.num_envs == 8 and (args.seed, args.replicate) != (326, "b"):
        raise ValueError("8-env non-evidence smoke must be exact registered run 326b")
    if args.seed in PROHIBITED_FORMAL_SEEDS:
        raise ValueError("formal seeds 331-333 are prohibited in development validation")
    if args.replicate not in {"a", "b"}:
        raise ValueError("replicate must be 'a' or 'b'")
    _require_exact("window_steps", args.window_steps, WINDOW_STEPS)
    _require_exact("token_scale", args.token_scale, TOKEN_SCALE)
    _require_exact("distal_scale", args.distal_scale, DISTAL_SCALE)
    _require_exact("raw_z_abs_cap", args.raw_z_abs_cap, RAW_Z_ABS_CAP)
    _require_exact("token_component_cap", args.token_component_cap, TOKEN_COMPONENT_CAP)
    _require_exact(
        "distal_component_cap", args.distal_component_cap, DISTAL_COMPONENT_CAP
    )
    _require_exact("pre_tanh_l2_cap", args.pre_tanh_l2_cap, PRE_TANH_L2_CAP)
    if getattr(args, "kit_args", None) != KIT_ARGS:
        raise ValueError(f"fixed validation requires exact --kit_args={KIT_ARGS!r}")

    v6 = _checkpoint_directory(args.v6_checkpoint, label="exact V6 checkpoint")
    expected_v6 = (root / artifact_contract.V6_CHECKPOINT_PATH).resolve(strict=True)
    if v6 != expected_v6:
        raise ValueError(f"V6 checkpoint must be exact registered path {expected_v6}")
    search = _exact_registered_file(
        root,
        args.search_checkpoint,
        artifact_contract.SEARCH_CHECKPOINT_PATH,
        "SEARCH checkpoint",
    )
    fixed_direction = _exact_registered_file(
        root,
        args.fixed_direction,
        artifact_contract.FIXED_DIRECTION_PATH,
        "sealed fixed direction",
    )
    validation_plan = _regular_file(
        root / artifact_contract.VALIDATION_PLAN, label="sealed validation plan"
    ).resolve(strict=True)
    fixed_manifest = _regular_file(
        root / artifact_contract.FIXED_DIRECTION_MANIFEST,
        label="sealed fixed-direction manifest",
    ).resolve(strict=True)

    v6_hashes = _checkpoint_hashes(v6)
    search_hash = artifact_contract.sha256_file(search)
    plan_hash = artifact_contract.sha256_file(validation_plan)
    manifest_hash = artifact_contract.sha256_file(fixed_manifest)
    fixed_hash = artifact_contract.sha256_file(fixed_direction)
    if v6_hashes != EXPECTED_V6_SHA256:
        raise ValueError("fixed validation requires the exact sealed V6 checkpoint")
    if search_hash != EXPECTED_SEARCH_SHA256:
        raise ValueError("fixed validation requires the exact sealed SEARCH checkpoint")
    if manifest_hash != artifact_contract.FIXED_DIRECTION_MANIFEST_SHA256:
        raise ValueError("fixed-direction manifest SHA256 changed")
    if fixed_hash != artifact_contract.FIXED_DIRECTION_SHA256:
        raise ValueError("fixed-direction payload SHA256 changed")
    fixed_payload = artifact_contract.load_fixed_direction(fixed_direction)
    fixed_z = fixed_payload["fixed_z"]
    if not torch.equal(fixed_z, artifact_contract.expected_fixed_z()):
        raise RuntimeError("loaded fixed direction differs from the sealed hand14")

    output_stem = Path(os.path.abspath(os.fspath(args.output_stem)))
    if args.num_envs == 8:
        expected_output = (
            root
            / "logs/flashsac/pick_tool/51_c39_fixed_direction_smoke_s326_b/trial"
        ).resolve()
    else:
        expected_output = (
            root
            / (
                "logs/flashsac/pick_tool/51_c39_fixed_direction_dev_"
                f"s{args.seed}_{args.replicate}/trial"
            )
        ).resolve()
    if output_stem != expected_output:
        raise ValueError(
            f"fixed validation requires canonical output stem {expected_output}"
        )
    artifact_output = Path(f"{output_stem}.pt")
    report_output = Path(f"{output_stem}.json")
    for output in (artifact_output, report_output):
        if _owned(output):
            raise FileExistsError(f"fixed validation output already exists: {output}")

    treatment = artifact_contract.exact_balanced_treatment_mask(
        seed=args.seed, num_envs=args.num_envs, replicate=args.replicate
    )
    rank = artifact_contract.assignment_rank(
        seed=args.seed, num_envs=args.num_envs
    )
    if int(treatment.sum()) != args.num_envs // 2:
        raise RuntimeError("pre-launch fixed-direction assignment is not balanced")
    if not torch.equal(
        treatment,
        (rank < args.num_envs // 2)
        if args.replicate == "a"
        else (rank >= args.num_envs // 2),
    ):
        raise RuntimeError("pre-launch treatment and assignment rank disagree")

    return CollectionSpec(
        repository_root=root,
        v6_checkpoint=v6,
        search_checkpoint=search,
        fixed_direction=fixed_direction,
        validation_plan=validation_plan,
        fixed_direction_manifest=fixed_manifest,
        seed=int(args.seed),
        replicate=str(args.replicate),
        num_envs=int(args.num_envs),
        window_steps=int(args.window_steps),
        token_scale=float(args.token_scale),
        distal_scale=float(args.distal_scale),
        raw_z_abs_cap=float(args.raw_z_abs_cap),
        token_component_cap=float(args.token_component_cap),
        distal_component_cap=float(args.distal_component_cap),
        pre_tanh_l2_cap=float(args.pre_tanh_l2_cap),
        kit_args=KIT_ARGS,
        output_stem=output_stem,
        artifact_output=artifact_output,
        report_output=report_output,
        immutable_sha256={
            "v6": v6_hashes,
            "search": search_hash,
            "validation_plan": plan_hash,
            "fixed_direction_manifest": manifest_hash,
            "fixed_direction": fixed_hash,
        },
        treatment=treatment.detach().clone(),
        assignment_rank=rank.detach().clone(),
        fixed_z=fixed_z.detach().clone(),
    )


def source_fingerprints(root: Path) -> dict[str, str]:
    result = ab_source_fingerprints(root)
    for relative in EXTRA_SOURCE_FILES:
        result[relative] = artifact_contract.sha256_file(
            _regular_file(root / relative, label=f"source {relative}")
        )
    return dict(sorted(result.items()))


def _current_hashes(spec: CollectionSpec) -> dict[str, Mapping[str, str] | str]:
    return {
        "v6": _checkpoint_hashes(spec.v6_checkpoint),
        "search": artifact_contract.sha256_file(spec.search_checkpoint),
        "validation_plan": artifact_contract.sha256_file(spec.validation_plan),
        "fixed_direction_manifest": artifact_contract.sha256_file(
            spec.fixed_direction_manifest
        ),
        "fixed_direction": artifact_contract.sha256_file(spec.fixed_direction),
    }


@torch.inference_mode()
def run_collection(
    spec: CollectionSpec, *, device_string: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    device = torch.device(device_string)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            f"Candidate 39 fixed validation requires CUDA Isaac physics, got {device}"
        )

    treatment_check = artifact_contract.exact_balanced_treatment_mask(
        seed=spec.seed, num_envs=spec.num_envs, replicate=spec.replicate
    )
    rank_check = artifact_contract.assignment_rank(
        seed=spec.seed, num_envs=spec.num_envs
    )
    fixed_check = artifact_contract.load_fixed_direction(spec.fixed_direction)[
        "fixed_z"
    ]
    if (
        not torch.equal(spec.treatment, treatment_check)
        or not torch.equal(spec.assignment_rank, rank_check)
        or not torch.equal(spec.fixed_z, fixed_check)
    ):
        raise RuntimeError("pre-launch fixed-direction design changed")

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
        raise RuntimeError(
            "fixed-direction source and FlashSAC trees must be committed and clean"
        )

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
        if load["fork_commit"] != EXPECTED_FLASHSAC_FORK_COMMIT:
            raise RuntimeError("loaded FlashSAC fork differs from the sealed plan")
        if load["frozen_semantic_sha256"] != EXPECTED_FROZEN_SEMANTIC_SHA256:
            raise RuntimeError("frozen LIFT semantic receipt changed")
        if load["frozen_source_actor_sha256"] != EXPECTED_FROZEN_SOURCE_SHA256:
            raise RuntimeError("frozen LIFT source actor receipt changed")
        search_actor = _load_diagnostic_approach_actor(
            spec.search_checkpoint, device=device
        ).eval()
        if _current_hashes(spec) != spec.immutable_sha256:
            raise RuntimeError("a fixed-validation immutable input changed while loading")

        observation, _ = env.reset(seed=spec.seed, randomize_episode_lengths=False)
        if observation.shape != (spec.num_envs, OBSERVATION_DIM):
            raise RuntimeError("fixed validation reset violated the obs115 contract")
        initial_truth = _read_physical_truth(env.unwrapped, task_mode=FULL_TASK_MODE)
        if bool(initial_truth.grasped.any()):
            raise RuntimeError("fixed validation reset unexpectedly begins latched")
        tracker = StrictEpisodeTracker(
            episodes=spec.num_envs,
            num_envs=spec.num_envs,
            device=env.device,
            initial_truth=initial_truth,
            task_mode=FULL_TASK_MODE,
        )

        treatment = spec.treatment.to(device=env.device)
        fixed_z = spec.fixed_z.to(device=env.device).expand(spec.num_envs, -1)
        scale = option_residual_screen.grouped_pre_tanh_scale(
            token_scale=spec.token_scale,
            distal_scale=spec.distal_scale,
            dtype=torch.float32,
            device=env.device,
        )
        component_cap = torch.tensor(
            [spec.token_component_cap] * option_residual_screen.TOKEN_ACTION_DIM
            + [spec.distal_component_cap]
            * option_residual_screen.DISTAL_ACTION_DIM,
            dtype=torch.float32,
            device=env.device,
        )
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
        first_latch_step = torch.full_like(trigger_step, -1)
        latch_released_after_first = torch.zeros_like(option_active)
        intervention_steps = torch.zeros_like(trigger_step)
        episode_step = torch.zeros_like(trigger_step)
        trajectory_max_force = torch.zeros(
            spec.num_envs, dtype=torch.float32, device=env.device
        )

        row_shapes = {
            "row_env_slot": (),
            "row_episode_step": (),
            "row_close_age": (),
            "row_public_latch_before": (),
            "row_residual_active": (),
            "row_observation": (OBSERVATION_DIM,),
            "row_base_mean_hand": (HAND_ACTION_DIM,),
            "row_applied_delta": (HAND_ACTION_DIM,),
            "row_baseline_action": (ACTION_DIM,),
            "row_candidate_action": (ACTION_DIM,),
            "row_executed_action": (ACTION_DIM,),
            "row_transition_public_latch": (),
            "row_transition_grasped": (),
            "row_transition_true_clearance_m": (),
            "row_grasp_quality": (),
            "row_hold_quality": (),
            "row_max_force_n": (),
        }
        row_dtypes = {
            "row_env_slot": torch.int64,
            "row_episode_step": torch.int64,
            "row_close_age": torch.int64,
            "row_public_latch_before": torch.bool,
            "row_residual_active": torch.bool,
            "row_observation": torch.float32,
            "row_base_mean_hand": torch.float32,
            "row_applied_delta": torch.float32,
            "row_baseline_action": torch.float32,
            "row_candidate_action": torch.float32,
            "row_executed_action": torch.float32,
            "row_transition_public_latch": torch.bool,
            "row_transition_grasped": torch.bool,
            "row_transition_true_clearance_m": torch.float32,
            "row_grasp_quality": torch.float32,
            "row_hold_quality": torch.float32,
            "row_max_force_n": torch.float32,
        }
        rows: dict[str, list[torch.Tensor]] = {name: [] for name in row_shapes}
        cfg = env.unwrapped.cfg
        vector_steps = 0

        while not tracker.complete and vector_steps < MAX_EPISODE_ACTIONS:
            active_before = tracker.active.clone()
            handoff = update_online_handoff(
                observation,
                ready_count_before=ready_count,
                option_active_before=option_active,
                min_score=HANDOFF_MIN_SCORE,
                hold_steps=HANDOFF_HOLD_STEPS,
            )
            trigger = active_before & handoff["trigger"]
            if bool((trigger & triggered).any()):
                raise RuntimeError("an environment triggered fixed validation twice")
            triggered |= trigger
            trigger_step = torch.where(trigger, episode_step, trigger_step)
            trigger_score = torch.where(
                trigger, handoff["score"].to(torch.float32), trigger_score
            )
            ready_count = handoff["ready_count_after"]
            option_active = handoff["option_active_after"]

            latch_float = observation[:, PUBLIC_LATCH_INDEX]
            if not bool(((latch_float == 0.0) | (latch_float == 1.0)).all()):
                raise RuntimeError("public grasp latch is not binary")
            public_latch = latch_float == 1.0
            if bool((active_before & (~ever_latched) & public_latch).any()):
                raise RuntimeError("public latch edge was not audited on preceding action")
            window = option_residual_screen.option_residual_window(
                episode_active=active_before,
                option_active=option_active,
                public_latch=public_latch,
                ever_latched=ever_latched,
                episode_step=episode_step,
                trigger_step=trigger_step,
                window_steps=spec.window_steps,
            )
            close_age = window.close_age
            intervention_window = window.active

            actor_mean = agent.deterministic_actor_mean(observation)
            raw_close_action = agent.apply_action_authority(
                torch.tanh(actor_mean), observation
            )
            baseline_action = agent.sample_actions(
                vector_steps + 1,
                {"next_observation": observation},
                training=False,
            )
            if baseline_action.shape != (spec.num_envs, ACTION_DIM):
                raise RuntimeError("V6 baseline action violated action21")
            if actor_mean.shape != baseline_action.shape or not bool(
                torch.isfinite(actor_mean).all()
            ):
                raise RuntimeError("V6 raw deterministic mean violated action21")
            if not bool(torch.isfinite(baseline_action).all()) or bool(
                (baseline_action.abs() > 1.0).any()
            ):
                raise RuntimeError("V6 baseline action is non-finite or out of bounds")
            close_rows = active_before & option_active & (~public_latch)
            if bool(close_rows.any()) and not torch.equal(
                raw_close_action[close_rows], baseline_action[close_rows]
            ):
                raise RuntimeError(
                    "deterministic V6 CLOSE sample differs from raw mean authority path"
                )
            if bool(close_rows.any()) and not torch.equal(
                baseline_action[close_rows, :ARM_ACTION_DIM],
                torch.zeros_like(baseline_action[close_rows, :ARM_ACTION_DIM]),
            ):
                raise RuntimeError("exact V6 emitted a non-zero CLOSE arm action")
            lift_rows = active_before & option_active & public_latch
            if bool(lift_rows.any()):
                expected_lift = agent.frozen_lift_actions(observation)
                if not torch.equal(
                    baseline_action[lift_rows], expected_lift[lift_rows]
                ):
                    raise RuntimeError("post-latch action differs from common frozen LIFT")

            residual = option_residual_screen.apply_option_residual(
                baseline_action=baseline_action,
                option_active=intervention_window,
                public_latch=public_latch,
                treatment=treatment,
                raw_z=fixed_z,
                pre_tanh_scale=scale,
                raw_z_abs_cap=spec.raw_z_abs_cap,
                pre_tanh_abs_cap=component_cap,
                pre_tanh_l2_cap=spec.pre_tanh_l2_cap,
            )
            expected_eligible = intervention_window & treatment
            if not torch.equal(residual.eligible, expected_eligible):
                raise RuntimeError("fixed-direction residual eligibility changed")
            if not torch.equal(residual.raw_z, fixed_z):
                raise RuntimeError("residual helper changed the sealed fixed direction")
            if not torch.equal(
                residual.effective_z,
                torch.where(treatment.unsqueeze(-1), fixed_z, torch.zeros_like(fixed_z)),
            ):
                raise RuntimeError("fixed-direction effective residual changed")
            if not torch.equal(
                residual.action[:, :ARM_ACTION_DIM],
                baseline_action[:, :ARM_ACTION_DIM],
            ):
                raise RuntimeError("fixed-direction residual changed arm authority")
            ineligible = ~expected_eligible
            if bool(ineligible.any()) and not torch.equal(
                residual.action[ineligible], baseline_action[ineligible]
            ):
                raise RuntimeError("fixed direction changed an ineligible action")
            if bool(ineligible.any()) and not torch.equal(
                residual.pre_tanh_residual[ineligible],
                torch.zeros_like(residual.pre_tanh_residual[ineligible]),
            ):
                raise RuntimeError("fixed direction assigned budget outside eligibility")
            expected_delta = artifact_contract.expected_applied_delta().to(env.device)
            if bool(expected_eligible.any()) and not torch.equal(
                residual.pre_tanh_residual[expected_eligible],
                expected_delta.expand(int(expected_eligible.sum()), -1),
            ):
                raise RuntimeError("treated slots did not receive the one exact fixed delta")
            residual_l2 = torch.linalg.vector_norm(
                residual.pre_tanh_residual, dim=-1
            )
            if bool((residual_l2 > spec.pre_tanh_l2_cap + 1e-6).any()):
                raise RuntimeError("fixed-direction pre-tanh L2 cap was exceeded")
            if not bool(torch.isfinite(residual.action).all()) or bool(
                (residual.action.abs() > 1.0).any()
            ):
                raise RuntimeError("fixed-direction candidate escaped tanh bounds")

            search_action = search_actor(observation).clamp(-1.0, 1.0)
            if search_action.shape != baseline_action.shape or not bool(
                torch.isfinite(search_action).all()
            ):
                raise RuntimeError("SEARCH emitted an invalid action")
            action = torch.where(
                option_active.unsqueeze(-1), residual.action, search_action
            )
            action = torch.where(
                active_before.unsqueeze(-1), action, torch.zeros_like(action)
            )
            option_rows = active_before & option_active
            search_rows = active_before & (~option_active)
            if bool(option_rows.any()) and not torch.equal(
                action[option_rows], residual.action[option_rows]
            ):
                raise RuntimeError("option execution differs from fixed overlay")
            if bool(search_rows.any()) and not torch.equal(
                action[search_rows], search_action[search_rows]
            ):
                raise RuntimeError("pre-handoff execution differs from exact SEARCH")
            if bool(close_rows.any()) and not torch.equal(
                action[close_rows, :ARM_ACTION_DIM],
                torch.zeros_like(action[close_rows, :ARM_ACTION_DIM]),
            ):
                raise RuntimeError("executed CLOSE action has non-zero arm authority")

            record_mask = intervention_window
            pending = {
                "row_env_slot": torch.arange(
                    spec.num_envs, dtype=torch.int64, device=env.device
                ),
                "row_episode_step": episode_step,
                "row_close_age": close_age,
                "row_public_latch_before": public_latch,
                "row_residual_active": residual.eligible,
                "row_observation": observation,
                "row_base_mean_hand": actor_mean[:, ARM_ACTION_DIM:],
                "row_applied_delta": residual.pre_tanh_residual,
                "row_baseline_action": baseline_action,
                "row_candidate_action": residual.action,
                "row_executed_action": action,
            }

            next_observation, reward, terminated, truncated, info = env.step(action)
            vector_steps += 1
            executed = env.last_executed_policy_action
            if executed is None or not torch.equal(executed, action):
                raise RuntimeError("adapter executed a different fixed-validation action")
            telemetry = _read_transition_telemetry(
                info, num_envs=spec.num_envs, device=env.device
            )
            transition_observation = info.get("transition_next_observation")
            if (
                not isinstance(transition_observation, torch.Tensor)
                or transition_observation.shape != observation.shape
                or transition_observation.dtype != observation.dtype
                or transition_observation.device != observation.device
                or not bool(torch.isfinite(transition_observation).all())
            ):
                raise RuntimeError("adapter omitted a valid transition_next_observation")
            transition_latch_float = transition_observation[:, PUBLIC_LATCH_INDEX]
            if not bool(
                ((transition_latch_float == 0.0) | (transition_latch_float == 1.0)).all()
            ):
                raise RuntimeError("transition public latch is not binary")
            pending.update(
                {
                    "row_transition_public_latch": transition_latch_float == 1.0,
                    "row_transition_grasped": telemetry["is_grasped"],
                    "row_transition_true_clearance_m": telemetry["true_clearance"],
                    "row_grasp_quality": telemetry["grasp_quality"],
                    "row_hold_quality": telemetry["hold_quality"],
                    "row_max_force_n": telemetry["max_force"],
                }
            )
            _append_rows(rows, mask=record_mask, values=pending)
            intervention_steps += residual.eligible.long()
            trajectory_max_force = torch.where(
                active_before,
                torch.maximum(trajectory_max_force, telemetry["max_force"]),
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
            newly_latched, newly_released = _public_latch_transition_masks(
                transition_observation=transition_observation,
                active_before=active_before,
                ever_latched_before=ever_latched,
            )
            if bool((newly_latched & ever_latched).any()):
                raise RuntimeError("fixed validation observed a second first-latch edge")
            first_latch_step = torch.where(
                newly_latched, episode_step, first_latch_step
            )
            latch_released_after_first |= newly_released
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
                f"fixed validation completed {len(tracker.records)}/{spec.num_envs} episodes"
            )
        records_by_slot = {
            int(record["env_slot"]): record for record in tracker.records
        }
        if set(records_by_slot) != set(range(spec.num_envs)):
            raise RuntimeError("fixed validation tracker omitted an environment slot")
        ordered = [records_by_slot[index] for index in range(spec.num_envs)]
        if any(int(record["slot_episode_index"]) != 0 for record in ordered):
            raise RuntimeError("fixed validation artifact contains a post-reset episode")

        episode_tensors = {
            "env_slot": torch.arange(spec.num_envs, dtype=torch.int64),
            "treatment": spec.treatment.to(torch.bool),
            "assignment_rank": spec.assignment_rank.to(torch.int64),
            "fixed_z": spec.fixed_z.to(torch.float32).expand(
                spec.num_envs, -1
            ).clone(),
            "triggered": triggered.cpu(),
            "trigger_step": trigger_step.cpu(),
            "trigger_score": trigger_score.cpu(),
            "first_latch_step": first_latch_step.cpu(),
            "latch_released_after_first": latch_released_after_first.cpu(),
            "intervention_steps": intervention_steps.cpu(),
            "episode_length": torch.tensor(
                [int(record["length"]) for record in ordered], dtype=torch.int64
            ),
            "trajectory_max_force_n": trajectory_max_force.cpu(),
            "max_true_clearance_m": torch.tensor(
                [float(record["max_true_clearance_m"]) for record in ordered],
                dtype=torch.float32,
            ),
            "success": torch.tensor(
                [bool(record["success"]) for record in ordered], dtype=torch.bool
            ),
            "failure": torch.tensor(
                [bool(record["failure"]) for record in ordered], dtype=torch.bool
            ),
            "time_out": torch.tensor(
                [bool(record["time_out"]) for record in ordered], dtype=torch.bool
            ),
            "dropped": torch.tensor(
                [bool(record["dropped"]) for record in ordered], dtype=torch.bool
            ),
            "unsafe_force": torch.tensor(
                [bool(record["unsafe_force"]) for record in ordered],
                dtype=torch.bool,
            ),
            "unlatched_clearance_ge_5cm": torch.tensor(
                [
                    bool(record["ever_unlatched_clearance_ge_5cm"])
                    for record in ordered
                ],
                dtype=torch.bool,
            ),
            "ever_grasped": torch.tensor(
                [bool(record["ever_grasped"]) for record in ordered],
                dtype=torch.bool,
            ),
            "ever_clearance_ge_20cm": torch.tensor(
                [bool(record["ever_clearance_ge_20cm"]) for record in ordered],
                dtype=torch.bool,
            ),
            "latched_within_window": (
                triggered.cpu()
                & (first_latch_step.cpu() >= trigger_step.cpu())
                & (
                    (first_latch_step.cpu() - trigger_step.cpu())
                    < spec.window_steps
                )
            ),
        }
        step_tensors = _cat_rows(rows, shapes=row_shapes, dtypes=row_dtypes)
        per_env_rows = torch.bincount(
            step_tensors["row_env_slot"], minlength=spec.num_envs
        )
        if bool((per_env_rows > spec.window_steps).any()):
            raise RuntimeError("fixed validation recorded more than 32 rows per slot")
        if bool(step_tensors["row_public_latch_before"].any()):
            raise RuntimeError("fixed validation recorded a latched pre-action row")
        if not torch.equal(
            step_tensors["row_candidate_action"][:, :ARM_ACTION_DIM],
            step_tensors["row_baseline_action"][:, :ARM_ACTION_DIM],
        ) or not torch.equal(
            step_tensors["row_executed_action"],
            step_tensors["row_candidate_action"],
        ):
            raise RuntimeError("fixed-validation recorded action authority is inconsistent")
        active_counts = torch.bincount(
            step_tensors["row_env_slot"][step_tensors["row_residual_active"]],
            minlength=spec.num_envs,
        )
        if not torch.equal(active_counts, intervention_steps.cpu()):
            raise RuntimeError("intervention_steps disagrees with fixed step rows")
        if bool((active_counts[~spec.treatment] != 0).any()) or bool(
            (active_counts > spec.window_steps).any()
        ):
            raise RuntimeError("fixed intervention escaped treatment/window")

        source_after = source_fingerprints(spec.repository_root)
        runtime_assets_after = runtime_asset_fingerprints(spec.repository_root)
        git_after = git_provenance(spec.repository_root, git_paths)
        if (
            source_after != source_before
            or runtime_assets_after != runtime_assets_before
            or git_after != git_before
            or _current_hashes(spec) != spec.immutable_sha256
        ):
            raise RuntimeError(
                "fixed-validation source, runtime asset, or immutable input changed"
            )
        v6_hashes = spec.immutable_sha256["v6"]
        if not isinstance(v6_hashes, Mapping):
            raise TypeError("V6 checkpoint hash manifest is not a mapping")
        metadata = {
            **artifact_contract.REQUIRED_METADATA,
            "seed": spec.seed,
            "replicate": spec.replicate,
            "num_envs": spec.num_envs,
            "assignment_mask_sha256": artifact_contract.assignment_mask_sha256(
                spec.treatment
            ),
            # Artifacts bind the canonical plan strings.  The resolved absolute
            # paths remain in ``spec`` for loading and immutable byte checks.
            "v6_checkpoint": artifact_contract.V6_CHECKPOINT_PATH,
            "search_checkpoint": artifact_contract.SEARCH_CHECKPOINT_PATH,
            "validation_plan": artifact_contract.VALIDATION_PLAN,
            "validation_plan_sha256": spec.immutable_sha256["validation_plan"],
            "fixed_direction_manifest": artifact_contract.FIXED_DIRECTION_MANIFEST,
            "fixed_direction_manifest_sha256": spec.immutable_sha256[
                "fixed_direction_manifest"
            ],
            "fixed_direction_path": artifact_contract.FIXED_DIRECTION_PATH,
            "fixed_direction_sha256": spec.immutable_sha256["fixed_direction"],
            "v6_actor_sha256": v6_hashes["actor.pt"],
            "v6_task_contract_sha256": v6_hashes["task_contract.json"],
            "v6_bridge_state_sha256": v6_hashes["torch_bridge_state.pt"],
            "frozen_lift_actor_sha256": v6_hashes["frozen_lift_actor.pt"],
            "frozen_lift_semantic_sha256": load["frozen_semantic_sha256"],
            "frozen_lift_source_actor_sha256": load[
                "frozen_source_actor_sha256"
            ],
            "search_checkpoint_sha256": spec.immutable_sha256["search"],
            "source_manifest_sha256": artifact_contract.manifest_sha256(
                source_before
            ),
            "runtime_asset_manifest_sha256": artifact_contract.manifest_sha256(
                runtime_assets_before
            ),
            "source_sha256": source_before,
            "runtime_asset_sha256": runtime_assets_before,
            "flashsac_upstream_commit": load["upstream_commit"],
            "flashsac_fork_commit": load["fork_commit"],
            "git": git_before,
            "runtime": runtime_provenance(seed=spec.seed, device=env.device),
        }
        artifact = artifact_contract.build_artifact(
            metadata=metadata, episodes=episode_tensors, steps=step_tensors
        )
        artifact_contract.validate_artifact(artifact)
        report = {
            "kind": artifact_contract.REPORT_KIND,
            "status": "complete",
            "collector": artifact_contract.COLLECTOR,
            "seed": spec.seed,
            "replicate": spec.replicate,
            "num_envs": spec.num_envs,
            "vector_steps": vector_steps,
            "v6_actor_sha256": metadata["v6_actor_sha256"],
            "search_checkpoint_sha256": metadata["search_checkpoint_sha256"],
            "fixed_direction_sha256": metadata["fixed_direction_sha256"],
            "summary": artifact_contract.summarize_artifact(artifact),
        }
        artifact_contract.validate_report(report, artifact)
        return artifact, report
    finally:
        env.close()


def parse_args() -> tuple[argparse.Namespace, CollectionSpec, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--v6_checkpoint", type=Path, required=True)
    parser.add_argument("--search_checkpoint", type=Path, required=True)
    parser.add_argument(
        "--fixed_direction",
        type=Path,
        default=Path(artifact_contract.FIXED_DIRECTION_PATH),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--replicate", choices=("a", "b"), required=True)
    parser.add_argument("--num_envs", type=int, default=NUM_ENVS)
    parser.add_argument("--window_steps", type=int, default=WINDOW_STEPS)
    parser.add_argument("--token_scale", type=float, default=TOKEN_SCALE)
    parser.add_argument("--distal_scale", type=float, default=DISTAL_SCALE)
    parser.add_argument("--raw_z_abs_cap", type=float, default=RAW_Z_ABS_CAP)
    parser.add_argument(
        "--token_component_cap", type=float, default=TOKEN_COMPONENT_CAP
    )
    parser.add_argument(
        "--distal_component_cap", type=float, default=DISTAL_COMPONENT_CAP
    )
    parser.add_argument("--pre_tanh_l2_cap", type=float, default=PRE_TANH_L2_CAP)
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
    except (
        TypeError,
        ValueError,
        FileNotFoundError,
        FileExistsError,
    ) as error:
        parser.error(str(error))
    launcher = AppLauncher(args)
    return args, spec, launcher.app


def publish_failure_attempt(spec: CollectionSpec, error: BaseException) -> Path:
    payload = {
        "kind": artifact_contract.REPORT_KIND,
        "status": "failed",
        "seed": spec.seed,
        "replicate": spec.replicate,
        "num_envs": spec.num_envs,
        "fixed_direction_sha256": artifact_contract.FIXED_DIRECTION_SHA256,
        "canonical_artifact_output": str(spec.artifact_output),
        "canonical_report_output": str(spec.report_output),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    for attempt in range(1, 10_000):
        output = Path(f"{spec.output_stem}.failed_attempt_{attempt:03d}.json")
        try:
            artifact_contract.publish_json_no_clobber(payload, output)
        except FileExistsError:
            continue
        return output
    raise RuntimeError("fixed-validation failure-attempt namespace is exhausted")


def main() -> None:
    args, spec, simulation_app = parse_args()
    try:
        try:
            artifact, report = run_collection(
                spec, device_string=str(args.device or "cuda:0")
            )
            digest = artifact_contract.publish_artifact_and_report_no_clobber(
                artifact,
                report,
                artifact_output=spec.artifact_output,
                report_output=spec.report_output,
            )
            print(
                "[candidate39-fixed-residual-ab] "
                f"seed={spec.seed} replicate={spec.replicate} "
                f"envs={spec.num_envs} rows={report['summary']['step_rows']} "
                f"sha256={digest}",
                flush=True,
            )
        except BaseException as error:
            failure = publish_failure_attempt(spec, error)
            print(
                f"[candidate39-fixed-residual-ab] failure recorded at {failure}",
                flush=True,
            )
            raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
