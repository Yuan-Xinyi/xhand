#!/usr/bin/env python3
"""Collect one same-process randomized baseline-versus-candidate CLOSE trial.

Every environment executes the same frozen SEARCH policy until the public
q=0.30, four-frame handoff.  A balanced assignment fixed before simulator task
imports then chooses one deterministic CLOSE actor.  Both arms use the exact
same frozen LIFT actor after the public grasp latch.  Only each slot's first
episode is retained.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
from importlib import metadata as importlib_metadata
import math
import os
from pathlib import Path
import platform
import subprocess
import traceback
from typing import Any, Mapping

import torch

from online_close_ab import (
    ACTION_DIM,
    ARTIFACT_KIND,
    ASSIGNMENT_CONTRACT,
    ASSIGNMENT_SALT,
    COLLECTION_CONTRACT,
    FORMAT_VERSION,
    HANDOFF_HOLD_STEPS,
    HANDOFF_MIN_SCORE,
    KIT_ARGS,
    MAX_EPISODE_ACTIONS,
    OBSERVATION_DIM,
    REPORT_KIND,
    RUNTIME_ASSET_NAMES,
    assignment_candidate_mask,
    build_artifact,
    publish_artifact_and_report_no_clobber,
    publish_json_no_clobber,
    select_executed_action,
    sha256_file,
    summarize_artifact,
)
from online_handoff import update_online_handoff


SOURCE_FILES = (
    "scripts/flashsac/collect_online_close_ab.py",
    "scripts/flashsac/online_close_ab.py",
    "scripts/flashsac/online_handoff.py",
    "scripts/flashsac/agent_bridge.py",
    "scripts/flashsac/adapter.py",
    "scripts/flashsac/evaluate.py",
    "scripts/flashsac/train.py",
    "scripts/rl_games/bc_pick_tool.py",
    "source/xhand_inhand/xhand_inhand/__init__.py",
    "source/xhand_inhand/xhand_inhand/tasks/__init__.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/__init__.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_cube/__init__.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_cube_token/__init__.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/__init__.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/public_gate_state.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/pick_tool_token_env.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/pick_tool_token_env_cfg.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/grasp_signals.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/hybrid_action.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/tool_asset.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/textured_mesh.obj",
)
RUNTIME_ASSET_EXPECTED_SHA256 = {
    "/tmp/xhand_inhand/pick_tool_token/.asset_hash": (
        "3341d37631e4617824f4ccfd71bbdcd2a23feafd00c597458dec592050787706"
    ),
    "/tmp/xhand_inhand/pick_tool_token/Props/instanceable_meshes.usd": (
        "66b81c836ba3f2df9cd0b1b29bc4dd0080ad78604147daa374b923a8f082b126"
    ),
    "/tmp/xhand_inhand/pick_tool_token/tool_hammer.usd": (
        "e176f0aa83f6978334e89451a7d19c16c85bb83cb6f09d8aff394445960117f2"
    ),
    "source/xhand_inhand/xhand_inhand/assets/xarm7_xhand/configuration/xarm7_xhand_base.usd": (
        "96c1dc55b920dd06360da7e4f5347a8dfd37d9e99aceaff80157c1e4893e2e01"
    ),
    "source/xhand_inhand/xhand_inhand/assets/xarm7_xhand/configuration/xarm7_xhand_physics.usd": (
        "7bc6170eedce5219a943a2b79161dbf572338beadfe9d78cfa6a183fdcc6dd50"
    ),
    "source/xhand_inhand/xhand_inhand/assets/xarm7_xhand/configuration/xarm7_xhand_robot.usd": (
        "e76cd219931d54ac841de1bf166284f8d14951c1696e7735d98eadf4cef9c3f2"
    ),
    "source/xhand_inhand/xhand_inhand/assets/xarm7_xhand/configuration/xarm7_xhand_sensor.usd": (
        "332110c8bd4afd45ecab6289b46640a4dd371b4b47500905260434c41c6139ae"
    ),
    "source/xhand_inhand/xhand_inhand/assets/xarm7_xhand/xarm7_xhand.usd": (
        "c361d7879735cf31107f0089a68bed014b1401eda4d340a14a8ae4630a9d756c"
    ),
}
CHECKPOINT_FILES = (
    "actor.pt",
    "task_contract.json",
    "frozen_lift_actor.pt",
    "torch_bridge_state.pt",
)


@dataclass(frozen=True)
class CollectionSpec:
    repository_root: Path
    baseline_checkpoint: Path
    candidate_checkpoint: Path
    search_checkpoint: Path
    seed: int
    replicate: str
    num_envs: int
    kit_args: str
    output_stem: Path
    artifact_output: Path
    report_output: Path
    checkpoint_sha256: Mapping[str, Mapping[str, str] | str]
    assignment_candidate: torch.Tensor


def _path_owned(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _checkpoint_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise FileNotFoundError(f"{label} must be a real checkpoint directory: {path}")
    resolved = path.resolve()
    for filename in CHECKPOINT_FILES:
        _regular_file(resolved / filename, label=f"{label}/{filename}")
    return resolved


def _checkpoint_hashes(path: Path) -> dict[str, str]:
    return {filename: sha256_file(path / filename) for filename in CHECKPOINT_FILES}


def build_spec(args: argparse.Namespace) -> CollectionSpec:
    root = Path(__file__).resolve().parents[2]
    if isinstance(args.seed, bool):
        raise TypeError("seed must be an integer")
    if args.num_envs < 2 or args.num_envs % 2:
        raise ValueError("num_envs must be at least two and even")
    if getattr(args, "kit_args", None) != KIT_ARGS:
        raise ValueError(f"A/B collection requires exact --kit_args={KIT_ARGS!r}")
    baseline = _checkpoint_directory(args.baseline_checkpoint, label="baseline")
    candidate = _checkpoint_directory(args.candidate_checkpoint, label="candidate")
    if baseline == candidate:
        raise ValueError("baseline and candidate checkpoints must differ")
    search = _regular_file(args.search_checkpoint, label="SEARCH checkpoint")
    output_stem = Path(os.path.abspath(os.fspath(args.output_stem)))
    artifact_output = Path(f"{output_stem}.pt")
    report_output = Path(f"{output_stem}.json")
    for output in (artifact_output, report_output):
        if _path_owned(output):
            raise FileExistsError(f"A/B evidence output already exists: {output}")
    baseline_hashes = _checkpoint_hashes(baseline)
    candidate_hashes = _checkpoint_hashes(candidate)
    if baseline_hashes["actor.pt"] == candidate_hashes["actor.pt"]:
        raise ValueError("baseline and candidate actor bytes are identical")
    if baseline_hashes["task_contract.json"] != candidate_hashes[
        "task_contract.json"
    ]:
        raise ValueError("baseline and candidate task contracts differ")
    if baseline_hashes["frozen_lift_actor.pt"] != candidate_hashes[
        "frozen_lift_actor.pt"
    ]:
        raise ValueError("baseline and candidate frozen LIFT bytes differ")
    return CollectionSpec(
        repository_root=root,
        baseline_checkpoint=baseline,
        candidate_checkpoint=candidate,
        search_checkpoint=search,
        seed=int(args.seed),
        replicate=str(args.replicate),
        num_envs=int(args.num_envs),
        kit_args=KIT_ARGS,
        output_stem=output_stem,
        artifact_output=artifact_output,
        report_output=report_output,
        checkpoint_sha256={
            "baseline": baseline_hashes,
            "candidate": candidate_hashes,
            "search": sha256_file(search),
        },
        # Materialize treatment before AppLauncher starts Isaac.  Task imports,
        # asset generation, and reset state therefore cannot influence it.
        assignment_candidate=assignment_candidate_mask(
            seed=int(args.seed),
            num_envs=int(args.num_envs),
            replicate=str(args.replicate),
        ),
    )


def source_fingerprints(root: Path) -> dict[str, str]:
    """Authenticate all code, retargeting, task, mesh, and generated asset bytes."""

    from evaluate import diagnostic_handoff_source_fingerprints
    from search_replay_trace import source_fingerprints as trace_source_fingerprints

    result = trace_source_fingerprints(root)
    diagnostic = diagnostic_handoff_source_fingerprints(root)
    for relative, digest in diagnostic.items():
        if relative in result and result[relative] != digest:
            raise RuntimeError(f"source fingerprint helpers disagree for {relative}")
        result[relative] = digest
    for relative in SOURCE_FILES:
        path = _regular_file(root / relative, label=f"source {relative}")
        result[relative] = sha256_file(path)
    return dict(sorted(result.items()))


def runtime_asset_fingerprints(root: Path) -> dict[str, str]:
    """Bind the exact generated hammer and composed robot USD inputs."""

    if set(RUNTIME_ASSET_EXPECTED_SHA256) != set(RUNTIME_ASSET_NAMES):
        raise RuntimeError("collector and artifact runtime-asset contracts disagree")
    result: dict[str, str] = {}
    for declared, expected in RUNTIME_ASSET_EXPECTED_SHA256.items():
        declared_path = Path(declared)
        path = declared_path if declared_path.is_absolute() else root / declared_path
        path = _regular_file(path, label=f"runtime asset {declared}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"runtime asset SHA256 changed: {declared}")
        result[declared] = actual
    return result


def git_provenance(root: Path, source_paths: tuple[str, ...]) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()

    flash_root = root / "third_party/FlashSAC"
    flash_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=flash_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    flash_dirty = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=flash_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "source_files_dirty": bool(run("status", "--porcelain", "--", *source_paths)),
        "flashsac_commit": flash_commit,
        "flashsac_dirty": bool(flash_dirty),
    }


def runtime_provenance(*, seed: int, device: torch.device) -> dict[str, Any]:
    from isaaclab.utils.version import get_isaac_sim_version

    packages: dict[str, str] = {}
    for name in (
        "isaaclab",
        "isaaclab_tasks",
        "isaaclab_assets",
        "numpy",
        "gymnasium",
    ):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    try:
        inventory = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,driver_version",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        inventory = f"unavailable:{type(error).__name__}"
    if device.type != "cuda":
        raise ValueError("runtime provenance requires the actual CUDA device")
    index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "cudnn": int(torch.backends.cudnn.version() or 0),
        "cuda_device_index": int(index),
        "cuda_device_name": torch.cuda.get_device_name(index),
        "cuda_device_capability": list(torch.cuda.get_device_capability(index)),
        "isaac_sim": str(get_isaac_sim_version()),
        "nvidia_smi_inventory": inventory,
        "platform": platform.platform(),
        "packages": packages,
        "seed": seed,
    }


def _validate_live_task(env: Any) -> None:
    cfg = env.unwrapped.cfg
    modes = {
        "close_option_mode": bool(cfg.close_option_mode),
        "power_close_option_mode": bool(cfg.power_close_option_mode),
        "coupled_power_align_close_option_mode": bool(
            cfg.coupled_power_align_close_option_mode
        ),
        "hold_arm_until_stable_grasp": bool(cfg.hold_arm_until_stable_grasp),
    }
    if any(modes.values()):
        raise RuntimeError(f"A/B collector requires the ordinary full task: {modes}")
    integer_checks = {
        "observation_space": (int(cfg.observation_space), OBSERVATION_DIM),
        "action_space": (int(cfg.action_space), ACTION_DIM),
        "max_episode_steps": (int(env.max_episode_steps), 1000),
        "hard_force_terminate_steps": (int(cfg.tactile_hard_terminate_steps), 10),
        "grasp_confirm_steps": (int(cfg.grasp_confirm_steps), 4),
        "grasp_release_steps": (int(cfg.grasp_release_steps), 6),
        "overforce_terminate_steps": (int(cfg.tactile_terminate_steps), 2),
        "success_hold_steps": (int(cfg.success_hold_steps), 15),
    }
    wrong = {key: value for key, value in integer_checks.items() if value[0] != value[1]}
    if wrong:
        raise RuntimeError(f"live task integer contract changed: {wrong}")
    float_checks = {
        "episode_length_s": (float(cfg.episode_length_s), 20.0),
        "hard_force_limit_n": (float(cfg.tactile_hard_force_limit), 30.0),
        "overforce_limit_n": (float(cfg.tactile_terminate_force_limit), 60.0),
        "success_true_clearance_m": (float(cfg.lift_success_height), 0.2),
        "curriculum_reset_probability": (
            float(cfg.curriculum_reset_probability),
            0.0,
        ),
        "curriculum_joint_noise": (float(cfg.curriculum_joint_noise), 0.0),
    }
    wrong_float = {
        key: value
        for key, value in float_checks.items()
        if not math.isclose(value[0], value[1], rel_tol=0.0, abs_tol=1.0e-9)
    }
    if wrong_float or str(cfg.curriculum_dataset) != "":
        raise RuntimeError(
            f"live task float/curriculum contract changed: {wrong_float}"
        )


def _load_routed_agent(env: Any, checkpoint: Path, *, seed: int) -> tuple[Any, dict[str, Any]]:
    from train import read_checkpoint_task_contract
    from agent_bridge import (
        FLASH_SAC_COMMIT,
        FLASH_SAC_FORK_COMMIT,
        ActionAuthorityRule,
        ActionNoiseGroup,
        FlashSACTorchBridge,
        PublicLatchFrozenActorRouter,
        build_agent_config,
    )
    from evaluate import (
        ARM_ACTION_DIM,
        FULL_TASK_GRASP_LATCH_OBSERVATION_INDEX,
        NOISE_GROUP_SPECS,
        PRODUCTION_ACTOR_BLOCKS,
        PRODUCTION_ACTOR_HIDDEN,
        PRODUCTION_CRITIC_BINS,
        PRODUCTION_CRITIC_BLOCKS,
        PRODUCTION_CRITIC_HIDDEN,
        infer_checkpoint_actor_action_dim,
        infer_checkpoint_architecture,
        validate_checkpoint_evaluation_contract,
    )

    contract = read_checkpoint_task_contract(checkpoint)
    if contract.get("version") != 6 or contract.get("task_mode") != "full_task":
        raise ValueError("A/B checkpoint must use the authored full-task V6 contract")
    router = contract.get("policy_router")
    if not isinstance(router, Mapping) or router.get("kind") != "public_latch_frozen_actor_v1":
        raise ValueError("A/B checkpoint lacks the checkpoint-native router")
    actor_dim = infer_checkpoint_actor_action_dim(checkpoint)
    source_contract, target_contract, source_indices = (
        validate_checkpoint_evaluation_contract(
            checkpoint_task_mode="full_task",
            checkpoint_contract=contract,
            requested_task_mode="full_task",
            actor_action_dim=actor_dim,
        )
    )
    if source_indices is not None or source_contract != target_contract:
        raise RuntimeError("A/B actor load must be an exact full-task load")
    if infer_checkpoint_architecture(
        checkpoint,
        expected_action_dim=ACTION_DIM,
        expected_observation_dim=OBSERVATION_DIM,
    ) != "production":
        raise RuntimeError("A/B actor does not use the production architecture")
    authority = [
        {
            "name": "arm_after_public_latch",
            "start": 0,
            "stop": ARM_ACTION_DIM,
            "observation_index": FULL_TASK_GRASP_LATCH_OBSERVATION_INDEX,
            "active_value": 1.0,
        }
    ]
    if contract.get("policy_action_authority") != authority:
        raise ValueError("A/B checkpoint action authority changed")
    config = build_agent_config(
        seed=seed,
        normalize_reward=True,
        normalized_G_max=5.0,
        device_type=str(env.device),
        buffer_device_type=str(env.device),
        buffer_max_length=max(env.num_envs, 32),
        buffer_min_length=1,
        sample_batch_size=1,
        n_step=3,
        actor_num_blocks=PRODUCTION_ACTOR_BLOCKS,
        actor_hidden_dim=PRODUCTION_ACTOR_HIDDEN,
        critic_num_blocks=PRODUCTION_CRITIC_BLOCKS,
        critic_hidden_dim=PRODUCTION_CRITIC_HIDDEN,
        critic_num_bins=PRODUCTION_CRITIC_BINS,
        use_compile=False,
        use_amp=False,
        load_optimizer=False,
        load_reward_normalizer=False,
    )
    noise_groups = tuple(
        ActionNoiseGroup(
            name,
            start,
            stop,
            scale=scale,
            zeta_mu=zeta_mu,
            zeta_max=zeta_max,
        )
        for name, start, stop, scale, zeta_mu, zeta_max in NOISE_GROUP_SPECS
    )
    agent = FlashSACTorchBridge(
        env.observation_space,
        env.action_space,
        env.env_info,
        config,
        noise_groups=noise_groups,
        restore_rng_state_on_load=False,
        action_authority_rules=tuple(ActionAuthorityRule(**rule) for rule in authority),
        public_latch_frozen_actor_router=PublicLatchFrozenActorRouter(
            name=str(router["kind"]),
            observation_index=int(router["observation_index"]),
            trainable_start=int(router["trainable_action_slice"][0]),
            trainable_stop=int(router["trainable_action_slice"][1]),
            close_value=float(router["close_value"]),
            frozen_value=float(router["frozen_value"]),
        ),
        unit_normalize_actor_mean_head=True,
    )
    agent.load_actor(str(checkpoint))
    agent.load_frozen_lift_actor_sidecar(str(checkpoint))
    frozen = router.get("frozen_actor")
    if not isinstance(frozen, Mapping) or (
        agent.frozen_lift_actor_sha256 != frozen.get("network_sha256")
        or agent.frozen_lift_actor_source_sha256 != frozen.get("source_actor_sha256")
    ):
        raise RuntimeError("A/B frozen LIFT lineage is invalid")
    agent.reset_exploration(batch_size=env.num_envs)
    return agent, {
        "upstream_commit": FLASH_SAC_COMMIT,
        "fork_commit": FLASH_SAC_FORK_COMMIT,
        "frozen_semantic_sha256": agent.frozen_lift_actor_sha256,
        "frozen_source_actor_sha256": agent.frozen_lift_actor_source_sha256,
        "task_contract": contract,
    }


def _current_hashes(spec: CollectionSpec) -> dict[str, Mapping[str, str] | str]:
    return {
        "baseline": _checkpoint_hashes(spec.baseline_checkpoint),
        "candidate": _checkpoint_hashes(spec.candidate_checkpoint),
        "search": sha256_file(spec.search_checkpoint),
    }


@torch.inference_mode()
def run_collection(spec: CollectionSpec, *, device_string: str) -> tuple[dict[str, Any], dict[str, Any]]:
    device = torch.device(device_string)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"online CLOSE A/B requires CUDA Isaac physics, got {device}")
    assignment_cpu = assignment_candidate_mask(
        seed=spec.seed, num_envs=spec.num_envs, replicate=spec.replicate
    )
    if not torch.equal(assignment_cpu, spec.assignment_candidate):
        raise RuntimeError("pre-launch A/B assignment changed")

    # Treatment is fixed before task registration and simulator state creation.
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
    # Task registration materializes the content-addressed hammer USD in /tmp.
    # Assignment was already fixed in build_spec, so this cannot influence it.
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
        raise RuntimeError("A/B source and FlashSAC trees must be committed and clean")
    env = make_pick_tool_env(
        num_envs=spec.num_envs,
        device=device_string,
        seed=spec.seed,
        strict=True,
        validate_finite=True,
    )
    try:
        _validate_live_task(env)
        baseline_agent, baseline_load = _load_routed_agent(
            env, spec.baseline_checkpoint, seed=spec.seed
        )
        candidate_agent, candidate_load = _load_routed_agent(
            env, spec.candidate_checkpoint, seed=spec.seed
        )
        if (
            baseline_load["task_contract"] != candidate_load["task_contract"]
            or baseline_load["upstream_commit"] != candidate_load["upstream_commit"]
            or baseline_load["fork_commit"] != candidate_load["fork_commit"]
            or baseline_load["frozen_semantic_sha256"]
            != candidate_load["frozen_semantic_sha256"]
            or baseline_load["frozen_source_actor_sha256"]
            != candidate_load["frozen_source_actor_sha256"]
        ):
            raise RuntimeError("A/B policy router or frozen LIFT semantics differ")
        if git_before["flashsac_commit"] != baseline_load["fork_commit"]:
            raise RuntimeError("loaded FlashSAC fork differs from the submodule HEAD")
        search_actor = _load_diagnostic_approach_actor(
            spec.search_checkpoint, device=device
        ).eval()
        if _current_hashes(spec) != spec.checkpoint_sha256:
            raise RuntimeError("an A/B checkpoint changed while loading")

        observation, _ = env.reset(seed=spec.seed, randomize_episode_lengths=False)
        if observation.shape != (spec.num_envs, OBSERVATION_DIM):
            raise RuntimeError("A/B reset violated the obs115 contract")
        initial_truth = _read_physical_truth(env.unwrapped, task_mode=FULL_TASK_MODE)
        tracker = StrictEpisodeTracker(
            episodes=spec.num_envs,
            num_envs=spec.num_envs,
            device=env.device,
            initial_truth=initial_truth,
            task_mode=FULL_TASK_MODE,
        )
        assignment = assignment_cpu.to(device=env.device)
        ready_count = torch.zeros(spec.num_envs, dtype=torch.long, device=env.device)
        option_active = torch.zeros(spec.num_envs, dtype=torch.bool, device=env.device)
        triggered = torch.zeros_like(option_active)
        trigger_step = torch.full(
            (spec.num_envs,), -1, dtype=torch.long, device=env.device
        )
        trigger_score = torch.zeros(
            spec.num_envs, dtype=torch.float32, device=env.device
        )
        episode_step = torch.zeros_like(trigger_step)
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
                raise RuntimeError("an environment triggered the sticky handoff twice")
            triggered |= trigger
            trigger_step = torch.where(trigger, episode_step, trigger_step)
            trigger_score = torch.where(
                trigger, handoff["score"].to(dtype=torch.float32), trigger_score
            )
            ready_count = handoff["ready_count_after"]
            option_active = handoff["option_active_after"]

            with torch.no_grad():
                search_action = search_actor(observation).clamp(-1.0, 1.0)
            baseline_action = baseline_agent.sample_actions(
                vector_steps + 1,
                {"next_observation": observation},
                training=False,
            )
            candidate_action = candidate_agent.sample_actions(
                vector_steps + 1,
                {"next_observation": observation},
                training=False,
            )
            latch = observation[:, 106]
            if not bool(((latch == 0.0) | (latch == 1.0)).all()):
                raise RuntimeError("public grasp latch is not binary")
            action = select_executed_action(
                search_action=search_action,
                baseline_action=baseline_action,
                candidate_action=candidate_action,
                option_active=option_active,
                assignment_candidate=assignment,
                public_latch=latch == 1.0,
            )
            action = torch.where(
                active_before.unsqueeze(-1), action, torch.zeros_like(action)
            )
            next_observation, reward, terminated, truncated, info = env.step(action)
            vector_steps += 1
            executed = env.last_executed_policy_action
            if executed is None or not torch.equal(executed, action):
                raise RuntimeError("adapter executed a different A/B action")
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
            post_reset_truth = _read_physical_truth(
                env.unwrapped, task_mode=FULL_TASK_MODE
            )
            tracker.step(
                reward=reward.to(dtype=torch.float32),
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
                f"A/B trial completed {len(tracker.records)}/{spec.num_envs} first episodes"
            )
        records_by_slot = {int(record["env_slot"]): record for record in tracker.records}
        if set(records_by_slot) != set(range(spec.num_envs)):
            raise RuntimeError("A/B tracker did not produce one record per env slot")
        ordered = [records_by_slot[index] for index in range(spec.num_envs)]
        if any(int(record["slot_episode_index"]) != 0 for record in ordered):
            raise RuntimeError("A/B evidence contains a post-reset episode")
        tensors = {
            "env_slot": torch.arange(spec.num_envs, dtype=torch.long),
            "assignment_candidate": assignment_cpu,
            "triggered": triggered.cpu(),
            "trigger_step": trigger_step.cpu(),
            "trigger_score": trigger_score.cpu(),
            "episode_length": torch.tensor(
                [int(record["length"]) for record in ordered], dtype=torch.long
            ),
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
                [bool(record["unsafe_force"]) for record in ordered], dtype=torch.bool
            ),
            "unlatched_clearance_ge_5cm": torch.tensor(
                [
                    bool(record["ever_unlatched_clearance_ge_5cm"])
                    for record in ordered
                ],
                dtype=torch.bool,
            ),
            "ever_grasped": torch.tensor(
                [bool(record["ever_grasped"]) for record in ordered], dtype=torch.bool
            ),
            "ever_clearance_ge_20cm": torch.tensor(
                [bool(record["ever_clearance_ge_20cm"]) for record in ordered],
                dtype=torch.bool,
            ),
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
            raise RuntimeError("A/B source or checkpoint changed during collection")
        metadata = {
            "kind": ARTIFACT_KIND,
            "format_version": FORMAT_VERSION,
            "collection_contract": COLLECTION_CONTRACT,
            "assignment_contract": ASSIGNMENT_CONTRACT,
            "assignment_salt": ASSIGNMENT_SALT,
            "handoff_min_score": HANDOFF_MIN_SCORE,
            "handoff_hold_steps": HANDOFF_HOLD_STEPS,
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "max_episode_actions": MAX_EPISODE_ACTIONS,
            "kit_args": spec.kit_args,
            "seed": spec.seed,
            "replicate": spec.replicate,
            "num_envs": spec.num_envs,
            "baseline_checkpoint": str(spec.baseline_checkpoint),
            "candidate_checkpoint": str(spec.candidate_checkpoint),
            "search_checkpoint": str(spec.search_checkpoint),
            "baseline_actor_sha256": spec.checkpoint_sha256["baseline"]["actor.pt"],
            "candidate_actor_sha256": spec.checkpoint_sha256["candidate"]["actor.pt"],
            "baseline_task_contract_sha256": spec.checkpoint_sha256["baseline"][
                "task_contract.json"
            ],
            "candidate_task_contract_sha256": spec.checkpoint_sha256["candidate"][
                "task_contract.json"
            ],
            "common_frozen_lift_actor_sha256": spec.checkpoint_sha256["baseline"][
                "frozen_lift_actor.pt"
            ],
            "common_frozen_lift_semantic_sha256": baseline_load[
                "frozen_semantic_sha256"
            ],
            "common_frozen_lift_source_actor_sha256": baseline_load[
                "frozen_source_actor_sha256"
            ],
            "search_checkpoint_sha256": spec.checkpoint_sha256["search"],
            "baseline_bridge_state_sha256": spec.checkpoint_sha256["baseline"][
                "torch_bridge_state.pt"
            ],
            "candidate_bridge_state_sha256": spec.checkpoint_sha256["candidate"][
                "torch_bridge_state.pt"
            ],
            "flashsac_upstream_commit": baseline_load["upstream_commit"],
            "flashsac_fork_commit": baseline_load["fork_commit"],
            "source_sha256": source_before,
            "runtime_asset_sha256": runtime_assets_before,
            "git": git_before,
            "runtime": runtime_provenance(seed=spec.seed, device=env.device),
        }
        artifact = build_artifact(metadata=metadata, tensors=tensors)
        summary = summarize_artifact(artifact)
        report = {
            "kind": REPORT_KIND,
            "status": "complete",
            "seed": spec.seed,
            "replicate": spec.replicate,
            "num_envs": spec.num_envs,
            "vector_steps": vector_steps,
            "collection_contract": COLLECTION_CONTRACT,
            "baseline_actor_sha256": metadata["baseline_actor_sha256"],
            "candidate_actor_sha256": metadata["candidate_actor_sha256"],
            "common_frozen_lift_actor_sha256": metadata[
                "common_frozen_lift_actor_sha256"
            ],
            "search_checkpoint_sha256": metadata["search_checkpoint_sha256"],
            "summary": summary,
        }
        return artifact, report
    finally:
        env.close()


def parse_args() -> tuple[argparse.Namespace, CollectionSpec, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--baseline_checkpoint", type=Path, required=True)
    parser.add_argument("--candidate_checkpoint", type=Path, required=True)
    parser.add_argument("--search_checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--replicate", choices=("a", "b"), required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument(
        "--output_stem",
        type=Path,
        required=True,
        help="immutable output stem; .pt and .json are derived together",
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
    """Write an append-only sibling failure record without consuming canonical paths."""

    payload = {
        "kind": REPORT_KIND,
        "status": "failed",
        "seed": spec.seed,
        "replicate": spec.replicate,
        "num_envs": spec.num_envs,
        "canonical_artifact_output": str(spec.artifact_output),
        "canonical_report_output": str(spec.report_output),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    for attempt in range(1, 10_000):
        output = Path(f"{spec.output_stem}.failed_attempt_{attempt:03d}.json")
        try:
            publish_json_no_clobber(payload, output)
        except FileExistsError:
            continue
        return output
    raise RuntimeError("A/B failure-attempt namespace is exhausted")


def main() -> None:
    args, spec, simulation_app = parse_args()
    try:
        try:
            artifact, report = run_collection(
                spec, device_string=str(args.device or "cuda:0")
            )
            artifact_sha = publish_artifact_and_report_no_clobber(
                artifact,
                report,
                artifact_output=spec.artifact_output,
                report_output=spec.report_output,
            )
            print(
                "[online-close-ab] "
                f"seed={spec.seed} replicate={spec.replicate} "
                f"envs={spec.num_envs} sha256={artifact_sha}",
                flush=True,
            )
        except BaseException as error:
            failure_output = publish_failure_attempt(spec, error)
            print(
                f"[online-close-ab] failure recorded at {failure_output}",
                flush=True,
            )
            raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
