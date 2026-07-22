#!/usr/bin/env python3
"""Record one deterministic full-task SEARCH rollout for replay A/A checks.

This deliberately does not clone state between Isaac environment slots.  A
replay comparison is made by running this program twice with the same seed and
then comparing the same slot across the two independent simulator processes.
That is the only tested layout that avoids slot-specific PhysX contact state.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Mapping, Sequence

import torch


TRACE_KIND = "pick_tool_search_replay_trace_v1"
REPORT_KIND = "pick_tool_search_replay_trace_report_v1"
OBSERVATION_DIM = 115
ACTION_DIM = 21
TERMINAL_NAMES = (
    "success",
    "failure",
    "time_out",
    "dropped",
    "unsafe_force",
    "unlatched_clearance_ge_5cm",
)
SOURCE_FILES = (
    "scripts/flashsac/search_replay_trace.py",
    "scripts/flashsac/adapter.py",
    "scripts/flashsac/evaluate.py",
    "scripts/rl_games/bc_pick_tool.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/pick_tool_token_env.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/pick_tool_token_env_cfg.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/grasp_signals.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/hybrid_action.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/tool_asset.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/textured_mesh.obj",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/material.mtl",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_cube/pick_cube_env.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_cube/pick_cube_env_cfg.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_cube_token/pick_cube_token_env.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_cube_token/pick_cube_token_env_cfg.py",
    "source/xhand_inhand/xhand_inhand/tasks/direct/pick_cube_token/retarget_infer.py",
    "source/xhand_inhand/xhand_inhand/robots/xarm7_xhand.py",
    "tools/crossdex_retarget/models/retarget_nn_xhand.pt",
    "tools/crossdex_retarget/models/retarget_nn_xhand_meta.pkl",
)
SOURCE_TREE_DIRS = (
    # The composed robot USD sublayers and meshes are all simulator inputs.
    "source/xhand_inhand/xhand_inhand/assets/xarm7_xhand",
)
POLICY_CONTRACT = "frozen_rlgames_search_clamped_deterministic_mean_v1"
READINESS_CONTRACT = (
    "g=min(obs[96],second_largest(obs[92:96]));"
    "eligible=(obs[106]==0 and g>=0.10);consecutive_cap4;sticky_once"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def source_fingerprints(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"trace source must be a regular file: {path}")
        result[relative] = sha256_file(path)
    for relative_directory in SOURCE_TREE_DIRS:
        directory = root / relative_directory
        if not directory.is_dir() or directory.is_symlink():
            raise FileNotFoundError(f"trace source tree must be a real directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise RuntimeError(f"trace source tree contains a symlink: {path}")
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                result[relative] = sha256_file(path)
    return result


def git_provenance(root: Path, source_paths: Sequence[str]) -> dict[str, Any]:
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

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "source_files_dirty": bool(run("status", "--porcelain", "--", *source_paths)),
    }


def _strict_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _path_is_owned(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def publish_trace_and_report_no_clobber(
    trace: Mapping[str, Any],
    report: Mapping[str, Any],
    trace_output: Path,
    report_output: Path,
) -> str:
    """Transactionally publish immutable PT and JSON files via hard links."""

    # abspath preserves a dangling final symlink so it remains an owned path;
    # Path.resolve() would incorrectly follow it and weaken no-clobber.
    trace_output = Path(os.path.abspath(os.fspath(trace_output)))
    report_output = Path(os.path.abspath(os.fspath(report_output)))
    if trace_output == report_output:
        raise ValueError("trace and report outputs must be different paths")
    for path in (trace_output, report_output):
        if _path_is_owned(path):
            raise FileExistsError(f"replay evidence output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    trace_temp: Path | None = None
    report_temp: Path | None = None
    linked: list[Path] = []
    try:
        descriptor, raw_name = tempfile.mkstemp(
            prefix=f".{trace_output.name}.tmp-", dir=trace_output.parent
        )
        os.close(descriptor)
        trace_temp = Path(raw_name)
        with trace_temp.open("wb") as stream:
            torch.save(dict(trace), stream)
            stream.flush()
            os.fsync(stream.fileno())
        trace_sha256 = sha256_file(trace_temp)

        final_report = dict(report)
        final_report["trace_sha256"] = trace_sha256
        final_report["trace_output"] = str(trace_output)
        report_bytes = _strict_json_bytes(final_report)
        descriptor, raw_name = tempfile.mkstemp(
            prefix=f".{report_output.name}.tmp-", dir=report_output.parent
        )
        report_temp = Path(raw_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(report_bytes)
            stream.flush()
            os.fsync(stream.fileno())

        os.link(trace_temp, trace_output)
        linked.append(trace_output)
        os.link(report_temp, report_output)
        linked.append(report_output)
        for directory in {trace_output.parent, report_output.parent}:
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return trace_sha256
    except Exception:
        for path in reversed(linked):
            if _path_is_owned(path):
                path.unlink()
        raise
    finally:
        for path in (trace_temp, report_temp):
            if path is not None and _path_is_owned(path):
                path.unlink()


def publish_json_no_clobber(payload: Mapping[str, Any], output: Path) -> None:
    output = Path(os.path.abspath(os.fspath(output)))
    if _path_is_owned(output):
        raise FileExistsError(f"report already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if _path_is_owned(temporary):
        raise FileExistsError(f"temporary report already exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(_strict_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        if _path_is_owned(temporary):
            temporary.unlink()


def update_readiness(
    observation: torch.Tensor,
    ready_count: torch.Tensor,
    fork_used: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Update the public-only readiness state used by the future fork gate."""

    if observation.ndim != 2 or observation.shape[1] != OBSERVATION_DIM:
        raise ValueError("readiness observation must be [N,115]")
    count = observation.shape[0]
    if ready_count.shape != (count,) or ready_count.dtype != torch.long:
        raise ValueError("ready_count must be long[N]")
    if fork_used.shape != (count,) or fork_used.dtype != torch.bool:
        raise ValueError("fork_used must be bool[N]")
    if ready_count.device != observation.device or fork_used.device != observation.device:
        raise ValueError("readiness tensors must share a device")
    if not bool(torch.isfinite(observation).all()):
        raise FloatingPointError("readiness observation contains non-finite values")
    second_other = torch.topk(observation[:, 92:96], k=2, dim=-1).values[:, 1]
    score = torch.minimum(observation[:, 96], second_other)
    latch = observation[:, 106]
    if not bool(((latch == 0.0) | (latch == 1.0)).all()):
        raise RuntimeError("public grasp latch must be binary")
    eligible = (latch == 0.0) & (score >= 0.10)
    next_count = torch.where(
        eligible,
        torch.clamp(ready_count + 1, max=4),
        torch.zeros_like(ready_count),
    )
    trigger = (next_count == 4) & (~fork_used)
    next_used = fork_used | trigger
    high_g = score >= 0.35
    high_close = observation[:, 102] >= 0.20
    stratum = high_g.long() * 2 + high_close.long()
    return {
        "score": score,
        "eligible": eligible,
        "ready_count_before": ready_count,
        "ready_count_after": next_count,
        "trigger": trigger,
        "fork_used_before": fork_used,
        "fork_used_after": next_used,
        "stratum": stratum,
    }


def _masked_cpu(value: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    detached = value.detach()
    mask = active
    while mask.ndim < detached.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, detached, torch.zeros_like(detached)).cpu()


def installed_package_versions(names: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def nvidia_runtime_inventory() -> str:
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,driver_version",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return f"unavailable:{type(error).__name__}"
    return completed.stdout.strip()


def build_trace_payload(
    *,
    metadata: Mapping[str, Any],
    rows: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    """Build and validate the CPU-only dense first-episode trace artifact."""

    metadata_copy = dict(metadata)
    json.dumps(metadata_copy, sort_keys=True, allow_nan=False)
    if metadata_copy.get("kind") != TRACE_KIND:
        raise ValueError("trace metadata kind is invalid")
    num_envs = int(metadata_copy.get("num_envs", -1))
    if num_envs < 1 or not rows:
        raise ValueError("trace requires positive num_envs and at least one row")
    fields = {
        "active": (torch.bool, (num_envs,)),
        "observation": (torch.float32, (num_envs, OBSERVATION_DIM)),
        "executed_action": (torch.float32, (num_envs, ACTION_DIM)),
        "reward": (torch.float32, (num_envs,)),
        "terminated": (torch.bool, (num_envs,)),
        "truncated": (torch.bool, (num_envs,)),
        "terminal_events": (torch.bool, (num_envs, len(TERMINAL_NAMES))),
        "dof_target": (torch.float32, None),
        "joint_pos_target": (torch.float32, None),
        "readiness_score": (torch.float32, (num_envs,)),
        "readiness_eligible": (torch.bool, (num_envs,)),
        "readiness_count_before": (torch.long, (num_envs,)),
        "readiness_count_after": (torch.long, (num_envs,)),
        "readiness_trigger": (torch.bool, (num_envs,)),
        "readiness_fork_used_before": (torch.bool, (num_envs,)),
        "readiness_fork_used_after": (torch.bool, (num_envs,)),
        "readiness_stratum": (torch.long, (num_envs,)),
    }
    first_dof_shape: tuple[int, ...] | None = None
    for row_index, row in enumerate(rows):
        if set(row) != set(fields):
            raise ValueError(f"row {row_index} has an invalid field set")
        for name, (dtype, shape) in fields.items():
            value = row[name]
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                raise TypeError(f"row {row_index} field {name} must be a CPU tensor")
            if value.dtype != dtype:
                raise TypeError(f"row {row_index} field {name} must be {dtype}")
            if shape is not None and tuple(value.shape) != shape:
                raise ValueError(f"row {row_index} field {name} has shape {tuple(value.shape)}")
            if name in {"dof_target", "joint_pos_target"}:
                if value.ndim != 2 or value.shape[0] != num_envs:
                    raise ValueError(f"row {row_index} field {name} must be [N,D]")
                if first_dof_shape is None:
                    first_dof_shape = tuple(value.shape)
                elif tuple(value.shape) != first_dof_shape:
                    raise ValueError("controller target width changed within a trace")
            if dtype.is_floating_point and not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"row {row_index} field {name} is non-finite")
    tensors = {name: torch.stack([row[name] for row in rows]) for name in fields}
    active = tensors["active"]
    done = tensors["terminated"] | tensors["truncated"]
    if active.shape[0] > 1 and not torch.equal(
        active[1:], active[:-1] & (~done[:-1])
    ):
        raise ValueError("first-episode active mask does not follow prior done masks")
    if bool((done & (~active)).any()):
        raise ValueError("inactive trace rows cannot terminate")
    terminal = tensors["terminal_events"]
    if not torch.equal(terminal[..., 0] | terminal[..., 1], tensors["terminated"]):
        raise ValueError("terminal success/failure disagrees with terminated")
    if not torch.equal(terminal[..., 2], tensors["truncated"]):
        raise ValueError("terminal time_out disagrees with truncated")
    if bool((tensors["terminated"] & tensors["truncated"]).any()):
        raise ValueError("terminated and truncated overlap")
    if bool((active[-1] & (~done[-1])).any()):
        raise ValueError("trace ended before every slot completed its first episode")
    return {"metadata": metadata_copy, "tensors": tensors}


@torch.inference_mode()
def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    from adapter import make_pick_tool_env
    from evaluate import FULL_TASK_MODE, _load_diagnostic_approach_actor, validate_terminal_events
    from isaaclab.utils.version import get_isaac_sim_version

    device_string = str(args.device or "cuda:0")
    device = torch.device(device_string)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"SEARCH replay requires CUDA Isaac physics, got {device_string}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    root = repository_root()
    checkpoint = args.checkpoint.resolve()
    actor = _load_diagnostic_approach_actor(checkpoint, device=device).eval()
    env = make_pick_tool_env(
        num_envs=args.num_envs,
        device=device_string,
        seed=args.seed,
        cfg_overrides={
            "sim.physx.enable_enhanced_determinism": bool(args.enhanced_determinism),
        },
        strict=True,
        validate_finite=True,
    )
    rows: list[dict[str, torch.Tensor]] = []
    try:
        observation, _ = env.reset(seed=args.seed, randomize_episode_lengths=False)
        if observation.shape != (args.num_envs, OBSERVATION_DIM):
            raise RuntimeError("adapter reset violated the 115-D observation contract")
        if int(env.max_episode_steps) != 1000:
            raise RuntimeError("replay contract requires the native 1000-step horizon")
        task = env.unwrapped
        cfg = task.cfg
        active = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
        ready_count = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
        fork_used = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
        completed_step = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)

        for step_index in range(1, int(env.max_episode_steps) + 1):
            if not bool(active.any()):
                break
            active_before = active.clone()
            readiness = update_readiness(observation, ready_count, fork_used)
            action = actor(observation)
            if action.shape != (args.num_envs, ACTION_DIM):
                raise RuntimeError("SEARCH actor violated the 115/21 contract")
            if not bool(torch.isfinite(action).all()):
                raise FloatingPointError("SEARCH actor produced NaN or infinity")
            action = action.clamp(-1.0, 1.0)
            next_observation, reward, terminated, truncated, info = env.step(action)
            executed = env.last_executed_policy_action
            if executed is None or not torch.equal(executed, action):
                raise RuntimeError("adapter executed action differs from frozen SEARCH action")
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
            terminal_matrix = torch.stack([events[name] for name in TERMINAL_NAMES], dim=-1)
            row = {
                "active": active_before.cpu(),
                "observation": _masked_cpu(observation, active_before),
                "executed_action": _masked_cpu(executed, active_before),
                "reward": _masked_cpu(reward, active_before),
                "terminated": _masked_cpu(terminated, active_before),
                "truncated": _masked_cpu(truncated, active_before),
                "terminal_events": _masked_cpu(terminal_matrix, active_before),
                "dof_target": _masked_cpu(task.dof_targets, active_before),
                "joint_pos_target": _masked_cpu(task.robot.data.joint_pos_target, active_before),
                "readiness_score": _masked_cpu(readiness["score"], active_before),
                "readiness_eligible": _masked_cpu(readiness["eligible"], active_before),
                "readiness_count_before": _masked_cpu(readiness["ready_count_before"], active_before),
                "readiness_count_after": _masked_cpu(readiness["ready_count_after"], active_before),
                "readiness_trigger": _masked_cpu(readiness["trigger"], active_before),
                "readiness_fork_used_before": _masked_cpu(readiness["fork_used_before"], active_before),
                "readiness_fork_used_after": _masked_cpu(readiness["fork_used_after"], active_before),
                "readiness_stratum": _masked_cpu(readiness["stratum"], active_before),
            }
            rows.append(row)
            done = terminated | truncated
            completed_now = active_before & done
            completed_step = torch.where(
                completed_now,
                torch.full_like(completed_step, step_index),
                completed_step,
            )
            active = active_before & (~done)
            ready_count = torch.where(active, readiness["ready_count_after"], ready_count)
            fork_used = torch.where(active, readiness["fork_used_after"], fork_used)
            observation = next_observation

        if bool(active.any()):
            raise RuntimeError("native horizon ended with incomplete first episodes")
        fingerprints = source_fingerprints(root)
        metadata = {
            "kind": TRACE_KIND,
            "task_mode": FULL_TASK_MODE,
            "observation_contract": "pick_tool_markov115_v1",
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "terminal_names": list(TERMINAL_NAMES),
            "seed": int(args.seed),
            "num_envs": int(args.num_envs),
            "env_slots": list(range(args.num_envs)),
            "randomize_episode_lengths": False,
            "native_max_episode_steps": int(env.max_episode_steps),
            "episode_length_s": float(cfg.episode_length_s),
            "enhanced_determinism": bool(args.enhanced_determinism),
            "device": device_string,
            "policy_contract": POLICY_CONTRACT,
            "readiness_contract": READINESS_CONTRACT,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "source_sha256": fingerprints,
            "git": git_provenance(root, tuple(fingerprints)),
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_device_capability": list(torch.cuda.get_device_capability(device)),
                "isaac_sim": str(get_isaac_sim_version()),
                "packages": installed_package_versions(
                    ("isaaclab", "isaaclab_tasks", "isaaclab_assets", "numpy", "gymnasium")
                ),
                "torch_default_dtype": str(torch.get_default_dtype()),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "nvidia_smi_inventory": nvidia_runtime_inventory(),
                "platform": platform.platform(),
                "environment": {
                    name: os.environ.get(name)
                    for name in (
                        "CUDA_VISIBLE_DEVICES",
                        "CUBLAS_WORKSPACE_CONFIG",
                        "PYTHONHASHSEED",
                    )
                },
            },
            "collection_layout": "same_seed_same_slot_independent_process_replay_v1",
            "dependency_boundary": {
                "repository_inputs": "every path named by source_sha256, including composed robot USD/meshes and retarget weights",
                "runtime_generated_tool_usd": "derived from hashed textured_mesh.obj, material.mtl, and tool_asset.py",
                "external_table_asset": str(cfg.table_usd),
                "external_simulator": "Isaac Sim/Isaac Lab and PhysX binaries are runtime-version provenance, not byte-hashed repository inputs",
            },
            "controller_target_semantics": (
                "post-env.step audit value; on terminal rows Isaac has already auto-reset "
                "the slot, so this is not the pre-reset transition target"
            ),
        }
        trace = build_trace_payload(metadata=metadata, rows=rows)
        tensors = trace["tensors"]
        event_totals = {
            name: int(tensors["terminal_events"][..., index].sum())
            for index, name in enumerate(TERMINAL_NAMES)
        }
        report = {
            "kind": REPORT_KIND,
            "status": "complete",
            "trace_kind": TRACE_KIND,
            "seed": int(args.seed),
            "num_envs": int(args.num_envs),
            "steps_recorded": len(rows),
            "completed_step_min": int(completed_step.min()),
            "completed_step_max": int(completed_step.max()),
            "terminal_event_totals": event_totals,
            "readiness_trigger_total": int(tensors["readiness_trigger"].sum()),
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "source_sha256": metadata["source_sha256"],
            "policy_contract": POLICY_CONTRACT,
            "collection_layout": metadata["collection_layout"],
            "controller_target_semantics": metadata["controller_target_semantics"],
        }
        return trace, report
    finally:
        env.close()


def parse_args() -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--num_envs", "--num-envs", dest="num_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace_output", "--trace-output", dest="trace_output", type=Path, required=True)
    parser.add_argument("--report_output", "--report-output", dest="report_output", type=Path, required=True)
    parser.add_argument("--enhanced_determinism", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num_envs must be positive")
    if not args.checkpoint.is_file() or args.checkpoint.is_symlink():
        parser.error("--checkpoint must be a regular, non-symlink file")
    if _path_is_owned(args.trace_output) or _path_is_owned(args.report_output):
        parser.error("trace/report output already exists; replay evidence is immutable")
    launcher = AppLauncher(args)
    return args, launcher.app


def main() -> None:
    args, simulation_app = parse_args()
    try:
        try:
            trace, report = run(args)
            trace_sha = publish_trace_and_report_no_clobber(
                trace, report, args.trace_output, args.report_output
            )
            print(
                "[search-replay-trace] "
                f"seed={args.seed} slots={args.num_envs} "
                f"steps={report['steps_recorded']} sha256={trace_sha}",
                flush=True,
            )
        except Exception as error:
            if not _path_is_owned(args.report_output):
                publish_json_no_clobber(
                    {
                        "kind": REPORT_KIND,
                        "status": "failed",
                        "seed": int(args.seed),
                        "num_envs": int(args.num_envs),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                    args.report_output,
                )
            raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
