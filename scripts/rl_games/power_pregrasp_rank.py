#!/usr/bin/env python3
"""Rank captured close-start states by thumb-plus-three pregrasp geometry.

Historical handoff datasets were selected with a thumb-plus-two objective.  This diagnostic loads
every unique ``close_start`` state, restores all of them in one Isaac batch, and ranks the third
legal non-thumb proximity bottleneck without using contact or lift proxies.  Top states are emitted
as recoverable JSON artifacts for strict power-close CEM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--inputs", nargs="+", required=True, help="curriculum .pt datasets")
parser.add_argument("--top_k", type=int, default=12)
parser.add_argument("--output_dir", default="/tmp/pick_tool_power_pregrasps")
parser.add_argument("--report", default="/tmp/pick_tool_power_pregrasp_rank.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.top_k < 1:
    parser.error("--top_k must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401


REQUIRED_WIDTHS = {
    "joint_pos": 19,
    "joint_vel": 19,
    "dof_targets": 19,
    "object_local_pos": 3,
    "object_quat": 4,
    "object_velocity": 6,
    "last_action": 21,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_key(boundary: dict[str, torch.Tensor], index: int) -> bytes:
    # Quantization removes serialization noise while keeping sub-millimetre/radian distinctions.
    values = torch.cat(
        (
            boundary["joint_pos"][index],
            boundary["object_local_pos"][index],
            boundary["object_quat"][index],
        )
    )
    return torch.round(values * 1.0e5).to(torch.int64).numpy().tobytes()


@torch.inference_mode()
def main() -> None:
    states: dict[str, list[torch.Tensor]] = {key: [] for key in REQUIRED_WIDTHS}
    provenance: list[dict] = []
    seen: set[bytes] = set()
    input_records: list[dict] = []

    for raw_path in args_cli.inputs:
        path = Path(raw_path).resolve()
        dataset = torch.load(path, map_location="cpu", weights_only=False)
        try:
            boundary = dataset["boundaries"]["close_start"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path} has no boundaries.close_start") from exc
        count = None
        for key, width in REQUIRED_WIDTHS.items():
            value = boundary.get(key)
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != width:
                shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
                raise ValueError(f"{path}: close_start.{key} expected [K,{width}], got {shape}")
            count = value.shape[0] if count is None else count
            if value.shape[0] != count:
                raise ValueError(f"{path}: inconsistent close_start row counts")
        assert count is not None
        kept = 0
        for index in range(count):
            key = _state_key(boundary, index)
            if key in seen:
                continue
            seen.add(key)
            for name in REQUIRED_WIDTHS:
                states[name].append(boundary[name][index].to(torch.float32))
            provenance.append(
                {
                    "dataset": str(path),
                    "dataset_state_index": index,
                    "dataset_seed": dataset.get("meta", {}).get("seed"),
                }
            )
            kept += 1
        input_records.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "states": count,
                "unique_states_added": kept,
            }
        )

    if not provenance:
        raise ValueError("no close_start states were loaded")
    packed = {key: torch.stack(value, dim=0) for key, value in states.items()}
    num_envs = len(provenance)
    cfg = parse_env_cfg(
        "Pick-Tool-Token-Direct-v0", device=args_cli.device, num_envs=num_envs
    )
    cfg.episode_length_s = 120.0
    cfg.terminate_on_drop = False
    cfg.success_hold_steps = 100000
    env = gym.make("Pick-Tool-Token-Direct-v0", cfg=cfg)
    u = env.unwrapped
    env.reset()
    dev = u.device
    env_ids = u.robot._ALL_INDICES
    packed = {key: value.to(dev) for key, value in packed.items()}

    joint_pos = packed["joint_pos"]
    # Match close-option reset semantics: stop the arm and object while retaining captured finger
    # velocity.  Geometry ranking itself is read before any physics step.
    joint_vel = packed["joint_vel"].clone()
    joint_vel[:, u._arm_ids_t] = 0.0
    dof_targets = packed["dof_targets"].clone()
    dof_targets[:, u._arm_ids_t] = joint_pos[:, u._arm_ids_t]
    u.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    u.robot.set_joint_position_target(dof_targets, env_ids=env_ids)
    u.dof_targets.copy_(dof_targets)
    pose = torch.empty((num_envs, 7), dtype=torch.float32, device=dev)
    pose[:, :3] = packed["object_local_pos"] + u.scene.env_origins
    pose[:, 3:] = packed["object_quat"]
    u.object.write_root_pose_to_sim(pose, env_ids=env_ids)
    u.object.write_root_velocity_to_sim(
        torch.zeros((num_envs, 6), device=dev), env_ids=env_ids
    )
    u.actions.copy_(packed["last_action"])
    u.prev_actions.copy_(packed["last_action"])
    u._compute_intermediate_values()

    signals = u._compute_grasp_signals()
    legal_proximity = signals["power_legal_finger_proximity"]
    thumb_near = legal_proximity[:, u._contact_thumb_idx]
    other_near = legal_proximity[:, u._contact_other_ids]
    ranked_other, ranked_local = torch.topk(other_near, k=4, dim=1)
    third_near = ranked_other[:, 2]
    fourth_near = ranked_other[:, 3]
    bottleneck = torch.minimum(thumb_near, third_near)
    mean_support = (thumb_near + ranked_other[:, :3].sum(dim=-1)) / 4.0
    rank_score = signals["power_palm_score"] * (
        0.75 * bottleneck + 0.25 * mean_support
    )
    clearance = u._object_true_min_z() - u._table_surface_z
    rank_score = torch.where(
        clearance.abs() <= 0.005,
        rank_score,
        torch.full_like(rank_score, -1.0),
    )

    order = torch.argsort(rank_score, descending=True).cpu().tolist()
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    finger_names = list(u.ee_names)
    for rank, index in enumerate(order):
        selected_sensor_indices = u._contact_other_ids[ranked_local[index]].cpu().tolist()
        record = {
            "rank": rank + 1,
            "global_state_index": index,
            **provenance[index],
            "score": float(rank_score[index].item()),
            "thumb_near": float(thumb_near[index].item()),
            "third_other_near": float(third_near[index].item()),
            "fourth_other_near": float(fourth_near[index].item()),
            "mean_support": float(mean_support[index].item()),
            "power_palm_score": float(signals["power_palm_score"][index].item()),
            "true_clearance": float(clearance[index].item()),
            "finger_order": finger_names,
            "legal_finger_proximity": legal_proximity[index].cpu().tolist(),
            "finger_alignment_score": signals["power_finger_alignment_score"][index]
            .cpu()
            .tolist(),
            "other_opposition_score": signals["power_finger_opposition_score"][index]
            .cpu()
            .tolist(),
            "ranked_other_fingers": [finger_names[i] for i in selected_sensor_indices],
        }
        if rank < min(args_cli.top_k, num_envs):
            artifact_path = output_dir / f"rank_{rank + 1:02d}_state_{index:02d}.json"
            artifact = {
                "format_version": 1,
                "contract": "power_pregrasp_third_finger_bottleneck_v1",
                "source": record,
                "pregrasp": {
                    "joint_pos": packed["joint_pos"][index].cpu().tolist(),
                    "joint_vel": packed["joint_vel"][index].cpu().tolist(),
                    "dof_targets": packed["dof_targets"][index].cpu().tolist(),
                    "object_local_pos": packed["object_local_pos"][index]
                    .cpu()
                    .tolist(),
                    "object_quat": packed["object_quat"][index].cpu().tolist(),
                    "object_velocity": packed["object_velocity"][index].cpu().tolist(),
                    "last_action": packed["last_action"][index].cpu().tolist(),
                    "token": packed["last_action"][index, 7:16].cpu().tolist(),
                },
            }
            artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
            record["artifact"] = str(artifact_path.resolve())
        records.append(record)

    report = {
        "format_version": 1,
        "contract": "power_pregrasp_third_finger_bottleneck_v1",
        "inputs": input_records,
        "unique_states": num_envs,
        "top_k": min(args_cli.top_k, num_envs),
        "records": records,
    }
    report_path = Path(args_cli.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    best = records[0]
    print(
        f"ranked {num_envs} unique close_start states; "
        f"best score={best['score']:.4f} thumb={best['thumb_near']:.3f} "
        f"third={best['third_other_near']:.3f} fourth={best['fourth_other_near']:.3f} "
        f"palm={best['power_palm_score']:.3f}",
        flush=True,
    )
    print(f"wrote {report_path} and {min(args_cli.top_k, num_envs)} artifacts", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
