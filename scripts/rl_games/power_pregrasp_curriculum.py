#!/usr/bin/env python3
"""Build a static coupled-power curriculum from ranked physical close-start states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


RANK_CONTRACT = "power_pregrasp_third_finger_bottleneck_v1"
CURRICULUM_CONTRACT = "coupled_power_static_close_start_v1"
BOUNDARY_FIELDS = (
    "joint_pos",
    "joint_vel",
    "dof_targets",
    "object_local_pos",
    "object_quat",
    "object_velocity",
    "last_action",
    "contact_steps",
    "lost_contact_steps",
    "is_grasped",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rank_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if not isinstance(report, dict) or report.get("contract") != RANK_CONTRACT:
        raise ValueError(f"rank report must use contract {RANK_CONTRACT!r}")
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("rank report contains no records")
    return report


def build_static_curriculum(rank_report: Path, top_k: int) -> dict[str, Any]:
    """Select the first ``top_k`` records and remove all captured dynamics/preload."""

    if isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    report = _load_rank_report(rank_report)
    records = report["records"]
    if top_k > len(records):
        raise ValueError(f"top_k={top_k} exceeds the {len(records)} ranked records")

    dataset_cache: dict[Path, dict[str, Any]] = {}
    selected_rows: dict[str, list[torch.Tensor]] = {field: [] for field in BOUNDARY_FIELDS}
    selected_meta: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for expected_rank, record in enumerate(records[:top_k], start=1):
        if not isinstance(record, dict) or int(record.get("rank", -1)) != expected_rank:
            raise ValueError("rank records must be contiguous and ordered from one")
        source_path = Path(str(record.get("dataset", ""))).expanduser().resolve()
        state_index = int(record.get("dataset_state_index", -1))
        if not source_path.is_file():
            raise FileNotFoundError(f"ranked source dataset does not exist: {source_path}")
        if source_path not in dataset_cache:
            dataset = torch.load(source_path, map_location="cpu", weights_only=False)
            if not isinstance(dataset, dict):
                raise TypeError(f"source dataset is not a dictionary: {source_path}")
            dataset_cache[source_path] = dataset
            source_hashes[str(source_path)] = sha256_file(source_path)
        dataset = dataset_cache[source_path]
        try:
            boundary = dataset["boundaries"]["close_start"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"source dataset lacks boundaries.close_start: {source_path}") from exc
        if not isinstance(boundary, dict):
            raise TypeError(f"close_start boundary is not a dictionary: {source_path}")
        for field in BOUNDARY_FIELDS:
            tensor = boundary.get(field)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim not in (1, 2):
                raise ValueError(f"invalid close_start.{field} in {source_path}")
            if state_index < 0 or state_index >= tensor.shape[0]:
                raise IndexError(
                    f"state index {state_index} is outside close_start.{field} with "
                    f"{tensor.shape[0]} rows"
                )
            selected_rows[field].append(tensor[state_index].detach().cpu().clone())
        selected_meta.append(
            {
                "rank": expected_rank,
                "dataset": str(source_path),
                "dataset_state_index": state_index,
                "dataset_seed": record.get("dataset_seed"),
                "score": float(record.get("score", 0.0)),
                "thumb_near": float(record.get("thumb_near", 0.0)),
                "third_other_near": float(record.get("third_other_near", 0.0)),
                "power_palm_score": float(record.get("power_palm_score", 0.0)),
            }
        )

    output_boundary = {
        field: torch.stack(rows, dim=0) for field, rows in selected_rows.items()
    }
    joint_pos = output_boundary["joint_pos"]
    output_boundary["joint_vel"] = torch.zeros_like(joint_pos)
    output_boundary["dof_targets"] = joint_pos.clone()
    output_boundary["object_velocity"] = torch.zeros_like(
        output_boundary["object_velocity"]
    )
    output_boundary["last_action"] = torch.zeros_like(output_boundary["last_action"])
    output_boundary["contact_steps"] = torch.zeros_like(
        output_boundary["contact_steps"], dtype=torch.long
    )
    output_boundary["lost_contact_steps"] = torch.zeros_like(
        output_boundary["lost_contact_steps"], dtype=torch.long
    )
    output_boundary["is_grasped"] = torch.zeros_like(
        output_boundary["is_grasped"], dtype=torch.bool
    )

    return {
        "boundaries": {"close_start": output_boundary},
        "meta": {
            "format_version": 1,
            "contract": CURRICULUM_CONTRACT,
            "task_mode": "coupled_power_align_close_option_v1",
            "observation_contract": "pick_tool_coupled_power_align_close_state131_v1",
            "action_layout": "arm_delta7|crossdex_token9|distal_residual5",
            "rank_report": str(rank_report.resolve()),
            "rank_report_sha256": sha256_file(rank_report),
            "top_k": top_k,
            "source_dataset_sha256": source_hashes,
            "selected": selected_meta,
            "staticization": {
                "joint_velocity": "zero",
                "dof_targets": "joint_pos",
                "object_velocity": "zero",
                "last_action": "zero",
                "legacy_grasp_latch": "clear",
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank_report", required=True, type=Path)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curriculum = build_static_curriculum(args.rank_report, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(curriculum, args.output)
    print(
        f"wrote {args.output} contract={CURRICULUM_CONTRACT} "
        f"states={args.top_k} sha256={sha256_file(args.output)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
