#!/usr/bin/env python3
"""Validate the external-data contract for the functional-pregrasp tool set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "functional_pregrasp" / "tools_v1.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _require_file(dataset_root: Path, relative: str) -> Path:
    path = dataset_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing dataset file: {path}")
    return path


def validate_manifest(manifest_path: Path, dataset_root: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("only functional-pregrasp manifest schema_version=1 is supported")
    objects = manifest.get("objects")
    if not isinstance(objects, list) or len(objects) != 10:
        raise ValueError("the v1 benchmark must contain exactly 10 objects")

    object_ids: set[str] = set()
    names: set[str] = set()
    pose_count = 0
    for expected_order, item in enumerate(objects):
        if not isinstance(item, dict):
            raise TypeError(f"objects[{expected_order}] must be an object")
        object_id = item.get("object_id")
        name = item.get("name")
        if item.get("order") != expected_order:
            raise ValueError(f"{object_id}: order must be contiguous and deterministic")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError(f"objects[{expected_order}]: object_id is required")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{object_id}: name is required")
        if object_id in object_ids or name in names:
            raise ValueError(f"duplicate object id/name: {object_id}/{name}")
        object_ids.add(object_id)
        names.add(name)
        if item.get("intent") != "use":
            raise ValueError(f"{object_id}: Phase 1 requires an explicit use intent")

        max_length = float(item.get("max_length_m", 0.0))
        min_length = float(item.get("min_length_m", 0.0))
        if not (0.01 <= min_length <= max_length <= 0.40):
            raise ValueError(f"{object_id}: implausible metric mesh dimensions")
        _require_file(dataset_root, str(item.get("mesh")))
        _require_file(dataset_root, str(item.get("collision_mesh")))

        symmetry = item.get("symmetry")
        if not isinstance(symmetry, dict) or symmetry.get("kind") is None:
            raise ValueError(f"{object_id}: symmetry contract is required")
        yaw_order = symmetry.get("yaw_order")
        if not isinstance(yaw_order, int) or yaw_order < 1:
            raise ValueError(f"{object_id}: symmetry.yaw_order must be a positive integer")
        if symmetry.get("kind") == "continuous_axial_roll":
            axis = symmetry.get("axis_local") or symmetry.get("axis")
            if axis is None:
                raise ValueError(f"{object_id}: continuous symmetry requires an axis")

        rigid_mode = item.get("rigid_mode")
        if rigid_mode not in {"already_rigid", "lock_and_merge_all_parts"}:
            raise ValueError(f"{object_id}: Phase 1 must state how articulation is frozen")

        pose_files = item.get("pose_files")
        if not isinstance(pose_files, list) or not pose_files:
            raise ValueError(f"{object_id}: at least one use target is required")
        for relative_pose in pose_files:
            pose_path = _require_file(dataset_root, str(relative_pose))
            pose = _load_json(pose_path)
            if pose.get("object_code") != object_id or pose.get("intent") != "use":
                raise ValueError(f"{pose_path}: target does not match {object_id}/use")
            if pose.get("object_pose_wrt_palm") is None:
                raise ValueError(f"{pose_path}: missing object_pose_wrt_palm")
            if float(pose.get("scale", 0.0)) != 1.0:
                raise ValueError(f"{pose_path}: expected metre-scale target with scale=1")
            pose_count += 1

    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": digest,
        "dataset_root": str(dataset_root.resolve()),
        "objects": len(objects),
        "use_targets": pose_count,
        "joint_training": bool(manifest["phase_1"]["joint_training"]),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("XHAND_UNIDEXFPM_ROOT", "/disk2/xhand_datasets/unidexfpm")),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = validate_manifest(args.manifest, args.dataset_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
