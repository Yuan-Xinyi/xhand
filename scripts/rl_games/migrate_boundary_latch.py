# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Migrate an old 7-field curriculum boundary dataset to the 10-field latch schema.

The env's curriculum loader now requires ``contact_steps`` / ``lost_contact_steps`` /
``is_grasped`` per boundary state (the grasp latch is a hysteresis counter -- genuine Markov
state that cannot be derived from a single frame).  Datasets captured before that change carry
only the 7 pose/controller fields and are rejected at ``gym.make``.

The latch values can be reconstructed faithfully per boundary *class*:

* **Unlatched boundaries** (pregrasp -- ``hover_start``/``descend_start``/``close_start``):
  the hand has not closed, so ``(contact_steps=0, lost_contact_steps=0, is_grasped=False)``
  is exactly correct.
* **Latched boundaries** (``micro_start``/``lift_start``/``micro_end``/``mid_lift``/
  ``settle_start``/``success``): the oracle only reached these states with the latch formed.
  ``(contact_steps=confirm_steps, lost_contact_steps=0, is_grasped=True)`` is behaviorally
  exact: the latch stays latched, and the normalized observation component
  ``clamp(contact_steps/confirm_steps, 0, 1)`` saturates at 1 either way.

Boundaries not listed in either class are left untouched (and reported), so an unknown future
boundary cannot be silently mislabeled.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch


UNLATCHED_BOUNDARIES = ("hover_start", "descend_start", "close_start")
LATCHED_BOUNDARIES = (
    "micro_start",
    "lift_start",
    "micro_end",
    "mid_lift",
    "settle_start",
    "success",
)
LATCH_KEYS = ("contact_steps", "lost_contact_steps", "is_grasped")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--confirm-steps",
        type=int,
        default=4,
        help="cfg.grasp_confirm_steps of the target env; latched boundaries are saturated here.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite latch fields that already exist (default: refuse).",
    )
    args = parser.parse_args()
    if args.confirm_steps < 1:
        parser.error("--confirm-steps must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    dataset = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(dataset, dict) or not isinstance(dataset.get("boundaries"), dict):
        raise TypeError("dataset must be a dict containing a 'boundaries' mapping")

    migrated: list[dict[str, object]] = []
    skipped: list[str] = []
    for name, boundary in dataset["boundaries"].items():
        if not isinstance(boundary, dict):
            raise TypeError(f"boundary {name!r} is not a mapping")
        existing = [key for key in LATCH_KEYS if key in boundary]
        if existing and not args.force:
            skipped.append(f"{name} (already has {existing})")
            continue
        joint_pos = boundary.get("joint_pos")
        if not torch.is_tensor(joint_pos) or joint_pos.ndim != 2:
            raise TypeError(f"boundary {name!r} lacks a [K, J] joint_pos tensor")
        count = joint_pos.shape[0]
        if name in UNLATCHED_BOUNDARIES:
            contact = torch.zeros(count, dtype=torch.long)
            grasped = torch.zeros(count, dtype=torch.bool)
            kind = "unlatched"
        elif name in LATCHED_BOUNDARIES:
            contact = torch.full((count,), args.confirm_steps, dtype=torch.long)
            grasped = torch.ones(count, dtype=torch.bool)
            kind = "latched"
        else:
            skipped.append(f"{name} (unknown boundary class, left untouched)")
            continue
        boundary["contact_steps"] = contact
        boundary["lost_contact_steps"] = torch.zeros(count, dtype=torch.long)
        boundary["is_grasped"] = grasped
        migrated.append({"boundary": name, "class": kind, "states": int(count)})

    if not migrated:
        raise RuntimeError(f"nothing migrated; skipped={skipped}")

    meta = dataset.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        dataset["meta"] = meta
    meta["boundary_latch_migration"] = {
        "format_version": 1,
        "source_path": str(args.input.resolve()),
        "source_sha256": _sha256(args.input),
        "confirm_steps": args.confirm_steps,
        "migrated": copy.deepcopy(migrated),
        "skipped": list(skipped),
        "unlatched_boundaries": list(UNLATCHED_BOUNDARIES),
        "latched_boundaries": list(LATCHED_BOUNDARIES),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, args.output)
    print(json.dumps(meta["boundary_latch_migration"], indent=2))
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
