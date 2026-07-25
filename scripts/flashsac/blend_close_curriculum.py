# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Spawn-space reverse curriculum between oracle pregrasp and post-nudge handoff states.

The close policy latches ~68% from oracle pregrasp spawns and 0% from the post-nudge handoff
states, and no reward change breaks the standoff (uniform -100 non-success outcomes still lose
to the discount + occupancy hover income).  The one recipe that has crossed such a gap in this
project is the v6 spawn anneal: start episodes where the skill exists and slide the spawn
distribution toward the target states while success income keeps flowing.

Each post-nudge state is paired with the oracle close_start state whose object pose (env-local
xy + rest-frame yaw) is nearest; every state field is then interpolated at blend b (0 = oracle,
1 = post-nudge): lerp for positions/joints/actions, slerp-by-wrapped-yaw for the object heading.
Latch fields stay exactly zero (both families are unlatched).  Each emitted dataset mixes an
oracle anchor fraction so the latch gradient never disappears mid-anneal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

LATCH_FIELDS = ("contact_steps", "lost_contact_steps", "is_grasped")


def wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def quat_yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_quat_like(q_ref: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rebuild a quaternion with the reference's roll/pitch replaced-by-nothing: both state
    families are flat on the table, so the full orientation is yaw x rest; compose yaw onto the
    de-yawed reference."""
    ref_yaw = quat_yaw(q_ref)
    half = 0.5 * (yaw - ref_yaw)
    dz = torch.stack(
        (torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)),
        dim=-1,
    )
    w1, x1, y1, z1 = dz.unbind(-1)
    w2, x2, y2, z2 = q_ref.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--postnudge", type=Path, required=True)
    parser.add_argument("--blends", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument(
        "--anchor_fraction",
        type=float,
        default=0.3,
        help="fraction of pure-oracle states mixed into every emitted dataset (replicated to "
        "this share) so latch successes keep paying throughout the anneal",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    oracle = torch.load(args.oracle, map_location="cpu", weights_only=False)
    post = torch.load(args.postnudge, map_location="cpu", weights_only=False)
    ob = oracle["boundaries"]["close_start"]
    pb = post["boundaries"]["close_start"]
    assert set(ob) == set(pb), f"schema mismatch: {set(ob) ^ set(pb)}"

    # Pair by object pose: env-local xy + wrapped yaw (both families sit in the canonical
    # pose family, so nearest-object pairing keeps the hand-object relative geometry sane
    # along the whole interpolation path).
    o_xy = ob["object_local_pos"][:, :2]
    p_xy = pb["object_local_pos"][:, :2]
    o_yaw = quat_yaw(ob["object_quat"])
    p_yaw = quat_yaw(pb["object_quat"])
    d_xy = torch.cdist(p_xy, o_xy)
    d_yaw = wrap(p_yaw.unsqueeze(1) - o_yaw.unsqueeze(0)).abs()
    pair = (d_xy + 0.10 * d_yaw).argmin(dim=1)
    paired_xy = d_xy.gather(1, pair.unsqueeze(1)).squeeze(1)
    paired_yaw = d_yaw.gather(1, pair.unsqueeze(1)).squeeze(1)
    print(
        f"pairing: xy err median {paired_xy.median():.3f} max {paired_xy.max():.3f} m; "
        f"yaw err median {paired_yaw.median():.3f} max {paired_yaw.max():.3f} rad"
    )

    n_post = pb["joint_pos"].shape[0]
    n_anchor = int(round(args.anchor_fraction / (1.0 - args.anchor_fraction) * n_post))
    reps = -(-n_anchor // ob["joint_pos"].shape[0])  # ceil
    anchor = {k: v.repeat((reps,) + (1,) * (v.dim() - 1))[:n_anchor] for k, v in ob.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for b in args.blends:
        blended: dict[str, torch.Tensor] = {}
        for key in pb:
            o_val = ob[key][pair]
            p_val = pb[key]
            if key == "is_grasped":
                # Both capture gates exclude latched states; the Schmitt flag stays False.
                assert not p_val.any() and not o_val.any(), key
                blended[key] = p_val.clone()
            elif key in LATCH_FIELDS:
                # Pre-latch Schmitt counters (contact/lost-contact streaks) can be nonzero in
                # either family; round the interpolation to keep them valid step counts.
                blended[key] = ((1.0 - b) * o_val.float() + b * p_val.float()).round().to(p_val.dtype)
            elif key == "object_quat":
                yaw = o_yaw[pair] + b * wrap(p_yaw - o_yaw[pair])
                q = yaw_quat_like(o_val, yaw)
                blended[key] = q / q.norm(dim=-1, keepdim=True)
            else:
                blended[key] = (1.0 - b) * o_val + b * p_val
        merged = {k: torch.cat([blended[k], anchor[k]]) for k in blended}
        for k, v in merged.items():
            if torch.is_floating_point(v):
                assert torch.isfinite(v).all(), k
        out = {
            "boundaries": {"close_start": merged},
            "meta": {
                "format_version": 1,
                "kind": "blend_close_curriculum",
                "blend": b,
                "anchor_fraction": args.anchor_fraction,
                "num_states": int(merged["joint_pos"].shape[0]),
                "sources": {"oracle": str(args.oracle), "postnudge": str(args.postnudge)},
            },
        }
        path = args.output_dir / f"blend_{int(round(b * 100)):03d}.pt"
        torch.save(out, path)
        print(f"b={b:.2f}: {merged['joint_pos'].shape[0]} states -> {path}")


if __name__ == "__main__":
    main()
