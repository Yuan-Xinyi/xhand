# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-shape friction assignment, ported VERBATIM in intent from SimToolReal.

SimToolReal sets every robot collision shape to a low friction (0.5) and only the
five fingertip distal links to a high friction (1.5). That 3x gap is the single
biggest lever that makes non-fingerpad grasps (knuckle / finger-gap / palm-edge)
physically unprofitable, so the policy converges to fingerpad grasping WITHOUT any
explicit contact-reward term. See ``simtoolreal/.../utils/scene_utils.py`` and the
``robot_friction=0.5 / finger_tip_friction=1.5`` defaults in its env cfg.
"""

from __future__ import annotations

import torch


def apply_fingertip_friction(
    robot,
    fingertip_link_names,
    robot_friction: float = 0.5,
    fingertip_friction: float = 1.5,
) -> None:
    """Set all robot collision shapes to ``robot_friction``, then override the shapes
    belonging to ``fingertip_link_names`` to ``fingertip_friction`` (restitution -> 0).

    Static, init-only (no domain randomization), exactly like SimToolReal's default
    (its friction scale ranges are ``(1.0, 1.0)``). Must be called AFTER the simulator
    has started, i.e. once ``robot.root_physx_view`` exists.

    Args:
        robot: the ``Articulation`` for the arm+hand.
        fingertip_link_names: body/link names of the distal fingertip links to lift to
            the high friction (e.g. the env's ``ee_body_names``).
        robot_friction: static == dynamic friction for every non-fingertip shape.
        fingertip_friction: static == dynamic friction for the fingertip-link shapes.
    """
    view = robot.root_physx_view
    # (num_envs, max_shapes, 3): [static_friction, dynamic_friction, restitution], on CPU
    materials = view.get_material_properties()
    materials[..., 0] = robot_friction
    materials[..., 1] = robot_friction
    materials[..., 2] = 0.0

    # Per-body shape counts: the flat shape axis is partitioned by body, in body-index order
    # (same workaround Isaac Lab's `randomize_rigid_body_material` uses, since the Articulation
    # has no direct per-body shape count).
    num_shapes_per_body = [
        robot._physics_sim_view.create_rigid_body_view(link_path).max_shapes
        for link_path in view.link_paths[0]
    ]
    if sum(num_shapes_per_body) != view.max_shapes:
        raise RuntimeError(
            f"Failed to parse per-body shape counts assigning fingertip friction: "
            f"summed {sum(num_shapes_per_body)} vs view.max_shapes {view.max_shapes}."
        )

    # Lift the fingertip distal links to the high friction, resolved by body index so we
    # don't depend on physx link-name ordering.
    fingertip_ids, _ = robot.find_bodies(list(fingertip_link_names))
    for body_id in fingertip_ids:
        start = sum(num_shapes_per_body[:body_id])
        end = start + num_shapes_per_body[body_id]
        materials[:, start:end, 0] = fingertip_friction
        materials[:, start:end, 1] = fingertip_friction

    indices = torch.arange(materials.shape[0], dtype=torch.int64)
    view.set_material_properties(materials, indices)
