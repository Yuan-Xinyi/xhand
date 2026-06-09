# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared scene-lighting helpers used across the direct tasks.

Keeping the lighting in one place means every training/eval environment renders
with the same "lab fluorescent" look instead of each task re-implementing its own
DomeLight call inside ``_setup_scene``.
"""

from __future__ import annotations

from isaaclab.sim.utils import get_current_stage
from pxr import Gf, Sdf, UsdGeom, UsdLux


def add_ceiling_fluorescent_lights(
    rows=4,
    cols=4,
    spacing_x=1.5,
    spacing_y=1.5,
    height=3.0,
    tube_length=1.2,
    tube_width=0.15,
    intensity=10000.0,
    color_temperature=4500.0,
):
    """Add a grid of long rectangular lights at ceiling height to mimic lab fluorescent tubes.

    RectLight emits along its local -Z by default, so placing the tubes flat at the
    ceiling makes them shine straight down without any rotation.
    """
    stage = get_current_stage()
    parent_path = "/World/CeilingLights"
    UsdGeom.Xform.Define(stage, parent_path)

    x0 = -(cols - 1) * spacing_x / 2.0
    y0 = -(rows - 1) * spacing_y / 2.0
    for r in range(rows):
        for c in range(cols):
            path = f"{parent_path}/Tube_{r}_{c}"
            light = UsdLux.RectLight.Define(stage, Sdf.Path(path))
            light.CreateWidthAttr(tube_length)
            light.CreateHeightAttr(tube_width)
            light.CreateIntensityAttr(intensity)
            light.CreateEnableColorTemperatureAttr(True)
            light.CreateColorTemperatureAttr(color_temperature)
            UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3d(x0 + c * spacing_x, y0 + r * spacing_y, height)
            )

    # faint ambient fill so shadows match a real room's bounced light instead of going pitch black
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path(f"{parent_path}/AmbientFill"))
    dome.CreateIntensityAttr(150.0)
    dome.CreateEnableColorTemperatureAttr(True)
    dome.CreateColorTemperatureAttr(color_temperature)
    return parent_path
