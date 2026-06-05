# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn the task's in-hand cube and print its physical bounding-box size."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import omni.usd
from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

from xhand_inhand.tasks.direct.xhand_repose.xhand_repose_env_cfg import XHandReposeEnvCfg


def main():
    cfg = XHandReposeEnvCfg()
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 120))
    # spawn the cube exactly as the task does (scale included)
    cfg.object_cfg.spawn.func("/World/object", cfg.object_cfg.spawn)
    sim.reset()

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/object")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    size = rng.GetSize()
    print(f"[CUBE] scale in cfg = {cfg.object_cfg.spawn.scale}")
    print(f"[CUBE] bbox size (m) = x:{size[0]:.4f}  y:{size[1]:.4f}  z:{size[2]:.4f}")
    print(f"[CUBE] edge length  = {size[0]*100:.2f} cm")

    simulation_app.close()


if __name__ == "__main__":
    main()
