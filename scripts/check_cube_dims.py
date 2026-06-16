"""Report the in-hand cube's actual physical size (AABB) and mass.

Spawns the DexCube exactly as the xhand_repose env configures it, then reads the
physics-engine values (computed mass + world AABB) so the numbers are ground truth,
not a guess from the USD/scale/density.

Run: conda activate env_isaaclab; python scripts/check_cube_dims.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject  # noqa: E402

from xhand_inhand.tasks.direct.xhand_repose.xhand_repose_env_cfg import XHandReposeEnvCfg  # noqa: E402


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 100, device=args.device))

    cfg = XHandReposeEnvCfg().object_cfg
    cfg = cfg.replace(prim_path="/World/cube")
    cfg.init_state.pos = (0.0, 0.0, 0.0)
    obj = RigidObject(cfg)

    sim.reset()

    # physics-computed mass (kg) from the rigid-body view
    masses = obj.root_physx_view.get_masses()  # (1, num_bodies)
    mass = float(masses.sum())

    # world AABB extents -> edge lengths (m), from the USD geometry bbox cache
    import omni.usd  # noqa: E402
    from pxr import Usd, UsdGeom  # noqa: E402

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/cube")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    size = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetSize()
    ex, ey, ez = float(size[0]), float(size[1]), float(size[2])

    spawn = XHandReposeEnvCfg().object_cfg.spawn
    density = spawn.mass_props.density
    scale = spawn.scale

    vol = ex * ey * ez
    print("\n================ DexCube (xhand_repose object_cfg) ================", flush=True)
    print(f"  usd scale            : {scale}", flush=True)
    print(f"  density (cfg)        : {density} kg/m^3", flush=True)
    print(f"  AABB edge lengths    : x={ex*100:.3f} cm  y={ey*100:.3f} cm  z={ez*100:.3f} cm", flush=True)
    print(f"  volume               : {vol*1e6:.2f} cm^3", flush=True)
    print(f"  MASS (physx)         : {mass*1000:.2f} g  ({mass:.4f} kg)", flush=True)
    print(f"  check density*vol    : {density*vol*1000:.2f} g", flush=True)
    print("===================================================================\n", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
