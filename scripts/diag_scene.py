"""Diagnostic: list every geometry prim in the SimToolReal scene, grouped by env.

Reveals any unexpected/extra geometry (e.g. a stray box). Run:
  conda activate env_isaaclab; python scripts/diag_scene.py --headless --num_envs 4
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="SimToolReal-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import xhand_inhand.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaacsim.core.utils.prims as prim_utils  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402


def main():
    import omni.usd
    from pxr import UsdPhysics
    cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=cfg).unwrapped
    stage = omni.usd.get_context().get_stage()

    # ---- count Object prims + which have a rigid-body API (BEFORE reset, which crashes) ----
    print("\n>>> OBJECT PRIM AUDIT (before reset) <<<", flush=True)
    obj_prims, obj_with_rb = [], []
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if p.startswith("/World/envs/env_") and p.endswith("/Object"):
            obj_prims.append(p)
            has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
            # check children too (the body may be a child link)
            child_rb = any(c.HasAPI(UsdPhysics.RigidBodyAPI) for c in prim.GetChildren())
            if has_rb or child_rb:
                obj_with_rb.append(p)
    print(f"  total /Object prims: {len(obj_prims)}", flush=True)
    print(f"  with RigidBodyAPI:   {len(obj_with_rb)}", flush=True)
    missing = sorted(set(obj_prims) - set(obj_with_rb))
    print(f"  MISSING rigid body ({len(missing)}): {missing}", flush=True)
    # which envs have NO /Object prim at all
    have = {p.split('/')[3] for p in obj_prims}
    allenv = {f"env_{i}" for i in range(args.num_envs)}
    print(f"  envs with NO /Object prim: {sorted(allenv - have)}", flush=True)

    try:
        env.reset()
        print(">>> reset OK <<<", flush=True)
    except Exception as e:
        print(f">>> reset FAILED (expected if instance mismatch): {e} <<<", flush=True)
    print("\n>>> STAGE ACQUIRED, traversing...", flush=True)

    GEOM = {"Mesh", "Cube", "Sphere", "Cylinder", "Cone", "Capsule"}
    print("\n================= GEOMETRY PRIMS BY SUBTREE =================")
    # group by top-level container
    rows = []
    for prim in stage.Traverse():
        t = prim.GetTypeName()
        if t in GEOM:
            path = str(prim.GetPath())
            # world position
            try:
                xf = UsdGeom.Xformable(prim)
                m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                tr = m.ExtractTranslation()
                pos = f"({tr[0]:.2f},{tr[1]:.2f},{tr[2]:.2f})"
            except Exception:
                pos = "?"
            rows.append((path, t, pos))

    # count geometry "top-level groups" (Object/Robot/Table/...) per env
    from collections import defaultdict
    per_env_tops = defaultdict(set)
    for path, t, pos in rows:
        if path.startswith("/World/envs/env_"):
            rest = path[len("/World/envs/"):]
            env_name = rest.split("/")[0]
            top = rest.split("/", 1)[1].split("/")[0] if "/" in rest else "?"
            per_env_tops[env_name].add(top)
    print("\n--- top-level geometry groups present in each env ---")
    for env_name in sorted(per_env_tops, key=lambda s: int(s.split("_")[1])):
        print(f"  {env_name}: {sorted(per_env_tops[env_name])}")

    # detailed dump for a normal env (0) and an eraser env (12 if present)
    for env_id in [0, 1, 12, 13]:
        if env_id >= args.num_envs:
            continue
        prefix = f"/World/envs/env_{env_id}/"
        print(f"\n--- env_{env_id} detail ---")
        for path, t, pos in rows:
            if path.startswith(prefix):
                rest = path[len(prefix):]
                top = rest.split("/")[0]
                print(f"  [{t:7}] {top:20} @ {pos}   ({rest})")

    # anything NOT under /World/envs (stray global geometry)
    print("\n--- geometry NOT under /World/envs (global/stray) ---")
    for path, t, pos in rows:
        if not path.startswith("/World/envs/"):
            print(f"  [{t:7}] {path} @ {pos}")

    # also report which object each env got (manifest order)
    print("\n--- per-env object assignment (env i -> object i % pool) ---")
    import json
    pool = json.load(open(env.cfg.object_manifest_path))["objects"]
    for env_id in range(min(args.num_envs, 16)):
        o = pool[env_id % len(pool)]
        print(f"  env_{env_id}: {o['name']:14} type={o['type']:11} shape={o['shape']}")

    print("============================================================\n")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
