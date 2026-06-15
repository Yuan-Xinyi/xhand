"""Generate the procedural handle-head object pool for the SimToolReal task.

This is the Isaac Lab analogue of the SimToolReal Isaac Gym pipeline, which generated
``handle_head_primitives`` URDFs on the fly. Here we instead PRE-generate a fixed pool
(URDF -> USD) once, and record each object's geometry to ``manifest.json`` so the env can:
  * spawn a different object per env (via MultiAssetSpawnerCfg over the USD pool), and
  * feed each env its object's true scale into the observation (``object_scales``).

Each object is a single rigid link = handle (box/cylinder) + head (box/cylinder/none),
exactly as in ``generate_objects.generate_handle_head_urdf`` with combined inertia.

Determinism: object i for tool-type T is sampled with a fixed seed derived from (T, i),
so re-running reproduces the same pool (and the env's per-env scales stay valid).

Run:
  conda activate env_isaaclab
  python gen_pool.py --headless [--per-type 20] [--types hammer screwdriver marker spatula eraser brush]
"""

import argparse
import json
import os

import numpy as np

from isaaclab.app import AppLauncher

# ---- generation params (pure python, no sim needed) -------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_POOL_DIR = os.path.join(_HERE, "pool")

parser = argparse.ArgumentParser(description="Generate the SimToolReal procedural object pool.")
parser.add_argument("--per-type", type=int, default=20, help="objects generated per tool type")
parser.add_argument(
    "--types",
    nargs="+",
    default=["hammer", "screwdriver", "marker", "spatula", "eraser", "brush"],
    help="tool types to include (subset of the 6 handle_head types)",
)
parser.add_argument("--seed", type=int, default=0, help="base RNG seed")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def _sample_pool(types, per_type, base_seed):
    """Pure-python: sample object geometries and write URDFs. Returns the manifest list."""
    from object_size_distributions import OBJECT_SIZE_DISTRIBUTIONS
    from generate_objects import generate_handle_head_urdf

    # group the distributions by tool type (each type may have several shape variants)
    by_type = {}
    for dist in OBJECT_SIZE_DISTRIBUTIONS:
        by_type.setdefault(dist.type, []).append(dist)

    os.makedirs(_POOL_DIR, exist_ok=True)
    manifest = []
    for t_idx, ttype in enumerate(types):
        variants = by_type[ttype]
        for i in range(per_type):
            rng = np.random.RandomState(base_seed + 1009 * t_idx + i)
            # pick a shape variant deterministically, then sample within its ranges
            dist = variants[rng.randint(len(variants))]

            def _u(lo, hi):
                return rng.uniform(np.asarray(lo), np.asarray(hi)).tolist()

            handle_scale = _u(dist.handle_min_lengths, dist.handle_max_lengths)
            handle_density = float(rng.uniform(dist.handle_min_density, dist.handle_max_density))
            if dist.head_min_lengths is None:
                head_scale = None
                head_density = None
            else:
                head_scale = _u(dist.head_min_lengths, dist.head_max_lengths)
                head_density = float(rng.uniform(dist.head_min_density, dist.head_max_density))

            name = f"{ttype}_{i:03d}"
            urdf_path = os.path.join(_POOL_DIR, f"{name}.urdf")
            generate_handle_head_urdf(
                filepath=urdf_path,
                handle_scale=tuple(handle_scale),
                head_scale=tuple(head_scale) if head_scale is not None else None,
                handle_density=handle_density,
                head_density=head_density,
            )

            # AABB extents of the composite, used by the env for keypoint scale / obs.
            ext = _aabb_extents(handle_scale, head_scale)
            manifest.append(
                {
                    "name": name,
                    "type": ttype,
                    "shape": dist.shape,
                    "handle_scale": handle_scale,
                    "head_scale": head_scale,
                    "handle_density": handle_density,
                    "head_density": head_density,
                    "aabb_extents": ext,
                    "urdf": f"{name}.urdf",
                    "usd": f"{name}/{name}.usd",
                }
            )
    return manifest


def _aabb_extents(handle_scale, head_scale):
    """Axis-aligned bounding box (x,y,z) of handle+head, both centered then head at +x.

    Handles the box (3-tuple) vs cylinder (2-tuple: height_x, diameter) cases, matching
    generate_objects' convention (cylinder axis rotated to +x; head offset along +x)."""

    def hx_hy_hz(scale):
        if scale is None:
            return 0.0, 0.0, 0.0
        if len(scale) == 3:
            return scale[0], scale[1], scale[2]
        # cylinder: (height along x, diameter)
        return scale[0], scale[1], scale[1]

    h_x, h_y, h_z = hx_hy_hz(handle_scale)
    if head_scale is None:
        return [h_x, h_y, h_z]
    d_x, d_y, d_z = hx_hy_hz(head_scale)
    # head center sits at handle_x/2 + head_x/2 along +x; total span along x:
    total_x = h_x + d_x
    total_y = max(h_y, d_y)
    total_z = max(h_z, d_z)
    return [total_x, total_y, total_z]


def main():
    manifest = _sample_pool(args.types, args.per_type, args.seed)
    print(f"[gen_pool] sampled {len(manifest)} objects; converting URDF -> USD ...")

    # ---- convert each URDF to USD in this single sim session --------------------
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    for entry in manifest:
        urdf_path = os.path.join(_POOL_DIR, entry["urdf"])
        usd_dir = os.path.join(_POOL_DIR, entry["name"])
        cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=usd_dir,
            usd_file_name=f"{entry['name']}.usd",
            force_usd_conversion=True,
            # NOT instanceable: instanced prims don't get per-instance collision/physics
            # properties applied (breaks MultiAssetSpawnerCfg -> some objects fail to
            # register as rigid bodies). The objects are tiny single-link prims, so the
            # memory cost of non-instanceable is negligible.
            make_instanceable=False,
            fix_base=False,  # the object is free-floating
            merge_fixed_joints=False,
            joint_drive=None,  # single rigid link, no actuated joints
            # convex_hull (not decomposition): robust on thin shafts/heads -- convex
            # decomposition produced degenerate pieces that PhysX's GPU view rejected
            # (3/16 objects dropped). The hull loses the handle/head concavity but is fine
            # for grasping these simple primitive tools.
            collider_type="convex_hull",
        )
        UrdfConverter(cfg)
    print(f"[gen_pool] converted {len(manifest)} objects to USD.")

    manifest_path = os.path.join(_POOL_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"objects": manifest}, f, indent=2)
    print(f"[gen_pool] wrote manifest: {manifest_path}")
    simulation_app.close()


if __name__ == "__main__":
    main()
