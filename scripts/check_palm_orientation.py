"""Quantify the hand orientation at the home pose vs the table tool.

Resets from-table (tool on the table, arm at home), then reports whether the palm's
grasp normal points TOWARD the tool (palm faces it) or away (back of hand faces it),
and where the fingertips / palm-center land relative to the tool.

Run: conda activate env_isaaclab; python scripts/check_palm_orientation.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import xhand_inhand.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402


def main():
    cfg = parse_env_cfg("SimToolReal-Direct-v0", num_envs=args.num_envs)
    cfg.inhand_reset_frac = 0.0  # all from-table
    env = gym.make("SimToolReal-Direct-v0", cfg=cfg).unwrapped
    env.reset()
    env._compute_intermediate_values()
    dev = env.device

    palm_rot = env.palm_rot  # (N,4) wxyz, palm body
    palm_c = env.palm_center_pos  # (N,3)
    ft = env.fingertip_pos  # (N,5,3)
    obj = env.object_pos  # (N,3)

    # palm "grasp normal" = the direction palm_offset points in (palm frame) -> world
    off = torch.tensor(cfg.palm_offset, device=dev).float()
    off_unit = off / off.norm()
    palm_normal_w = quat_apply(palm_rot, off_unit.unsqueeze(0).expand(env.num_envs, -1))  # (N,3)

    palm_to_obj = obj - palm_c
    palm_to_obj_u = palm_to_obj / (palm_to_obj.norm(dim=-1, keepdim=True) + 1e-8)
    align = (palm_normal_w * palm_to_obj_u).sum(dim=-1)  # +1 palm faces obj, -1 back faces obj

    ft_centroid = ft.mean(dim=1)  # (N,3)
    ft_spread = (ft - ft_centroid.unsqueeze(1)).norm(dim=-1).mean(dim=1)  # mean fingertip radius

    print("\n================ PALM ORIENTATION @ HOME POSE ================", flush=True)
    print(f"  palm_center world pos (mean):   {palm_c.mean(0).tolist()}", flush=True)
    print(f"  object   world pos (mean):      {obj.mean(0).tolist()}", flush=True)
    print(f"  palm grasp-normal world (mean): {palm_normal_w.mean(0).tolist()}", flush=True)
    print(f"  --> normal world Z (mean): {palm_normal_w[:,2].mean():.3f}  (NEGATIVE = palm faces DOWN/table = good)", flush=True)
    print(f"  alignment normal . (palm->obj): mean={align.mean():.3f} min={align.min():.3f} max={align.max():.3f}", flush=True)
    print(f"      (+1 = palm faces the tool; ~0 = sideways; -1 = BACK of hand faces the tool)", flush=True)
    print(f"  palm_center -> object dist (mean): {palm_to_obj.norm(dim=-1).mean():.3f} m", flush=True)
    print(f"  fingertip centroid -> object dist (mean): {(ft_centroid-obj).norm(dim=-1).mean():.3f} m", flush=True)
    print(f"  fingertip spread (mean radius): {ft_spread.mean():.3f} m", flush=True)
    print(f"  palm_center vs fingertip_centroid dist (mean): {(palm_c-ft_centroid).norm(dim=-1).mean():.3f} m", flush=True)
    frac_palm_faces = (align > 0.3).float().mean()
    frac_back_faces = (align < -0.3).float().mean()
    print(f"  >> frac palm-faces-tool: {frac_palm_faces:.2f} | frac BACK-faces-tool: {frac_back_faces:.2f}", flush=True)
    print("=============================================================\n", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
