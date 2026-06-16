"""Check whether the in-hand bootstrap grasp actually HOLDS the tool.

Resets all envs in-hand (object placed in the closed hand at the palm center), then
commands the fingers to stay closed for ~50 steps and measures how far the tool drifts
from the palm center. If it stays small -> the grasp holds; if it grows / z drops -> the
tool falls out (palm_offset / closed-pose calibration is wrong).

Run: conda activate env_isaaclab; python scripts/check_grasp_hold.py --headless
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


def main():
    cfg = parse_env_cfg("SimToolReal-Direct-v0", num_envs=args.num_envs)
    cfg.inhand_reset_frac = 1.0  # force ALL envs in-hand
    cfg.use_action_delay = False
    env = gym.make("SimToolReal-Direct-v0", cfg=cfg).unwrapped
    # reset #1 captures the home-pose palm center (in-hand placement is skipped until then);
    # reset #2 actually places every tool in the closed hand.
    env.reset()
    env.reset()
    dev = env.device

    # action that keeps fingers closing (arm 0, hand toward upper limit)
    act = torch.zeros((env.num_envs, env.num_actions), device=dev)
    act[:, env.num_arm_dofs :] = 0.9  # hand scale ~ near upper limit

    # palm center at reset
    env._compute_intermediate_values()
    palm0 = env.palm_center_pos.clone()
    obj0 = env.object_pos.clone()
    print(f"\n[grasp_hold] start: object-vs-palm dist mean={ (obj0-palm0).norm(dim=-1).mean():.3f} m", flush=True)
    print(f"[grasp_hold] start object z mean={obj0[:,2].mean():.3f}", flush=True)

    for step in range(50):
        env.step(act)
        if (step + 1) % 10 == 0:
            env._compute_intermediate_values()
            d = (env.object_pos - env.palm_center_pos).norm(dim=-1)
            z = env.object_pos[:, 2]
            held = (d < 0.12).float().mean()  # within 12cm of palm = plausibly held
            print(
                f"[grasp_hold] step {step+1:2d}: obj-palm dist mean={d.mean():.3f} max={d.max():.3f} "
                f"| obj z mean={z.mean():.3f} | held_frac(<12cm)={held:.2f}",
                flush=True,
            )
    print("[grasp_hold] DONE. If dist stays small and z high -> grasp holds; if dist grows / z drops to table(~0.42) -> tool fell out.", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
