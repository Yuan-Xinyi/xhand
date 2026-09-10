"""Sim2real readiness for the hammer NUDGE policy: evaluate under degraded observations.

Degradations model the real pipeline (FoundationPose + SDKs):
  * object pose noise (pos sigma, yaw sigma) on obs cols 56:59 (pos_b) and 59:63 (quat)
  * fixed observation delay of N control steps (whole vector)
  * zeroed force-derived features: cols 97:105 (finger forces, close/wrap/hold quality)
    and 109:115 (palm-frame slips) -- unavailable or unreliable on real hardware

Success counting: nudge option mode terminates with +100 (success) or -100 (failure);
timeouts truncate.  A terminated episode with reward > 50 at the final step is a success.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="Pick-Hammer-Token-Direct-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=512)
parser.add_argument("--pos_noise", type=float, default=0.0, help="object pos noise sigma (m)")
parser.add_argument("--yaw_noise_deg", type=float, default=0.0)
parser.add_argument("--delay", type=int, default=0, help="observation delay in control steps")
parser.add_argument("--zero_force", action="store_true")
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--video_folder", type=str, default="", help="record success clips (forces num_envs=1)")
parser.add_argument("--max_clips", type=int, default=3)
parser.add_argument("--fps", type=int, default=25)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import xhand_inhand.tasks  # noqa: F401, E402

_HERE = Path(__file__).resolve()
_REPO = Path("/disk2/xhand_inhand/xhand_inhand/.claude/worktrees/compassionate-chaum-2333b7")
sys.path.insert(0, str(_REPO / "scripts" / "flashsac"))
sys.path.insert(0, str(_REPO / "scripts" / "rl_games"))
import agent_bridge  # noqa: E402,F401  (installs the pinned flash_rl path)
from flash_rl.agents.flashSAC.network import FlashSACActor  # noqa: E402

_COMPILED_PREFIX = "_orig_mod."


def load_actor(checkpoint: Path, device: torch.device) -> FlashSACActor:
    payload = torch.load(checkpoint / "actor.pt", map_location="cpu", weights_only=True)
    state = payload["network_state_dict"]
    prefixed = [str(k).startswith(_COMPILED_PREFIX) for k in state]
    strip = all(prefixed) and len(prefixed) > 0
    canonical = {k.removeprefix(_COMPILED_PREFIX) if strip else k: v for k, v in state.items()}
    hidden = canonical["embedder.w.w.weight"].shape[0]
    blocks = len({k.split(".")[1] for k in canonical if k.startswith("encoder.")})
    actor = FlashSACActor(num_blocks=blocks, input_dim=115, hidden_dim=hidden, action_dim=21)
    actor.load_state_dict(canonical)
    return actor.to(device).eval()


def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """(N,) yaw -> (N,4) wxyz quat about z."""
    half = yaw * 0.5
    q = torch.zeros((yaw.shape[0], 4), device=yaw.device)
    q[:, 0] = torch.cos(half)
    q[:, 3] = torch.sin(half)
    return q


def quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def degrade(obs: torch.Tensor) -> torch.Tensor:
    out = obs.clone()
    n = out.shape[0]
    if args_cli.pos_noise > 0:
        out[:, 56:59] += torch.randn((n, 3), device=out.device) * args_cli.pos_noise
    if args_cli.yaw_noise_deg > 0:
        sigma = math.radians(args_cli.yaw_noise_deg)
        dq = yaw_quat(torch.randn((n,), device=out.device) * sigma)
        out[:, 59:63] = quat_mul_wxyz(dq, out[:, 59:63])
    if args_cli.zero_force:
        out[:, 97:105] = 0.0
        out[:, 109:115] = 0.0
    return out


def main():
    task = args_cli.task
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    cfg.nudge_option_mode = True
    cfg.nudge_pregrasp_min = 0.30
    cfg.nudge_spawn_blend_min = 0.0
    cfg.nudge_spawn_blend_max = 0.0
    cfg.episode_length_s = 6.0
    if args_cli.video_folder:
        args_cli.num_envs = 1
        cfg.scene.num_envs = 1
        cfg.viewer.eye = (1.9, 0.95, 0.9)
        cfg.viewer.lookat = (0.5, 0.0, 0.32)
        cfg.viewer.origin_type = "world"
    env = gym.make(task, cfg=cfg, render_mode="rgb_array" if args_cli.video_folder else None)
    dev = env.unwrapped.device
    actor = load_actor(Path(args_cli.checkpoint), torch.device(str(dev)))

    torch.manual_seed(args_cli.seed)
    obs_d, _ = env.reset(seed=args_cli.seed)
    buf: deque = deque(maxlen=args_cli.delay + 1)
    succ = fail = timeout = 0
    frames, saved = [], 0
    if args_cli.video_folder:
        import os
        os.makedirs(args_cli.video_folder, exist_ok=True)
        for _ in range(8):
            env.unwrapped.render()
    with torch.inference_mode():
        while succ + fail + timeout < args_cli.episodes:
            buf.append(obs_d["policy"].clone())
            fed = degrade(buf[0])  # oldest = delayed by len-1 steps
            mean, _ = actor.get_mean_and_std(fed, training=False)
            act = torch.tanh(mean)
            obs_d, rew, term, trunc, _ = env.step(act)
            if args_cli.video_folder and saved < args_cli.max_clips:
                frames.append(env.unwrapped.render())
                if bool(term[0]) or bool(trunc[0]):
                    if bool(term[0] & (rew[0] > 50.0)) and len(frames) > 10:
                        import imageio
                        import numpy as np
                        path = f"{args_cli.video_folder}/nudge_forcefree_{saved+1}.mp4"
                        imageio.mimwrite(path, [np.asarray(f) for f in frames], fps=args_cli.fps, macro_block_size=None)
                        saved += 1
                        print(f"[video] saved {path}", flush=True)
                    frames = []
            succ += int((term & (rew > 50.0)).sum())
            fail += int((term & (rew <= 50.0)).sum())
            timeout += int((trunc & ~term).sum())

    total = succ + fail + timeout
    print(
        f"[S2R] pos_noise={args_cli.pos_noise} yaw_deg={args_cli.yaw_noise_deg} "
        f"delay={args_cli.delay} zero_force={args_cli.zero_force} | "
        f"episodes={total} success={succ} ({100*succ/total:.1f}%) "
        f"fail={fail} ({100*fail/total:.1f}%) timeout={timeout} ({100*timeout/total:.1f}%)",
        flush=True,
    )
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
