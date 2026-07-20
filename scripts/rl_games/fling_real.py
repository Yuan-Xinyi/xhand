"""Launch the ungrasped object and verify real-contact lift reward remains zero."""

import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); a=p.parse_args()
app=AppLauncher(a).app
import torch, gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import xhand_inhand.tasks  # noqa
T="Pick-Tool-Token-Direct-v0"
cfg=parse_env_cfg(T,device="cuda:0",num_envs=1); env=gym.make(T,cfg=cfg,render_mode=None); u=env.unwrapped
env.reset(); dev=u.device
def lv(k): return float(u.extras.get("log",{}).get(k,float('nan')))
act=torch.zeros((1,u.cfg.action_space),device=dev)
for _ in range(10): env.step(act)
print("REAL contact (no monkeypatch). Object flung; hand at home (never grips).")
print("phase        clr    obj_contact q_wrap hold_q  slip_lin  latched R_lift  ungated")
def rep(tag):
    ungated = u.cfg.lift_progress_scale * min(max(lv("clearance_mean"), 0) / u.cfg.lift_success_height, 1)
    print(
        f"{tag:12s} {lv('clearance_mean'):+.3f}  {lv('valid_contact_frac'):.2f}       "
        f"{lv('q_wrap_mean'):.3f}  {lv('hold_quality_mean'):.3f}   {lv('slip_lin_mean'):6.2f}   "
        f"{lv('is_grasped_phase_frac'):.0f}     {lv('r_lift_mean'):6.3f}  {ungated:6.3f}"
    )
rep("rest")
for step in range(1,60):
    if step<=3:
        vel=torch.zeros((1,6),device=dev); vel[0,2]=3.0; u.object.write_root_velocity_to_sim(vel)
    env.step(act)
    if step in (4,8,15,25): rep(f"fling t={step}")
env.close()
app.close()
