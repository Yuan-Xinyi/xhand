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
u._get_rewards()
print("REAL contact (no monkeypatch). Object flung; hand at home (never grips).")
print("phase        clr    vc_frac  rel_lin  R_lift   R_lift_ungated")
def rep(tag):
    print(f"{tag:12s} {lv('clearance_mean'):+.3f}  {lv('valid_contact_frac'):.2f}    {lv('rel_lin_speed_mean'):6.2f}  {lv('r_lift_mean'):6.3f}   {u.cfg.lift_scale*min(max(lv('clearance_mean'),0)/u.cfg.lift_success_height,1):6.3f}")
rep("rest")
for step in range(1,60):
    if step<=3:
        vel=torch.zeros((1,6),device=dev); vel[0,2]=3.0; u.object.write_root_velocity_to_sim(vel)
    env.step(act); u._get_rewards()
    if step in (4,8,15,25): rep(f"fling t={step}")
env.close()
