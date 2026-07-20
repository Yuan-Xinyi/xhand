"""Worst-case fling test with legal wrap signals forced while transport slip stays physical."""

import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); a=p.parse_args()
app=AppLauncher(a).app
import torch, gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import xhand_inhand.tasks  # noqa
T="Pick-Tool-Token-Direct-v0"
cfg=parse_env_cfg(T,device="cuda:0",num_envs=1)
env=gym.make(T,cfg=cfg,render_mode=None); u=env.unwrapped
env.reset()
dev=u.device
def logval(k): 
    return float(u.extras.get("log",{}).get(k, float('nan')))
# Worst case: pretend the contact topology/geometry remains a legal wrap throughout the launch.  Keep
# the real COM-frame hold quality, so this isolates the transport gate rather than bypassing it.
real_signals = u._compute_grasp_signals
def forced_wrap_signals():
    signals = real_signals()
    forced_wrap = torch.full_like(signals["quality"], 0.8)
    signals["quality"] = forced_wrap
    signals["grasp_quality"] = torch.minimum(forced_wrap, signals["hold_quality"])
    signals["thumb_contact"] = torch.ones(1, dtype=torch.bool, device=dev)
    signals["other_contact_count"] = torch.full((1,), 3, dtype=torch.long, device=dev)
    signals["palm_facing"] = torch.ones(1, device=dev)
    signals["palm_score"] = torch.ones(1, device=dev)
    signals["alignment_score"] = torch.ones(1, device=dev)
    signals["opposition_raw"] = torch.ones(1, device=dev)
    return signals
u._compute_grasp_signals = forced_wrap_signals
act=torch.zeros((1,u.cfg.action_space),device=dev)
print("phase           clr    slip_lin hold_q grasp_q latched R_lift  ungated")
# settle
for _ in range(10): env.step(act)
lift_scale=u.cfg.lift_scale; H=u.cfg.lift_success_height
def report(tag):
    clr=logval("clearance_mean"); rl=logval("slip_lin_mean"); hq=logval("hold_quality_mean"); rlift=logval("r_lift_mean")
    ungated=lift_scale*min(max(clr,0)/H,1.0)
    print(
        f"{tag:14s} {clr:+.3f}   {rl:6.2f}  {hq:.3f}   {logval('grasp_quality_mean'):.3f}   "
        f"{logval('is_grasped_phase_frac'):.0f}     {rlift:6.3f}  {ungated:6.3f}"
    )
report("rest(on table)")
# FLING: write a big upward velocity to the object
for step in range(1,60):
    # keep re-injecting upward velocity for the first few steps to launch it
    if step<=3:
        vel=torch.zeros((1,6),device=dev); vel[0,2]=3.0  # 3 m/s up
        u.object.write_root_velocity_to_sim(vel)
    env.step(act)
    if step in (2,4,8,15,25,40):
        report(f"fling t={step}")
env.close()
app.close()
