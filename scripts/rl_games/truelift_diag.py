import argparse, os, numpy as np
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--steps",type=int,default=195); p.add_argument("--seed",type=int,default=2)
AppLauncher.add_app_launcher_args(p); a=p.parse_args()
app=AppLauncher(a).app
import torch, gymnasium as gym
from isaaclab.utils.math import quat_apply
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
import xhand_inhand.tasks  # noqa
from xhand_inhand.tasks.direct.pick_tool_token.tool_asset import TOOL_OBJ, TOOL_REST_Z, TOOL_REST_QUAT
# load mesh vertices from the obj (v lines)
V=[]
for ln in open(TOOL_OBJ):
    if ln.startswith("v "):
        _,x,y,z=ln.split()[:4]; V.append((float(x),float(y),float(z)))
V=np.array(V,dtype=np.float32); print(f"mesh verts: {V.shape}")
T="Pick-Tool-Token-Direct-v0"
cfg=parse_env_cfg(T,device="cuda:0",num_envs=1); cfg.seed=a.seed; cfg.terminate_on_arm_table_contact=False
ag=load_cfg_from_registry(T,"rl_games_cfg_entry_point"); ag["params"]["seed"]=a.seed; ag["params"]["config"]["full_experiment_name"]="0_tl"
env=gym.make(T,cfg=cfg,render_mode=None); env=RlGamesVecEnvWrapper(env,ag["params"]["config"]["device"],5.0,1.0,None,True)
vecenv.register("IsaacRlgWrapper",lambda cn,na,**kw:RlGamesGpuEnv(cn,na,**kw)); env_configurations.register("rlgpu",{"vecenv_type":"IsaacRlgWrapper","env_creator":lambda **kw:env})
ag["params"]["load_checkpoint"]=True; ag["params"]["load_path"]=a.checkpoint; ag["params"]["config"]["num_actors"]=1
r=Runner(); r.load(ag); pl=r.create_player(); pl.restore(a.checkpoint); pl.reset()
u=env.unwrapped; obs=env.reset()
if isinstance(obs,dict): obs=obs["obs"]
_=pl.get_batch_size(obs,1)
dev=u.device; Vt=torch.tensor(V,device=dev)          # (M,3) local verts
# table surface = true mesh min-z at REST pose
restq=torch.tensor(TOOL_REST_QUAT,device=dev).view(1,4).expand(Vt.shape[0],4)
rest_world=quat_apply(restq,Vt); table_top=(rest_world[:,2]+TOOL_REST_Z).min().item()
print(f"TRUE table surface (env-local z, from rest mesh) = {table_top:.4f}")
print(f"box _object_bottom_ref = {float(u._object_bottom_ref):.4f}  (diff = box is this far BELOW true table)")
print("step rootRise tilt°  BOXclr  TRUEclr")
for t in range(a.steps):
    with torch.inference_mode():
        obs=pl.obs_to_torch(obs); act=pl.get_action(obs,is_deterministic=pl.is_deterministic); obs,_,d,_=env.step(act)
    pos=u.object_pos_w[0]; q=u.object_quat_w[0]
    world=quat_apply(q.view(1,4).expand(Vt.shape[0],4),Vt)+pos.view(1,3)   # (M,3) world
    true_minz=(world[:,2]-u.scene.env_origins[0,2]).min().item()
    true_clr=true_minz-table_top
    root_rise=(pos[2]-u.scene.env_origins[0,2]-u.object_default_z[0]).item()
    box_clr=(u._object_min_corner_z()[0]-u._object_bottom_ref).item()
    qdot=abs(float((q*torch.tensor(TOOL_REST_QUAT,device=dev)).sum()))
    tilt=np.degrees(np.arccos(np.clip(2*qdot*qdot-1,-1,1)))
    if t%15==0: print(f"{t:4d}  {root_rise:+.3f}  {tilt:5.1f}  {box_clr:+.3f}  {true_clr:+.3f}")
env.close()
