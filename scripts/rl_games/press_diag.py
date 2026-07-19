import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--steps",type=int,default=200); p.add_argument("--seed",type=int,default=1)
AppLauncher.add_app_launcher_args(p); a=p.parse_args()
app=AppLauncher(a).app
import torch, gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
import xhand_inhand.tasks  # noqa
T="Pick-Tool-Token-Direct-v0"
cfg=parse_env_cfg(T,device="cuda:0",num_envs=1); cfg.seed=a.seed
cfg.terminate_on_arm_table_contact=False
ag=load_cfg_from_registry(T,"rl_games_cfg_entry_point"); ag["params"]["seed"]=a.seed
ag["params"]["config"]["full_experiment_name"]="0_pd"
env=gym.make(T,cfg=cfg,render_mode=None)
env=RlGamesVecEnvWrapper(env,ag["params"]["config"]["device"],5.0,1.0,None,True)
vecenv.register("IsaacRlgWrapper",lambda cn,na,**kw:RlGamesGpuEnv(cn,na,**kw))
env_configurations.register("rlgpu",{"vecenv_type":"IsaacRlgWrapper","env_creator":lambda **kw:env})
ag["params"]["load_checkpoint"]=True; ag["params"]["load_path"]=a.checkpoint; ag["params"]["config"]["num_actors"]=1
r=Runner(); r.load(ag); pl=r.create_player(); pl.restore(a.checkpoint); pl.reset()
u=env.unwrapped; obs=env.reset()
if isinstance(obs,dict): obs=obs["obs"]
_=pl.get_batch_size(obs,1)
bn=u.robot.body_names
tbl=float(u._object_bottom_ref)
arm=[bn.index(f"link{i}") for i in range(1,9)]
palm=[bn.index("palm")]
tips=[bn.index(n) for n in u.cfg.ee_body_names]
prox=[i for i in range(len(bn)) if i not in arm and i not in palm and i not in tips and i>=9]
def minz(ids): 
    return (u.robot.data.body_pos_w[0,ids,2]-u.scene.env_origins[0,2]).min().item()
print(f"table plane z={tbl:.4f}")
print("step lift  objMinCorner_aboveTable  armZ  palmZ  proxZ  tipZ")
for t in range(a.steps):
    with torch.inference_mode():
        obs=pl.obs_to_torch(obs); act=pl.get_action(obs,is_deterministic=pl.is_deterministic)
        obs,_,d,_=env.step(act)
    lift=(u.object_pos_w[0,2]-u.scene.env_origins[0,2]-u.object_default_z[0]).item()
    clr=(u._object_min_corner_z()[0]-u._object_bottom_ref).item()  # object lowest corner above table
    if t%15==0:
        print(f"{t:4d} {lift:+.3f}  {clr:+.4f}                {minz(arm):+.3f} {minz(palm):+.3f} {minz(prox):+.3f} {minz(tips):+.3f}")
env.close()
