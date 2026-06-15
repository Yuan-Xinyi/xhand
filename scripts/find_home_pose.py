"""FK-based search for an xArm7 home pose with >=30 cm hand-to-table clearance.

Spawns the table-mounted xArm7+XHand in N parallel envs, sets a different random arm
config per env, reads the REAL palm/fingertip world heights (Isaac Lab FK), and picks
the config whose lowest hand point is >=30 cm above the tabletop, sits over the front
workspace, and points the hand downward.

Run: conda activate env_isaaclab; python scripts/find_home_pose.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--rounds", type=int, default=40)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from xhand_inhand.robots import XARM7_XHAND_CFG  # noqa: E402

TABLE_TOP = 0.40
MIN_CLEAR = 0.30  # required hand clearance above the tabletop
TARGET_XY = (0.38, 0.0)  # palm should sit over the front workspace
PALM_Z_LO, PALM_Z_HI = TABLE_TOP + 0.30, TABLE_TOP + 0.45  # 30-45 cm up


@configclass
class _SceneCfg(InteractiveSceneCfg):
    robot = XARM7_XHAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    N = args.num_envs
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0"))
    scene_cfg = _SceneCfg(num_envs=N, env_spacing=2.5, replicate_physics=True)
    scene_cfg.robot.init_state.pos = (0.0, 0.0, TABLE_TOP)  # arm mounted on the tabletop
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]
    dev = robot.device

    arm_ids, arm_names = robot.find_joints("joint[1-7]", preserve_order=True)
    arm_ids_t = torch.tensor(arm_ids, device=dev)
    palm_idx = robot.find_bodies("palm")[0][0]
    ft_names = ["thumb_rota_link2", "index_rota_link2", "mid_link2", "ring_link2", "pinky_link2"]
    ft_ids = torch.tensor(robot.find_bodies(ft_names, preserve_order=True)[0], device=dev)

    lim = robot.root_physx_view.get_dof_limits().to(dev)[0]  # (ndof,2)
    lo, hi = lim[arm_ids_t, 0], lim[arm_ids_t, 1]

    # bias the search toward forward/down-reaching configs (j1,j3,j5,j7 near 0)
    bias_lo = torch.tensor([-0.3, -0.4, -0.4, 0.0, -0.4, -0.6, -0.4], device=dev)
    bias_hi = torch.tensor([0.3, 1.0, 0.4, 1.4, 0.4, 1.6, 0.4], device=dev)
    bias_lo = torch.maximum(bias_lo, lo)
    bias_hi = torch.minimum(bias_hi, hi)
    tgt = torch.tensor(TARGET_XY, device=dev)

    best_cost = 1e18
    best_j = None
    best_info = None
    total_valid = 0
    for r in range(args.rounds):
        cand = bias_lo + (bias_hi - bias_lo) * torch.rand((N, 7), device=dev)
        jp = robot.data.default_joint_pos.clone()
        jp[:, arm_ids_t] = cand
        jv = torch.zeros_like(jp)
        robot.write_joint_state_to_sim(jp, jv)
        robot.set_joint_position_target(jp)
        for _ in range(2):  # settle FK
            robot.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

        org = scene.env_origins
        palm = robot.data.body_pos_w[:, palm_idx]
        ft = robot.data.body_pos_w[:, ft_ids]
        palm_local_xy = palm[:, :2] - org[:, :2]
        palm_z = palm[:, 2]
        min_hand_z = torch.cat([palm[:, 2:3], ft[:, :, 2]], dim=1).min(dim=1).values
        mean_ft_z = ft[:, :, 2].mean(dim=1)

        palm_xy_err = torch.norm(palm_local_xy - tgt, dim=1)
        z_err = torch.relu(PALM_Z_LO - palm_z) + torch.relu(palm_z - PALM_Z_HI)
        down = torch.relu(mean_ft_z - palm_z + 0.02)
        cost = 2.0 * palm_xy_err + 2.0 * z_err + down
        valid = min_hand_z >= (TABLE_TOP + MIN_CLEAR)
        total_valid += int(valid.sum().item())
        cost = torch.where(valid, cost, torch.full_like(cost, 1e9))

        ci = torch.argmin(cost).item()
        if cost[ci].item() < best_cost:
            best_cost = cost[ci].item()
            best_j = cand[ci].tolist()
            best_info = (
                palm[ci].tolist(), min_hand_z[ci].item(), mean_ft_z[ci].item(),
                palm_local_xy[ci].tolist(),
            )
        print(f"  round {r+1}/{args.rounds}: valid so far={total_valid}, best_cost={best_cost:.4f}")

    print(f"\n[find_home_pose] total valid (>= {MIN_CLEAR*100:.0f} cm clearance): {total_valid}")
    if best_j is None or best_cost > 1e8:
        print("No candidate met the clearance constraint; widen the search bias.")
        simulation_app.close()
        return
    palm_pos, min_hand_z_b, mean_ft_z_b, palm_xy_b = best_info
    bj = best_j
    print("=========================================================")
    print("BEST HOME POSE (arm joints joint1..joint7, radians):")
    for nm, v in zip(arm_names, bj):
        print(f"    {nm}: {v:+.4f}")
    print(f"palm world pos:   ({palm_pos[0]:.3f}, {palm_pos[1]:.3f}, {palm_pos[2]:.3f})")
    print(f"min hand height:  {min_hand_z_b:.3f}  (clearance {min_hand_z_b - TABLE_TOP:.3f} m)")
    print(f"mean fingertip z: {mean_ft_z_b:.3f}  (< palm z -> hand points down)")
    print("=========================================================")
    simulation_app.close()


if __name__ == "__main__":
    main()
