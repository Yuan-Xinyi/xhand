#!/usr/bin/env python3
"""Verify the deployment obs builder + one-library FK against the sim dump.

Two checks against RealExperiments/repose_probe.npz (from repose_probe_dump.py):
  1. obs layout: rebuild the 34-D obs from raw sim quantities with build_obs()
     and diff against the env's own obs tensor.
  2. FK: palm-relative fingertip positions from the one library vs Isaac.

Run in the one env (or any env with numpy; FK check needs `one`):
    /home/lqin/miniconda3/envs/one/bin/python RealExperiments/verify_repose_obs.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foundationpose_repose_real as rr  # noqa: E402


def main():
    npz = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "repose_probe.npz"))

    # ---- check 1: obs layout ------------------------------------------------
    print("== check 1: obs layout ==")
    worst = 0.0
    for k in range(npz["q"].shape[0]):
        obs_mine = rr.build_obs(
            npz["tip_pos"][k], npz["obj_pos"][k], npz["obj_quat"][k], npz["goal_quat"][k], npz["actions"][k]
        )
        obs_sim = npz["obs"][k]
        # quat sign ambiguity in the rel-quat block: compare min(|d|, |sum|)
        d = np.abs(obs_mine - obs_sim)
        d[18:22] = np.minimum(d[18:22], np.abs(obs_mine[18:22] + obs_sim[18:22]))
        worst = max(worst, float(d.max()))
    print(f"max |obs_mine - obs_sim| over {npz['q'].shape[0]} configs: {worst:.2e}")
    assert worst < 1e-4, "obs layout mismatch!"
    print("obs layout OK\n")

    # ---- check 2: URDF FK ----------------------------------------------------
    print("== check 2: URDF FK vs Isaac fingertip positions ==")
    kin = rr.UrdfKinematics()
    worst = 0.0
    for k in range(npz["q"].shape[0]):
        q_isaac = npz["q"][k].astype(np.float32)
        kin.update(rr.DEFAULT_ARM_Q, q_isaac)
        base_T_palm = kin.palm_tf_base()
        tips_base = kin.fingertip_pos_base()
        # palm-relative (arm pose cancels out)
        inv = rr.tf_inv(base_T_palm)
        tips_palm_one = (inv[:3, :3] @ tips_base.T).T + inv[:3, 3]

        env_T_palm = rr.tf_from_pos_quat(npz["palm_pos"][k], npz["palm_quat"][k])
        inv_sim = rr.tf_inv(env_T_palm)
        tips_palm_sim = (inv_sim[:3, :3] @ npz["tip_pos"][k].T).T + inv_sim[:3, 3]

        err = np.linalg.norm(tips_palm_one - tips_palm_sim, axis=1)
        worst = max(worst, float(err.max()))
        if k == 0:
            for i, name in enumerate(rr.FINGERTIP_BODIES):
                print(f"  {name:16s} one={np.array2string(tips_palm_one[i], precision=4)} "
                      f"sim={np.array2string(tips_palm_sim[i], precision=4)} err={err[i] * 1000:.2f} mm")
    print(f"max fingertip position error over all configs: {worst * 1000:.2f} mm")
    if worst > 0.005:
        print("WARNING: FK mismatch > 5 mm — check one-lib XHand model vs Isaac URDF!")
    else:
        print("FK OK")


if __name__ == "__main__":
    main()
