"""
Build the CrossDex-style eigengrasp -> xhand retargeting network.

Faithful port of CrossDex/retargeting/{generate_dataset,train_retargeting_nn}.py,
specialized to our xhand (DexPilot). Two stages:

  1) generate : sample random 9-dim eigengrasp coords (smoothly interpolated, as in
                CrossDex), decode to 45-dim MANO axis-angle, MANO-FK to keypoints,
                DexPilot-retarget to 12 xhand joints. Saves dataset/*.pkl .
  2) train    : MLP RetargetingNN(45 -> 12), MSE, mirrors CrossDex. Saves
                models/retarget_nn_xhand.pt  +  models/retarget_nn_xhand_meta.pkl
                (joint_names in retargeter output order, pca, scaling, etc.)

The ONLINE env only needs the .pt + pca + joint_names (no manopth / dex_retargeting).

Run inside env_isaaclab:
  python build_retarget_nn.py generate --n-data 200000
  python build_retarget_nn.py train
  python build_retarget_nn.py all --n-data 200000
"""
import os
import sys
import time
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MANO_ROOT = HERE / "mano-models"  # holds MANO_RIGHT.pkl (licensed asset, kept with the tooling)
XHAND_URDF_DIR = REPO / "source/xhand_inhand/xhand_inhand/assets/xhand2R32"
PCA_FN = HERE / "assets/pca_9_grab.pkl"
CFG_FN = HERE / "configs/xhand_right_dexpilot.yml"
DATASET_DIR = HERE / "dataset"
MODELS_DIR = HERE / "models"
DATASET_FN = DATASET_DIR / "retarget_xhand_grab_dexpilot_random.pkl"
MODEL_FN = MODELS_DIR / "retarget_nn_xhand.pt"
META_FN = MODELS_DIR / "retarget_nn_xhand_meta.pkl"

MANO_TIP_SCALE = 1000.0  # manopth returns mm


def load_pca(fn):
    d = pickle.load(open(fn, "rb"))
    return (d["eigen_vectors"], d["min_values"], d["max_values"], d["D_mean"], d["D_std"])


def reconstruct_mano_pose45(eigen_vectors, D_mean, D_std, coords):
    return np.dot(coords, eigen_vectors) * D_std + D_mean


def build_retargeting():
    from dex_retargeting.retargeting_config import RetargetingConfig
    RetargetingConfig.set_default_urdf_dir(str(XHAND_URDF_DIR))
    cfg = RetargetingConfig.load_from_file(str(CFG_FN), override=dict(add_dummy_free_joint=True))
    return cfg.build(), cfg


# ----------------------------------------------------------------------------- generate
def generate(n_data=200000, seed=0):
    from manopth.manolayer import ManoLayer
    DATASET_DIR.mkdir(exist_ok=True)
    eig, vmin, vmax, dmean, dstd = load_pca(PCA_FN)
    mano = ManoLayer(mano_root=str(MANO_ROOT), use_pca=False, ncomps=45, flat_hand_mean=True)
    retargeting, cfg = build_retargeting()
    out_joint_names = retargeting.joint_names[6:]  # output order (drop 6 dummy DOF)
    idx = retargeting.optimizer.target_link_human_indices
    origin_idx, task_idx = idx[0, :], idx[1, :]
    print("output joint order:", out_joint_names)

    rng = np.random.default_rng(seed)
    mano_pose45 = np.empty((n_data, 45), dtype=np.float32)
    robot_joint_pos = np.empty((n_data, len(out_joint_names)), dtype=np.float32)

    last_act = np.zeros(9)
    max_step_size = 0.05  # smooth sequential trajectory => better DexPilot temporal results
    n = 0
    t0 = time.time()
    retargeting.reset()
    while n < n_data:
        next_act = rng.uniform(vmin, vmax)
        dist = np.linalg.norm(next_act - last_act)
        num_steps = max(int(np.ceil(dist / max_step_size)), 1)
        for t in range(1, num_steps + 1):
            act = last_act + (next_act - last_act) * t / num_steps
            data = reconstruct_mano_pose45(eig, dmean, dstd, act)
            pose = torch.zeros([1, 48])
            pose[0, 3:] = torch.tensor(data, dtype=torch.float32)
            _, hand_joints = mano(pose)
            jp = np.array(hand_joints[0] / MANO_TIP_SCALE)
            ref_value = jp[task_idx, :] - jp[origin_idx, :]
            qpos = retargeting.retarget(ref_value)[6:]
            mano_pose45[n] = data.astype(np.float32)
            robot_joint_pos[n] = qpos.astype(np.float32)
            n += 1
            if n >= n_data:
                break
        last_act = next_act
        if n % 5000 < num_steps:
            rate = n / (time.time() - t0 + 1e-9)
            print(f"  {n}/{n_data}  ({rate:.0f}/s, eta {(n_data-n)/max(rate,1):.0f}s)")

    ret = {
        "robot_name": "xhand_right",
        "joint_names": list(out_joint_names),
        "mano_pose45": mano_pose45,
        "robot_joint_pos": robot_joint_pos,
        "urdf_path": cfg.urdf_path,
        "scaling_factor": cfg.scaling_factor if hasattr(cfg, "scaling_factor") else None,
    }
    with open(DATASET_FN, "wb") as f:
        pickle.dump(ret, f)
    print(f"saved {DATASET_FN}  ({n_data} samples, {time.time()-t0:.0f}s)")
    return ret


# ----------------------------------------------------------------------------- model
class RetargetingNN(nn.Module):
    def __init__(self, robot_dim, mano_dim=45, hidden_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(mano_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, robot_dim),
        )

    def forward(self, x):
        return self.model(x)


# ----------------------------------------------------------------------------- train
def train(num_epochs=200, batch_size=256, device="cuda"):
    from torch.utils.data import DataLoader, TensorDataset
    MODELS_DIR.mkdir(exist_ok=True)
    data = pickle.load(open(DATASET_FN, "rb"))
    X = torch.tensor(data["mano_pose45"], dtype=torch.float32)
    Y = torch.tensor(data["robot_joint_pos"], dtype=torch.float32)
    robot_dim = Y.shape[1]
    n = X.shape[0]
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g)
    n_val = max(int(n * 0.025), 1)
    val_i, tr_i = perm[:n_val], perm[n_val:]
    tr = TensorDataset(X[tr_i], Y[tr_i])
    va = TensorDataset(X[val_i], Y[val_i])
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=1024, shuffle=False)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = RetargetingNN(robot_dim=robot_dim).to(dev)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters())
    print(f"train {len(tr)} / val {len(va)}  robot_dim={robot_dim}  device={dev}")

    best = float("inf")
    for ep in range(num_epochs):
        model.train()
        run = 0.0
        for xb, yb in tr_loader:
            opt.zero_grad()
            loss = crit(model(xb.to(dev)), yb.to(dev))
            loss.backward()
            opt.step()
            run += loss.item() * xb.size(0)
        tr_loss = run / len(tr)
        model.eval()
        vrun = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                vrun += crit(model(xb.to(dev)), yb.to(dev)).item() * xb.size(0)
        v_loss = vrun / len(va)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{num_epochs}  train {tr_loss:.5f}  val {v_loss:.5f}  "
                  f"(val rad rmse ~{v_loss**0.5:.4f})")
        if v_loss < best:
            best = v_loss
            torch.save(model.state_dict(), MODEL_FN)
    # meta for online use
    eig, vmin, vmax, dmean, dstd = load_pca(PCA_FN)
    meta = {
        "joint_names": data["joint_names"],     # output order of the NN
        "robot_dim": robot_dim,
        "hidden_dim": 512,
        "mano_dim": 45,
        "eigen_vectors": eig.astype(np.float32),
        "min_values": vmin.astype(np.float32),
        "max_values": vmax.astype(np.float32),
        "D_mean": dmean.astype(np.float32),
        "D_std": dstd.astype(np.float32),
        "n_eigengrasps": 9,
        "scaling_factor": data.get("scaling_factor"),
    }
    with open(META_FN, "wb") as f:
        pickle.dump(meta, f)
    print(f"saved {MODEL_FN} (best val {best:.5f}) and {META_FN}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    kw = {}
    for a in sys.argv[2:]:
        if a.startswith("--"):
            k, v = a[2:].split("=") if "=" in a else (a[2:], sys.argv[sys.argv.index(a) + 1])
            kw[k.replace("-", "_")] = v
    n_data = int(kw.get("n_data", 200000))
    if cmd in ("generate", "all"):
        generate(n_data=n_data)
    if cmd in ("train", "all"):
        train()
