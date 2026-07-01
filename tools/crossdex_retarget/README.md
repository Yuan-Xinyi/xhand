# CrossDex action tokenization → xhand retargeting

Ports the [CrossDex](https://github.com/PKU-RL/CrossDex) (ICLR 2025) eigengrasp action
tokenization + DexPilot retargeting to our **xhand**, so the RL policy outputs a
9-dim hand *token* (plus the 7 arm joint deltas) instead of 12 raw hand joint deltas.

## Pipeline

```
policy token (9, in [-1,1])
   └─ scale to eigengrasp coords  [min_values, max_values]          (per-PC bounds from GRAB)
        └─ pose45 = coords @ eigen_vectors * D_std + D_mean         (PCA decode, results/pca_9_grab.pkl)
             └─ RetargetingNN(45 → 12)                              (offline-trained MLP)
                  └─ reorder to xhand articulation joint order
                       └─ absolute hand joint targets (moving-average smoothed in the env)
```

Offline (this folder) we build the `45 → 12` network by:
`random eigengrasp coords → MANO-FK keypoints (manopth + MANO_RIGHT.pkl) → DexPilot
retarget (dex_retargeting + xhand config) → xhand 12 joints`, then fit the MLP.

Online (the env) only needs the trained `.pt` + meta `.pkl` (pure torch, batched on GPU);
**no manopth / dex_retargeting at RL time.**

## Files

- `assets/pca_9_grab.pkl` — CrossDex 9-dim eigengrasp basis (from their `results/`).
- `configs/xhand_right_dexpilot.yml` — DexPilot retargeting config for xhand
  (`scaling_factor=1.5`, swept). NOTE: DexPilot constrains only fingertip *positions*,
  so the 5 distal joints (`index_joint2, middle_joint1, ring_joint1, pinky_joint1,
  thumb_joint2`) stay at their regularized midpoint and are not driven by the token —
  the token controls the 7 proximal joints. Fine for a power grasp.
- `build_retarget_nn.py` — generate dataset + train NN. Outputs:
  - `models/retarget_nn_xhand.pt` — MLP state_dict (consumed by the env).
  - `models/retarget_nn_xhand_meta.pkl` — joint_names (NN output order), PCA basis,
    min/max/mean/std, dims (consumed by the env).
- `CrossDex_ref/` — upstream CrossDex clone, reference only.

## Rebuild (inside `conda activate env_isaaclab`)

```bash
cd tools/crossdex_retarget
python build_retarget_nn.py all --n-data 200000   # ~10 min generate + ~2 min train (GPU)
# or split:  python build_retarget_nn.py generate --n-data 200000 ;  python build_retarget_nn.py train
```

Dependencies already installed in `env_isaaclab`: `dex_retargeting==0.4.6`, `manopth`,
`chumpy` (patched for py3.11/numpy). The licensed `MANO_RIGHT.pkl` lives at
`tools/crossdex_retarget/mano-models/MANO_RIGHT.pkl` (git-ignored — download it from
https://mano.is.tue.mpg.de/ if missing).

## Train the tokenized task

```bash
python scripts/rl_games/train.py --task=Pick-Cube-Token-Direct-v0 --headless
# smoke test:
python scripts/check_task.py --task=Pick-Cube-Token-Direct-v0 --num_envs 4 --headless
```

`Pick-Cube-Token-Direct-v0` is identical to `Pick-Cube-Direct-v0` except the action space
is `16 = 7 arm deltas + 9 hand eigengrasp token` (obs `86`).
