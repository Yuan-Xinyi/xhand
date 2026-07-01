# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Online eigengrasp -> xhand-joint retargeting (CrossDex action tokenization).

Loads the offline-trained RetargetingNN + PCA eigengrasp basis and maps a batch of
9-dim eigengrasp tokens (already in coordinate space, i.e. AFTER min/max scaling) to
12-dim xhand joint targets. Pure torch, batched on GPU; no manopth / dex_retargeting
needed at RL time.

Pipeline (mirrors CrossDex EigenRetargetModel):
    coords (N,9) --clip[min,max]--> (N,9) --@E*std+mean--> mano_pose45 (N,45)
                 --MLP--> robot joints (N,12) in the network's output order.
"""
from __future__ import annotations

import pickle

import torch
import torch.nn as nn


class _RetargetingNN(nn.Module):
    """Same architecture as CrossDex retargeting/retargeting_nn_utils.RetargetingNN."""

    def __init__(self, robot_dim: int, mano_dim: int = 45, hidden_dim: int = 512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(mano_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, robot_dim),
        )

    def forward(self, x):
        return self.model(x)


class EigenRetarget:
    """Batched eigengrasp-token -> xhand-joint retargeter for use inside the env.

    Args:
        weights_path: path to the trained ``.pt`` state_dict.
        meta_path: path to the ``*_meta.pkl`` produced by build_retarget_nn.py.
        device: torch device string.
    """

    def __init__(self, weights_path: str, meta_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.joint_names: list[str] = list(meta["joint_names"])  # NN output order
        self.robot_dim: int = int(meta["robot_dim"])
        self.n_eigengrasps: int = int(meta["n_eigengrasps"])

        def _t(key):
            return torch.as_tensor(meta[key], dtype=torch.float32, device=self.device)

        self.eigen_vectors = _t("eigen_vectors")   # (9, 45)
        self.min_values = _t("min_values")          # (9,)
        self.max_values = _t("max_values")          # (9,)
        self.D_mean = _t("D_mean")                  # (45,)
        self.D_std = _t("D_std")                    # (45,)

        self.net = _RetargetingNN(
            robot_dim=self.robot_dim, mano_dim=int(meta["mano_dim"]), hidden_dim=int(meta["hidden_dim"])
        ).to(self.device)
        state = torch.load(weights_path, map_location=self.device)
        self.net.load_state_dict(state)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def coords_from_unit_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map policy action in [-1, 1] (N, 9) to eigengrasp coords in [min, max]."""
        a = action.clamp(-1.0, 1.0)
        return 0.5 * (a + 1.0) * (self.max_values - self.min_values) + self.min_values

    def eigengrasp_to_pose45(self, coords: torch.Tensor) -> torch.Tensor:
        coords = torch.max(torch.min(coords, self.max_values), self.min_values)
        return torch.matmul(coords, self.eigen_vectors) * self.D_std + self.D_mean

    @torch.no_grad()
    def retarget(self, coords: torch.Tensor) -> torch.Tensor:
        """coords (N, 9) eigengrasp coordinates -> (N, robot_dim) joints in NN order."""
        return self.net(self.eigengrasp_to_pose45(coords))

    @torch.no_grad()
    def retarget_from_unit_action(self, action: torch.Tensor) -> torch.Tensor:
        """action (N, 9) in [-1, 1] -> (N, robot_dim) joints in NN order."""
        return self.net(self.eigengrasp_to_pose45(self.coords_from_unit_action(action)))

    def permutation_to(self, target_joint_names: list[str]) -> torch.Tensor:
        """Index tensor ``perm`` s.t. ``nn_out[:, perm]`` is ordered as target_joint_names."""
        perm = [self.joint_names.index(n) for n in target_joint_names]
        return torch.tensor(perm, dtype=torch.long, device=self.device)
