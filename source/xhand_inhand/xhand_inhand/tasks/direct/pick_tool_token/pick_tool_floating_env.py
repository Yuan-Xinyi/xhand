"""Bounded free-root XHand ablation for PickTool."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .pick_tool_floating_env_cfg import PickToolFloatingXHandEnvCfg
from .pick_tool_token_env import PickToolTokenEnv


class PickToolFloatingXHandEnv(PickToolTokenEnv):
    cfg: PickToolFloatingXHandEnvCfg

    def __init__(self, cfg: PickToolFloatingXHandEnvCfg, render_mode: str | None = None, **kwargs):
        self._floating_root_velocity = None
        super().__init__(cfg, render_mode, **kwargs)
        self._floating_root_velocity = torch.zeros((self.num_envs, 6), device=self.device)
        self._workspace_min = torch.tensor(cfg.floating_workspace_min, device=self.device)
        self._workspace_max = torch.tensor(cfg.floating_workspace_max, device=self.device)

    def _preprocess_virtual_arm_action(self, arm_action: torch.Tensor) -> None:
        velocity = torch.empty_like(arm_action)
        velocity[:, :3] = arm_action[:, :3] * self.cfg.floating_max_linear_speed
        velocity[:, 3:] = arm_action[:, 3:] * self.cfg.floating_max_angular_speed
        if self._floating_root_velocity is None:
            self._floating_root_velocity = velocity
            return
        local_pos = self.robot.data.root_pos_w - self.scene.env_origins
        outward_low = (local_pos <= self._workspace_min) & (velocity[:, :3] < 0.0)
        outward_high = (local_pos >= self._workspace_max) & (velocity[:, :3] > 0.0)
        velocity[:, :3] = torch.where(outward_low | outward_high, 0.0, velocity[:, :3])
        self._floating_root_velocity.copy_(velocity)

    def _apply_action(self) -> None:
        super()._apply_action()
        if self._floating_root_velocity is not None:
            self.robot.write_root_velocity_to_sim(self._floating_root_velocity)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            ids = self.robot._ALL_INDICES
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        root_state = self.robot.data.default_root_state[ids].clone()
        root_state[:, :3] += self.scene.env_origins[ids]
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=ids)
        super()._reset_idx(ids)
        if self._floating_root_velocity is not None:
            self._floating_root_velocity[ids] = 0.0
