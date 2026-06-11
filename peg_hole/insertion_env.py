"""CSAC peg-in-hole insertion env — subclass of the real Factory PegInsert env.

We DO NOT re-implement contact physics, OSC control, or assets. We subclass
``FactoryEnv`` (confirmed real class, factory_env.py:23) and only:
  * add a 6D force/torque wrench to the observation (hard requirement #2),
  * implement a staged reward + a sparse-reward switch,
  * expose ``contact_signals()`` for the sigma/tau schedule,
  * add a force-abort termination (safety layer).

Hard-constraint status (see README for the full discussion):
  #1 OSC delta-pose action  : SATISFIED unchanged — Factory's 6D action already
     maps to ctrl_target_fingertip_midpoint_* via compute_dof_torque (factory_env
     _apply_action). We add only a unit-box clamp in train.py.
  #2 6D wrench in obs        : Factory exposes NO measured wrench. We add a
     ContactSensor on the peg (/World/envs/env_.*/HeldAsset, .data.net_forces_w)
     for the measured 3D force and estimate the 3D moment as (r_peg-ee) x F.
     cfg.wrench_source="commanded" falls back to Factory's self.applied_wrench.
  #3 sigma low in contact    : handled by sigma_tau_schedule, fed by contact_signals().
"""

from __future__ import annotations

import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab.sensors import ContactSensor

from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG

try:
    from .env_cfg import InsertionEnvCfg
except ImportError:  # allow `python train.py` from inside the folder
    from env_cfg import InsertionEnvCfg

# Observation terms we ADD on top of Factory's defaults (name -> dim).
_EXTRA_OBS = {
    "wrench": 6,                # 6D force/torque (hard requirement #2)
    "fingertip_pos": 3,         # absolute end-effector position
    "held_pos_rel_fixed": 3,    # peg position relative to the hole opening
    "held_quat": 4,             # peg orientation (peg-in-gripper pose proxy)
    "gripper_width": 1,         # gripper opening
}
_OBS_ORDER = [
    "fingertip_pos_rel_fixed",  # relative hole pose (ee)
    "fingertip_quat",           # ee orientation
    "ee_linvel",
    "ee_angvel",
    "wrench",
    "fingertip_pos",
    "held_pos_rel_fixed",
    "held_quat",
    "gripper_width",
]


class InsertionEnv(FactoryEnv):
    cfg: InsertionEnvCfg

    def __init__(self, cfg: InsertionEnvCfg, render_mode: str | None = None, **kwargs):
        # Enable contact reporting on the peg so the ContactSensor produces data.
        try:
            cfg.task.held_asset.spawn.activate_contact_sensors = True
        except Exception as exc:  # pragma: no cover - depends on spawn cfg type
            print(f"[InsertionEnv] could not enable peg contact sensors: {exc}")

        # Register the extra observation terms BEFORE FactoryEnv.__init__ computes
        # observation_space = sum(OBS_DIM_CFG[o] for o in obs_order) + action_space.
        OBS_DIM_CFG.update(_EXTRA_OBS)
        STATE_DIM_CFG.update({"wrench": 6, "gripper_width": 1})
        cfg.obs_order = list(_OBS_ORDER)

        super().__init__(cfg, render_mode, **kwargs)

    # ------------------------------------------------------------ tensors
    def _init_tensors(self):
        super()._init_tensors()
        self.contact_wrench = torch.zeros((self.num_envs, 6), device=self.device)
        # Success/abort masks (for buffer-done grounding; NOT for env termination).
        self._was_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._success_edge = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._abort_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._engage_succ = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # gripper finger joint indices (panda_finger_joint1/2)
        self._finger_dof_idx = [
            self._robot.joint_names.index(n)
            for n in self._robot.joint_names
            if "finger" in n
        ]

    def _reset_buffers(self, env_ids):
        super()._reset_buffers(env_ids)
        # Clear the per-episode success latch so the bonus can fire again next episode.
        self._was_success[env_ids] = False

    # ------------------------------------------------------------- scene
    def _setup_scene(self):
        super()._setup_scene()
        self._held_contact = None
        # Only add the measured-force ContactSensor when explicitly requested.
        # (Its initialization is deferred to sim-reset and raises uncatchably if
        # the prim_path doesn't match a contact-reporter body, so we keep it opt-in.)
        if self.cfg.wrench_source == "contact":
            self._held_contact = ContactSensor(self.cfg.held_contact_sensor)
            self.scene.sensors["held_contact"] = self._held_contact

    # ------------------------------------------------------- intermediate
    def _compute_intermediate_values(self, dt):
        super()._compute_intermediate_values(dt)
        self._update_contact_wrench()

    def _update_contact_wrench(self):
        """Build the 6D wrench used in obs / signals (see hard-constraint #2)."""
        use_commanded = (
            self.cfg.wrench_source == "commanded"
            or getattr(self, "_held_contact", None) is None
        )
        if use_commanded:
            # Factory's commanded task wrench (PD output); 6D, set in generate_ctrl_signals.
            w = getattr(self, "applied_wrench", None)
            if w is None:
                self.contact_wrench = torch.zeros((self.num_envs, 6), device=self.device)
            else:
                self.contact_wrench = w[:, :6].clone()
            return

        # Measured contact force from the ContactSensor on the peg.
        net = self._held_contact.data.net_forces_w  # (N, num_bodies, 3) or None
        if net is None:
            force = torch.zeros((self.num_envs, 3), device=self.device)
        else:
            force = net.sum(dim=1)  # sum across the peg's bodies -> (N, 3)
        # Estimate the moment about the fingertip: tau = r x F, r = peg - ee.
        r = self.held_pos - self.fingertip_midpoint_pos
        torque = torch.cross(r, force, dim=-1)
        self.contact_wrench = torch.cat([force, torque], dim=-1)

    # ------------------------------------------------------ observations
    def _get_factory_obs_state_dict(self):
        obs_dict, state_dict = super()._get_factory_obs_state_dict()
        gripper_width = self.joint_pos[:, self._finger_dof_idx].sum(dim=-1, keepdim=True)
        extra = {
            "wrench": self.contact_wrench,
            "fingertip_pos": self.fingertip_midpoint_pos,
            "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
            "held_quat": self.held_quat,
            "gripper_width": gripper_width,
        }
        obs_dict.update(extra)
        state_dict.update({"wrench": self.contact_wrench, "gripper_width": gripper_width})
        return obs_dict, state_dict

    # --------------------------------------------------- contact signals
    def contact_signals(self) -> dict:
        """Per-env (force_mag, depth, xy_err, ori_err, engage_dist), each (num_envs,).

        Reuses Factory's peg/hole pose helpers and the wrench computed above.
        Consumed by the sigma/tau schedule (force/depth/xy/ori) and the
        engage-shaping reward (engage_dist).
        """
        held_base_pos, _ = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg, self.num_envs, self.device,
        )
        target_base_pos, _ = factory_utils.get_target_held_base_pose(
            self.fixed_pos, self.fixed_quat, self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg, self.num_envs, self.device,
        )

        force_mag = torch.linalg.vector_norm(self.contact_wrench[:, 0:3], dim=-1)
        xy_err = torch.linalg.vector_norm(target_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=-1)
        # Insertion depth: how far the peg tip has descended below the hole opening.
        depth = torch.clamp(self.fixed_pos_obs_frame[:, 2] - held_base_pos[:, 2], min=0.0)
        # Engage distance: full 3D distance from the peg base to the seated target
        # pose. Large when hovering above the hole, ~0 when fully inserted -> a
        # monotone "down + in" potential for the engage-shaping reward.
        engage_dist = torch.linalg.vector_norm(target_base_pos - held_base_pos, dim=-1)
        # Orientation error: angle between peg and hole frames.
        q_rel = torch_utils.quat_mul(self.held_quat, torch_utils.quat_conjugate(self.fixed_quat))
        ori_err = 2.0 * torch.acos(torch.clamp(q_rel[:, 0].abs(), max=1.0))
        return {"force_mag": force_mag, "depth": depth, "xy_err": xy_err,
                "ori_err": ori_err, "engage_dist": engage_dist}

    # --------------------------------------------------------- rewards
    def _engage_success(self, engage_dist=None):
        """Success = peg base within success_radius of the seated target pose.
        Looser than Factory's xy<2.5mm criterion; used for the bonus + termination."""
        if engage_dist is None:
            engage_dist = self.contact_signals()["engage_dist"]
        return engage_dist < self.cfg.reward.success_radius

    def _get_rewards(self):
        """Reward (overrides Factory's keypoint reward). Branches:

        engage-shaping (DEFAULT):
          r = -w_reach*xy_dist - w_engage*engage_dist + r_success*[success] - force_pen
            engage_dist (3D peg-base -> seated target) is a continuous down+in
            potential, so the gradient never vanishes at the "hover" plateau.
        legacy staged:  -w_reach*xy - w_align*ang*[band] + w_insert*depth + r_success*[succ] - force_pen
        reach_only:     -w_reach*xy - w_align*ang*[band] - force_pen
        sparse:         r_success*[succ] - force_pen
        """
        rc = self.cfg.reward
        sig = self.contact_signals()
        xy_dist, depth, ori_err = sig["xy_err"], sig["depth"], sig["ori_err"]
        force_mag, engage_dist = sig["force_mag"], sig["engage_dist"]

        # Success bonus fires on the RISING EDGE only (computed in _get_dones,
        # which runs first this step) so it does not accumulate while the peg stays
        # seated and re-inflate the value with a per-step +r_success.
        succ_edge = self._success_edge.float()
        successes = self._engage_succ  # currently-seated mask (for logging)
        force_pen = rc.w_force * torch.clamp(force_mag - rc.force_safe, min=0.0)
        align_band = (xy_dist < rc.align_dist).float()

        if rc.sparse:
            rew = rc.r_success * succ_edge - force_pen
        elif rc.reach_only:
            rew = -rc.w_reach * xy_dist - rc.w_align * ori_err * align_band - force_pen
        elif rc.use_engage_shaping:
            rew = -rc.w_reach * xy_dist - rc.w_engage * engage_dist + rc.r_success * succ_edge - force_pen
        else:
            rew = (
                -rc.w_reach * xy_dist
                - rc.w_align * ori_err * align_band
                + rc.w_insert * depth
                + rc.r_success * succ_edge
                - force_pen
            )

        # Bookkeeping Factory normally does inside _get_rewards.
        self.prev_actions = self.actions.clone()
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(successes) / self.num_envs
        self.extras["logs_rew_engage"] = engage_dist.mean()
        self.extras["logs_rew_force_pen"] = force_pen.mean()
        return rew

    # ----------------------------------------------------------- dones
    def _get_dones(self):
        # super() runs _compute_intermediate_values (refreshing contact_wrench) and
        # returns (time_out, time_out). We KEEP resets fully synchronous (timeout
        # only) — Factory's reset assumes all envs reset together, so we must NOT
        # terminate per-env here. Success/abort are surfaced as masks for the buffer
        # bootstrap-done flag (see train.py), not as env terminations.
        time_out, time_out2 = super()._get_dones()

        force_mag = torch.linalg.vector_norm(self.contact_wrench[:, 0:3], dim=-1)
        self._abort_mask = force_mag > self.cfg.safety.force_abort_thresh
        # currently-seated mask + per-episode rising edge (reuses _was_success latch)
        self._engage_succ = self._engage_success()
        self._success_edge = self._engage_succ & (~self._was_success)
        self._was_success = self._was_success | self._engage_succ
        # synchronous reset on timeout only
        return time_out, time_out2
