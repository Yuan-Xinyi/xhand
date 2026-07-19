# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""xArm7 + XHand pick-a-tool (hammer) with CrossDex action tokenization (Direct workflow).

Subclass of :class:`PickCubeTokenEnv`: inherits the CrossDex token action pipeline and the
scene/observations, and uses a CUBE-ALIGNED grasp+lift reward.

The earlier reward plateaued at ~308 with success/contact/lift all 0: its CONTINUOUS reach term
was farmable by hovering ~3.6 cm from the handle, while the lift was gated on a contact grasp the
policy never discovered -> no lift gradient. The reward is now rebuilt term-for-term from the
working ``pick_cube``:

  * approach: a bounded PROGRESS RATCHET (only a new-closest fingertip->keypoint distance pays), so
    hovering earns nothing and the only way to keep earning is to LIFT. Fingertips are driven to a
    set of HANDLE keypoints (``cfg.grasp_keypoints``; draggable yellow markers for GUI calibration).
  * lift: rewarded DIRECTLY (dense ratchet + one-shot bonus), UNGATED by contact, so the lift
    gradient is immediate -- gated only on table clearance (P0-4) as a cheap anti-tip.
  * KEPT (our proven success experience): the palm-facing directional gate on the approach reward,
    so the policy is never guided into a back-of-hand / dorsal press pose.
  * one-shot contact nudge steers toward a real (thumb + >=1 other) grasp rather than a scoop.

Success/termination stays a REAL grasp (contact + clearance + lifted to success height + object
nearly still, held ``success_hold_steps``); reset uses a bounding-sphere clearance vs all hand
points + a safe fallback pose.
"""
from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_conjugate, sample_uniform

from ..pick_cube_token.pick_cube_token_env import PickCubeTokenEnv
from .pick_tool_token_env_cfg import PickToolTokenEnvCfg


class PickToolTokenEnv(PickCubeTokenEnv):
    cfg: PickToolTokenEnvCfg

    def __init__(self, cfg: PickToolTokenEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev, N = self.device, self.num_envs

        # handle grasp keypoints in the object's local frame -- (K, 3)
        self.grasp_keypoints_local = torch.tensor(cfg.grasp_keypoints, dtype=torch.float, device=dev)
        self._n_kp = self.grasp_keypoints_local.shape[0]

        # ---- object AABB corners (local) + at-rest lowest-corner z, for the table-clearance test ----
        lo, hi = cfg.object_aabb_min, cfg.object_aabb_max
        corners = torch.tensor(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
            dtype=torch.float, device=dev,
        )  # (8, 3)
        self._aabb_corners_local = corners
        rest_quat = torch.tensor(cfg.object_cfg.init_state.rot, dtype=torch.float, device=dev)
        rest_z = float(cfg.object_cfg.init_state.pos[2])
        # yaw about world Z (applied at reset) leaves each corner's z unchanged, so the at-rest low
        # point is a single yaw-invariant constant.
        corner_z_rest = quat_apply(rest_quat.expand(8, 4), corners)[:, 2] + rest_z
        self._object_bottom_ref = corner_z_rest.min()  # scalar; also = the table top plane (env-local z)

        # arm link body indices for the anti-hack "arm pressed the table" geometric termination
        arm_ids, _ = self.robot.find_bodies(list(cfg.arm_contact_bodies))
        self._arm_body_ids = torch.tensor(arm_ids, dtype=torch.long, device=dev)

        # ---- reward buffers (reset in _reset_idx) ----
        # contact-GATED lift with a per-grasp baseline (the ~299 version): lift is measured RELATIVE
        # to the height at which the contact grasp was formed, so knocking the hammer up earns nothing.
        self._grasp_baseline_lift = torch.full((N,), 1.0e6, device=dev)  # root lift at grasp (logging only now)
        self._grasp_baseline_clearance = torch.full((N,), 1.0e6, device=dev)  # object off-table clearance at grasp (P0-1)
        self._highest_rel_lift = torch.zeros(N, device=dev)              # (unused; old mvp20 ratchet)
        self._success_steps = torch.zeros(N, dtype=torch.long, device=dev)  # success hold counter (P0-3)
        # --- hysteresis grasp state machine: is_grasped is a STABLE latch, not raw flickering contact ---
        self._contact_steps = torch.zeros(N, dtype=torch.long, device=dev)       # consecutive valid-contact steps
        self._lost_contact_steps = torch.zeros(N, dtype=torch.long, device=dev)  # consecutive lost-contact steps
        self._is_grasped = torch.zeros(N, dtype=torch.bool, device=dev)          # confirmed stable grasp state
        self._grasp_bonus_given = torch.zeros(N, dtype=torch.bool, device=dev)   # mvp20: "ever contacted" latch (baseline lock)
        self._grasp_age = torch.zeros(N, dtype=torch.long, device=dev)           # steps since is_grasped became True (unused mvp20)
        self._success_paid = torch.zeros(N, dtype=torch.bool, device=dev)        # one-shot stable-success bonus latch (unused mvp20)
        self._lift_bonus_given = torch.zeros(N, dtype=torch.bool, device=dev)    # mvp20: one-shot lift-off bonus latch (per episode)
        # task-space arm control (mvp27): palm EEF Jacobian row index (fixed base -> body_idx - 1) and a
        # cached 6x6 identity for the damped-least-squares pseudo-inverse.
        self._palm_jac_idx = self.palm_idx - 1
        self._eye6 = torch.eye(6, device=dev)

        # contact-sensor body indices for the R_contact grasp condition (thumb + >=1 other)
        self._contact_thumb_idx = self._contact_sensor.find_bodies("thumb_rota_link2")[0][0]
        self._contact_other_ids = [
            self._contact_sensor.find_bodies(link)[0][0]
            for link in self.cfg.ee_body_names
            if link != "thumb_rota_link2"
        ]
        # map each contact-sensor body -> its index in ee_names, so we can gate each fingertip's NET
        # contact force by THAT fingertip's distance to the object (per-finger proximity = the multi-env
        # -safe stand-in for the dead filtered fingertip<->object force_matrix_w).
        self._contact2ee = torch.tensor(
            [self.ee_names.index(n) for n in self._contact_sensor.body_names], dtype=torch.long, device=dev
        )

        # ---- DIRECTIONAL keypoints (calibration): a unit direction per object keypoint (object-local)
        #      and per finger pad (link-local). Loaded always (headless world-dir compute); the beads
        #      are GUI-only. Reward is unchanged this step -- these are just computed + visualized. ----
        if getattr(cfg, "grasp_keypoint_dirs", None) is not None:
            kp_dirs = torch.tensor(cfg.grasp_keypoint_dirs, dtype=torch.float, device=dev)
        else:
            # auto-seed: radial-outward from the keypoint centroid so the first render is non-degenerate
            centroid = self.grasp_keypoints_local.mean(dim=0, keepdim=True)
            kp_dirs = self.grasp_keypoints_local - centroid
            kp_dirs = torch.where(
                kp_dirs.norm(dim=-1, keepdim=True) > 1e-6, kp_dirs, self.grasp_keypoints_local
            )
        self.grasp_keypoint_dirs_local = self._normalize(kp_dirs)  # (K, 3)
        self.finger_pad_normal_local = self._normalize(
            torch.tensor([cfg.finger_pad_normals[n] for n in self.ee_names], dtype=torch.float, device=dev)
        )  # (5, 3), ordered like self.ee_names
        self._dir_len = float(cfg.dir_viz_len)
        self._dir_beads = int(cfg.dir_viz_beads)
        # thumb pad normal is finalized to point OPPOSITE the palm normal (user reference). It must be
        # computed at the SETTLED home pose -- at __init__ the hand has not reached the home pose yet,
        # so a value computed there is ~42 deg off once the hand settles. Defer via a countdown so the
        # back-projection happens a few stepped frames in, then reposition the thumb markers.
        self._thumb_init_pending = bool(getattr(cfg, "thumb_normal_opposite_palm", True))
        self._thumb_init_countdown = 35  # the hand takes ~30 steps to settle before -palm is stable

        # interactive keypoint calibration markers (GUI only)
        self._grasp_marker_paths = []
        self._last_printed_keypoints = None
        self._enable_grasp_calib = bool(getattr(cfg, "debug_grasp_marker", True) and self.sim.has_gui())
        # direction-marker prim paths (GUI only): tip beads + trail beads for object + finger dirs
        self._obj_dir_tip_paths = []       # (K,) magenta tip beads (draggable) under the Object
        self._obj_dir_bead_paths = []      # (K, n_beads) trail beads under the Object
        self._finger_dir_tip_paths = {}    # ee_name -> cyan tip bead (draggable) under the link
        self._finger_dir_bead_paths = {}   # ee_name -> [trail beads] under the link
        self._last_printed_dirs = None

        # compute once (fills valid poses + finalizes the thumb normal from the rest pose) BEFORE
        # creating markers, so the bead trails render the finalized directions.
        self._compute_intermediate_values()

        if self._enable_grasp_calib:
            self._create_object_grasp_markers()
            self._create_object_dir_markers()
            self._create_finger_dir_markers()

    @staticmethod
    def _normalize(v: torch.Tensor) -> torch.Tensor:
        return v / (v.norm(dim=-1, keepdim=True) + 1e-9)

    # ------------------------------------------------------------------ scene (+ contact sensors)
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)

        # one contact sensor over the 5 fingertip links, filtered for the Object -> per-finger
        # fingertip<->object contact force (paper R_contact uses real contact, not distance proxy).
        tips = "|".join(self.cfg.ee_body_names)
        # NOTE: no filter_prim_paths_expr. The filtered force_matrix_w is broken at multi-env in IsaacLab
        # (PhysX 'filter did not match' -> reads 0 for num_envs>1). We read the unfiltered net_forces_w and
        # gate it on palm-object proximity instead (see _finger_contact_state).
        self._contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/({tips})",
            )
        )
        self.scene.sensors["fingertip_contact"] = self._contact_sensor

        table_spawn = sim_utils.UsdFileCfg(usd_path=self.cfg.table_usd)
        table_spawn.func(
            "/World/envs/env_.*/Table", table_spawn, translation=self.cfg.table_pos, orientation=self.cfg.table_rot
        )
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _finger_contact_state(self):
        """(thumb_contact, other_contact_count) from real fingertip contact forces.

        Uses NET contact force (net_forces_w), NOT the filtered force_matrix_w: the latter is broken at
        multi-env scale in IsaacLab (PhysX 'filter did not match' -> reads 0 for num_envs>1, worked only
        at num_envs=1). net_forces_w is unfiltered (any contact) but works at any scale; we gate it on the
        HAND being at the object (palm within contact_near_margin of the object) so a fingertip net force
        is object contact, not table/self contact.
        """
        net = self._contact_sensor.data.net_forces_w                 # (N, B, 3) net contact force per body
        mag = net.norm(dim=-1)                                        # (N, B)
        thr = self.cfg.contact_force_thr
        # PER-FINGER proximity gate: a fingertip's net force counts as OBJECT contact only if THAT fingertip
        # is within contact_near_margin of the object (its nearest handle keypoint). This tracks each finger
        # leaving the object -- so a crush-LAUNCH (object squirts out of the grip) drops contact the instant
        # the fingers separate, instead of the old loose palm-proximity gate that stayed True through it.
        finger_dist = self._curr_fingertip_distances[:, self._contact2ee]  # (N, B) in contact-body order
        contact = (mag > thr) & (finger_dist < self.cfg.contact_near_margin)  # (N, B)
        thumb_contact = contact[:, self._contact_thumb_idx]
        other_count = contact[:, self._contact_other_ids].sum(dim=1)
        return thumb_contact, other_count

    # ------------------------------------------------------------------ keypoint markers (GUI)
    def _create_object_grasp_markers(self):
        stage = sim_utils.get_current_stage()
        obj_path = "/World/envs/env_0/Object"
        if not stage.GetPrimAtPath(obj_path).IsValid():
            self._enable_grasp_calib = False
            return
        for k, kp in enumerate(self.cfg.grasp_keypoints):
            marker_path = f"{obj_path}/dbg_kp_{k}"
            if not stage.GetPrimAtPath(marker_path).IsValid():
                mcfg = sim_utils.SphereCfg(
                    radius=0.010,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0)),
                )
                mcfg.func(marker_path, mcfg, translation=tuple(float(v) for v in kp))
            self._grasp_marker_paths.append(marker_path)

    def _sync_keypoints_from_markers(self):
        if not getattr(self, "_enable_grasp_calib", False) or not self._grasp_marker_paths:
            return
        stage = sim_utils.get_current_stage()
        new = []
        for path in self._grasp_marker_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                return
            val = prim.GetAttribute("xformOp:translate").Get()
            if val is None:
                return
            new.append([float(val[0]), float(val[1]), float(val[2])])
        off = torch.tensor(new, dtype=torch.float, device=self.device)
        if torch.max(torch.abs(off - self.grasp_keypoints_local)) <= 1e-6:
            return
        self.grasp_keypoints_local = off
        if self._last_printed_keypoints is None or torch.max(torch.abs(off - self._last_printed_keypoints)) > 1e-3:
            self._last_printed_keypoints = off.clone()
            pts = ", ".join(f"({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})" for p in off.tolist())
            print(f"[grasp-calib] grasp_keypoints = ({pts},)")

    # ------------------------------------------------------------------ direction markers (GUI)
    @staticmethod
    def _spawn_marker_sphere(path, color, radius, translation):
        stage = sim_utils.get_current_stage()
        if stage.GetPrimAtPath(path).IsValid():
            return
        mcfg = sim_utils.SphereCfg(
            radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)
        )
        mcfg.func(path, mcfg, translation=tuple(float(v) for v in translation))

    def _make_dir_trail(self, parent_path, tag, base, direction, tip_color):
        """Palm-normal-style bead trail from ``base`` along ``direction`` (unit) with a draggable TIP
        bead at the end. Returns (tip_path, [bead_paths]). ``base``/``direction`` are in the parent
        prim's local frame."""
        tip = base + direction * self._dir_len
        bead_paths = []
        for j in range(self._dir_beads):
            t = float(j + 1) / float(self._dir_beads + 1)
            pos = base + direction * (self._dir_len * t)
            bp = f"{parent_path}/{tag}_bead_{j}"
            self._spawn_marker_sphere(bp, (0.6, 0.6, 0.6), 0.004, pos)
            bead_paths.append(bp)
        tip_path = f"{parent_path}/{tag}_tip"
        self._spawn_marker_sphere(tip_path, tip_color, 0.008, tip)  # bigger + colored -> grab this
        return tip_path, bead_paths

    def _create_object_dir_markers(self):
        obj_path = "/World/envs/env_0/Object"
        stage = sim_utils.get_current_stage()
        if not stage.GetPrimAtPath(obj_path).IsValid():
            return
        for k in range(self._n_kp):
            base = self.grasp_keypoints_local[k]
            d = self.grasp_keypoint_dirs_local[k]
            tip_path, bead_paths = self._make_dir_trail(obj_path, f"dbg_kpdir_{k}", base, d, (1.0, 0.0, 1.0))
            self._obj_dir_tip_paths.append(tip_path)
            self._obj_dir_bead_paths.append(bead_paths)

    def _create_finger_dir_markers(self):
        stage = sim_utils.get_current_stage()
        for i, name in enumerate(self.ee_names):
            link_path = f"/World/envs/env_0/Robot/{name}"
            if not stage.GetPrimAtPath(link_path).IsValid():
                continue
            base = self.finger_pad_offset[0, i]                 # link-local pad center
            d = self.finger_pad_normal_local[i]
            tip_path, bead_paths = self._make_dir_trail(link_path, "dbg_padnrm", base, d, (0.0, 1.0, 1.0))
            self._finger_dir_tip_paths[name] = tip_path
            self._finger_dir_bead_paths[name] = bead_paths

    def _read_local(self, path):
        stage = sim_utils.get_current_stage()
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return None
        v = prim.GetAttribute("xformOp:translate").Get()
        return None if v is None else torch.tensor([float(v[0]), float(v[1]), float(v[2])], device=self.device)

    def _set_local(self, path, pos):
        stage = sim_utils.get_current_stage()
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            attr = prim.GetAttribute("xformOp:translate")
            if attr:
                attr.Set(tuple(float(v) for v in pos.tolist()))

    def _sync_dirs_from_markers(self):
        """Read the dragged TIP beads, recompute the unit directions (tip - base), reposition each
        trail so it follows the tip, and print the calibrated directions (ready to paste into cfg)."""
        if not getattr(self, "_enable_grasp_calib", False):
            return
        changed = False

        # ---- object keypoint directions: base = the (draggable) yellow keypoint bead ----
        obj_dirs = []
        for k in range(self._n_kp):
            base = self.grasp_keypoints_local[k]
            tip = self._read_local(self._obj_dir_tip_paths[k]) if k < len(self._obj_dir_tip_paths) else None
            if tip is None:
                obj_dirs.append(self.grasp_keypoint_dirs_local[k])
                continue
            d = tip - base
            d = d / (d.norm() + 1e-9)
            obj_dirs.append(d)
            for j, bp in enumerate(self._obj_dir_bead_paths[k]):
                t = float(j + 1) / float(self._dir_beads + 1)
                self._set_local(bp, base + d * (self._dir_len * t))
        obj_dirs = torch.stack(obj_dirs, dim=0)
        if torch.max(torch.abs(obj_dirs - self.grasp_keypoint_dirs_local)) > 1e-6:
            self.grasp_keypoint_dirs_local = obj_dirs
            changed = True

        # ---- finger pad normals: base = the (draggable) green pad bead, read in link-local frame ----
        fin_dirs = []
        for i, name in enumerate(self.ee_names):
            base = self.finger_pad_offset[0, i]
            base_marker = self._read_local(self._debug_pad_prim_paths.get(name, ""))
            if base_marker is not None:
                base = base_marker
            tip = self._read_local(self._finger_dir_tip_paths.get(name, ""))
            if tip is None:
                fin_dirs.append(self.finger_pad_normal_local[i])
                continue
            d = tip - base
            d = d / (d.norm() + 1e-9)
            fin_dirs.append(d)
            for j, bp in enumerate(self._finger_dir_bead_paths.get(name, [])):
                t = float(j + 1) / float(self._dir_beads + 1)
                self._set_local(bp, base + d * (self._dir_len * t))
        fin_dirs = torch.stack(fin_dirs, dim=0)
        if torch.max(torch.abs(fin_dirs - self.finger_pad_normal_local)) > 1e-6:
            self.finger_pad_normal_local = fin_dirs
            changed = True

        if changed and (
            self._last_printed_dirs is None
            or torch.max(torch.abs(obj_dirs - self._last_printed_dirs[0])) > 1e-3
            or torch.max(torch.abs(fin_dirs - self._last_printed_dirs[1])) > 1e-3
        ):
            self._last_printed_dirs = (obj_dirs.clone(), fin_dirs.clone())
            od = ", ".join(f"({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})" for p in obj_dirs.tolist())
            print(f"[dir-calib] grasp_keypoint_dirs = ({od},)")
            print("[dir-calib] finger_pad_normals = {")
            for i, name in enumerate(self.ee_names):
                x, y, z = fin_dirs[i].tolist()
                print(f'        "{name}": ({x:.4f}, {y:.4f}, {z:.4f}),')
            print("    }")

    # ------------------------------------------------------------------ intermediate values
    def _compute_intermediate_values(self):
        ft_sentinel = self._closest_fingertip_dist < 0.0
        super()._compute_intermediate_values()

        if getattr(self, "_enable_grasp_calib", False):
            self._sync_keypoints_from_markers()
            self._sync_dirs_from_markers()

        # handle keypoints in world -> per-finger distance to the NEAREST keypoint
        kp_w = torch.stack(
            [
                self.object_pos_w + quat_apply(self.object_quat_w, self.grasp_keypoints_local[k].expand(self.num_envs, 3))
                for k in range(self._n_kp)
            ],
            dim=1,
        )
        self.grasp_keypoints_w = kp_w
        dists = torch.norm(self.finger_pad_w.unsqueeze(2) - kp_w.unsqueeze(1), dim=-1)  # (N,5,K)
        dmin = dists.min(dim=2)
        self._curr_fingertip_distances = dmin.values      # (N,5)
        self._nearest_kp_idx = dmin.indices               # (N,5) nearest keypoint per finger
        self._closest_fingertip_dist = torch.where(
            ft_sentinel, self._curr_fingertip_distances, self._closest_fingertip_dist
        )

        # ---- DIRECTIONS in world (for the later opposing-grasp reward; visualized this step) ----
        # object keypoint outward normals: rotate object-local dirs by the object orientation
        self.grasp_keypoints_dir_w = torch.stack(
            [
                quat_apply(self.object_quat_w, self.grasp_keypoint_dirs_local[k].expand(self.num_envs, 3))
                for k in range(self._n_kp)
            ],
            dim=1,
        )  # (N, K, 3)
        # finger pad normals: rotate link-local pad normals by each fingertip link's orientation
        ft_quat = self.robot.data.body_quat_w[:, self.ee_ids]  # (N, 5, 4)
        # finalize the thumb pad normal so that, at the SETTLED home pose, its WORLD direction is
        # exactly opposite the palm normal (user reference). Deferred by a countdown because the hand
        # is not at the home pose yet at __init__ / the first few frames. Back-project -palm_normal
        # into the thumb link's local frame -> a fixed local vector, and move the thumb markers to it.
        if getattr(self, "_thumb_init_pending", False):
            if self._thumb_init_countdown > 0:
                self._thumb_init_countdown -= 1
            else:
                ti = self._thumb_ee_idx
                desired_w = (-self.palm_normal_w[0]).unsqueeze(0)          # (1, 3) world
                tq = ft_quat[0, ti].unsqueeze(0)                          # (1, 4) thumb link quat
                local = quat_apply(quat_conjugate(tq), desired_w)[0]      # (3,) thumb-local frame
                self.finger_pad_normal_local[ti] = local / (local.norm() + 1e-9)
                # reposition the thumb tip + trail beads (GUI) so they show the corrected direction
                name = self.ee_names[ti]
                if name in self._finger_dir_tip_paths:
                    base = self.finger_pad_offset[0, ti]
                    d = self.finger_pad_normal_local[ti]
                    self._set_local(self._finger_dir_tip_paths[name], base + d * self._dir_len)
                    for j, bp in enumerate(self._finger_dir_bead_paths.get(name, [])):
                        t = float(j + 1) / float(self._dir_beads + 1)
                        self._set_local(bp, base + d * (self._dir_len * t))
                self._thumb_init_pending = False
                v = self.finger_pad_normal_local[ti].tolist()
                print(f"[dir-calib] thumb pad normal (local, opposite palm @home) = ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})")
        self.finger_pad_normal_w = torch.stack(
            [
                quat_apply(ft_quat[:, i], self.finger_pad_normal_local[i].expand(self.num_envs, 3))
                for i in range(len(self.ee_ids))
            ],
            dim=1,
        )  # (N, 5, 3)

        # ---- opposing-normal ALIGNMENT (replaces the palm-facing gate) ----
        # each finger pad should press INTO the handle surface, i.e. its normal points OPPOSITE the
        # nearest object keypoint's OUTWARD normal. align in [0,1]: 1 = perfectly opposed, 0 = same dir.
        nearest_kp_dir = torch.gather(
            self.grasp_keypoints_dir_w, 1, self._nearest_kp_idx.unsqueeze(-1).expand(-1, -1, 3)
        )  # (N, 5, 3) outward normal of each finger's nearest keypoint
        dot = (self.finger_pad_normal_w * nearest_kp_dir).sum(dim=-1)  # (N,5) in [-1,1]
        self._finger_align = (1.0 - dot) * 0.5  # (N,5) in [0,1]

    # ------------------------------------------------------------------ clearance
    def _object_min_corner_z(self) -> torch.Tensor:
        """Env-local z of the object's lowest AABB corner (for the table-clearance test, P0-4)."""
        N = self.num_envs
        cz = torch.stack(
            [
                (self.object_pos_w + quat_apply(self.object_quat_w, self._aabb_corners_local[k].expand(N, 3)))[:, 2]
                for k in range(8)
            ],
            dim=1,
        )
        return (cz - self.scene.env_origins[:, 2:3]).min(dim=1).values

    # ------------------------------------------------------------------ reward
    def _get_observations(self) -> dict:
        # append 3 PHASE features so the policy KNOWS which stage it is in (the reward switches hard at
        # is_grasped: before -> approach/close; after -> hold fingers + move up). Inferring the phase
        # from noisy contact force alone is much harder. Added to BOTH policy obs and the symmetric
        # "critic" group (the latter also feeds SAPG's states injection; plain PPO just uses the obs).
        d = super()._get_observations()
        # clearance-based lift progress (matches the reward): how far the WHOLE object is off the table,
        # relative to its off-table clearance at grasp formation.
        clearance = self._object_min_corner_z() - self._object_bottom_ref
        grasp_rel_lift = torch.clamp(clearance - self._grasp_baseline_clearance, min=0.0)
        extra = torch.stack(
            [
                self._is_grasped.float(),                                                   # phase flag 0/1
                torch.clamp(self._grasp_age.float() / float(self.max_episode_length), 0.0, 1.0),  # normalized grasp age
                torch.clamp(grasp_rel_lift / self.cfg.lift_success_height, 0.0, 1.0),       # normalized lift progress
            ],
            dim=-1,
        )
        obs = torch.cat([d["policy"], extra], dim=-1)
        self._policy_obs_cache = obs
        return {"policy": obs, "critic": obs}

    def _get_states(self) -> torch.Tensor:
        return getattr(self, "_policy_obs_cache", None)

    # NOTE: joint-space arm control (mvp20) -- no _pre_physics_step override; inherits
    # PickCubeTokenEnv._pre_physics_step (7 arm relative joint deltas + 9 hand eigengrasp token = 16).

    def _get_rewards(self) -> torch.Tensor:
        # FOUR-TERM reward with a HYSTERESIS grasp state machine (mvp26, joint-space):
        #   R_reach = reach_reward_scale * reach_val * (~is_grasped)  -- directional pre-grasp guidance
        #             (mvp20 kernel: palm_facing x align x coarse+fine distance); switches OFF once grasped.
        #   R_grasp = grasp_bonus * new_grasp_event                   -- ONE-SHOT on the first stable grasp
        #             (latched by grasp_bonus_given for the WHOLE episode -> grasp/drop/regrasp can't re-farm it).
        #   R_lift  = lift_step_max * clip(grasp_rel_lift/lift_success_height,0,1) * is_grasped -- per-step
        #             NORMALIZED occupancy HEIGHT (NOT a ratchet -> a crush-launch pays only while airborne;
        #             holding high pays continuously; no bounce-poisoned cliff).
        #   R_bonus = lift_success_bonus * newly_successful           -- one-shot on the STRICT stable success.
        # is_grasped is a STABLE latch (needs grasp_confirm_steps of valid contact to turn ON, grasp_release_steps
        # of loss to turn OFF) so R_reach switches cleanly and the reward can't chatter with flickering contact.
        cfg = self.cfg
        object_z = self.object_pos_w[:, 2] - self.scene.env_origins[:, 2]
        actual_lift = object_z - self.object_default_z

        # ---- hysteresis grasp state machine (valid_contact = thumb pad + >=1 other pad on the object) ----
        thumb_contact, other_count = self._finger_contact_state()
        valid_contact = thumb_contact & (other_count >= 1)
        self._contact_steps = torch.where(
            valid_contact, self._contact_steps + 1, torch.zeros_like(self._contact_steps)
        )
        self._lost_contact_steps = torch.where(
            valid_contact, torch.zeros_like(self._lost_contact_steps), self._lost_contact_steps + 1
        )
        confirm = self._contact_steps >= cfg.grasp_confirm_steps       # enough consecutive contact -> grasp ON
        release = self._lost_contact_steps >= cfg.grasp_release_steps  # enough consecutive loss    -> grasp OFF
        newly_grasped = (~self._is_grasped) & confirm
        self._is_grasped = (self._is_grasped | confirm) & ~release
        is_grasped = self._is_grasped
        self._grasp_age = torch.where(is_grasped, self._grasp_age + 1, torch.zeros_like(self._grasp_age))

        # P0-4: object AABB lowest corner above the table (env-local). This is the TRUE off-table height of
        # the WHOLE object -- used both for clearance and as the lift measure (see below).
        clearance = self._object_min_corner_z() - self._object_bottom_ref
        clearance_ok = clearance > cfg.clearance_margin

        # FIRST stable grasp of the episode (happens exactly once). The lift baseline is LOCKED here and
        # NEVER reset on a re-grasp -> a drop->lower->regrasp->relift loop cannot re-farm; and pre-grasp
        # knock-ups aren't credited (P0-1). Baseline is on CLEARANCE (see below), not root height.
        first_grasp = newly_grasped & (~self._grasp_bonus_given)
        self._grasp_baseline_clearance = torch.where(first_grasp, clearance, self._grasp_baseline_clearance)

        # LIFT is measured by CLEARANCE (how far the WHOLE object is off the table), NOT the object root
        # height. The root-height measure was hackable: tipping the hammer toward VERTICAL raises the root
        # ~half its length without lifting the object off the table (root +18cm while the bottom was only
        # +8.5cm off the table). Clearance-based lift pays only for the object actually leaving the table,
        # so "tip it vertical" earns only its true off-table height.
        grasp_rel_lift = torch.clamp(clearance - self._grasp_baseline_clearance, min=0.0)

        # ---- R_grasp: ONE-SHOT bonus for the FIRST stable grasp this episode (same event) ----
        r_grasp = cfg.grasp_bonus * first_grasp.float()
        self._grasp_bonus_given = self._grasp_bonus_given | is_grasped  # never re-paid, even after drop+regrasp

        # ---- R_lift = per-step NORMALIZED off-table HEIGHT (occupancy) + one-shot stable-success bonus.
        #      Per-step height pays for the CURRENT held clearance every step (no history/ratchet) -> gradient
        #      from mm 1 ALWAYS; a crush-launch/tip pays only while the object is actually off the table;
        #      holding the whole object high pays continuously (aligned with success). ----
        lift_fraction = torch.clamp(grasp_rel_lift / cfg.lift_success_height, 0.0, 1.0)
        r_lift_height = cfg.lift_step_max * lift_fraction * is_grasped.float()

        # one-shot pull to the finish line: paid once when the STRICT stable success first latches (held
        # success_hold_steps at >=20cm, cleared, nearly still). _is_success is set in _get_dones (runs
        # just before this each step).
        newly_successful = self._is_success & (~self._success_paid)
        r_lift_success = cfg.lift_success_bonus * newly_successful.float()
        self._success_paid = self._success_paid | self._is_success

        # ---- R_reach: directional occupancy pre-grasp guidance (mvp20 kernel), OFF once grasped ----
        # coarse+fine distance kernel (far/near) x palm_facing (whole-hand orientation) x align (per-finger
        # pad normals). palm_facing is load-bearing (mvp17: align alone lets a dorsal hover farm reach).
        d = self._curr_fingertip_distances
        a = self._finger_align  # (N,5) in [0,1], 1 = pad normal opposes its nearest keypoint normal
        other_d = d[:, self._other_ee_idx]
        other_a = a[:, self._other_ee_idx]
        near_val, near_idx = torch.topk(other_d, k=2, dim=1, largest=False)  # thumb + 2 nearest others
        grasp_dist = (d[:, self._thumb_ee_idx] + near_val.sum(dim=-1)) / 3.0
        align = (a[:, self._thumb_ee_idx] + torch.gather(other_a, 1, near_idx).sum(dim=-1)) / 3.0
        reach_far = 1.0 - torch.tanh(grasp_dist / cfg.reach_scale_far)
        reach_near = 1.0 - torch.tanh(grasp_dist / cfg.reach_scale)
        to_obj = self.object_pos_w - self.palm_center_w
        to_obj = to_obj / to_obj.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        palm_facing = 0.5 * (1.0 + (self.palm_normal_w * to_obj).sum(dim=-1))
        reach_val = 0.5 * (reach_far + reach_near) * palm_facing * align
        r_reach = cfg.reach_reward_scale * reach_val * (~is_grasped).float()

        if "log" not in self.extras:
            self.extras["log"] = dict()
        log = self.extras["log"]
        log["align_mean"] = align.mean()
        log["palm_facing_mean"] = palm_facing.mean()
        log["valid_contact_frac"] = valid_contact.float().mean()
        log["is_grasped_frac"] = is_grasped.float().mean()
        log["thumb_contact_frac"] = thumb_contact.float().mean()
        log["other_contact_count_mean"] = other_count.float().mean()
        log["clearance_ok_frac"] = clearance_ok.float().mean()
        log["grasp_rel_lift_mean"] = grasp_rel_lift.mean()
        log["grasp_rel_lift_max"] = grasp_rel_lift.max()
        log["actual_lift_mean"] = actual_lift.mean()
        log["avg_ft_to_kp_mean"] = self._curr_fingertip_distances.mean()
        log["success_frac"] = self._is_success.float().mean()
        # height distribution: fraction of envs whose grasp-relative lift is currently at/above each mark
        log["lift_ge_3cm_frac"] = (grasp_rel_lift >= 0.03).float().mean()
        log["lift_ge_5cm_frac"] = (grasp_rel_lift >= 0.05).float().mean()
        log["lift_ge_10cm_frac"] = (grasp_rel_lift >= 0.10).float().mean()
        log["lift_ge_20cm_frac"] = (grasp_rel_lift >= 0.20).float().mean()
        log["r_reach_mean"] = r_reach.mean()
        log["r_grasp_mean"] = r_grasp.mean()
        log["r_lift_height_mean"] = r_lift_height.mean()
        log["r_lift_success_mean"] = r_lift_success.mean()

        return r_reach + r_grasp + r_lift_height + r_lift_success

    # ------------------------------------------------------------------ termination
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        cfg = self.cfg
        object_z = self.object_pos_w[:, 2] - self.scene.env_origins[:, 2]
        actual_lift = object_z - self.object_default_z

        # success uses the SAME clearance-based lift as the reward (whole object off the table, relative to
        # grasp-formation clearance), so a "tip to vertical" can't reach the success height.
        clearance = self._object_min_corner_z() - self._object_bottom_ref
        grasp_rel_lift = torch.clamp(clearance - self._grasp_baseline_clearance, min=0.0)
        clearance_ok = clearance > cfg.clearance_margin
        obj_lin = self.object.data.root_lin_vel_w.norm(dim=-1)
        obj_ang = self.object.data.root_ang_vel_w.norm(dim=-1)
        slow = (obj_lin < cfg.success_max_obj_lin_speed) & (obj_ang < cfg.success_max_obj_ang_speed)

        # P0-3: success = STABLE grasp (is_grasped, hysteresis) + clearance + rel-lift (relative to the
        # grasp height), object nearly still, held for N steps. is_grasped is updated in _get_rewards
        # (runs right after this each step); a 1-step lag on the success gate is immaterial.
        success_inst = self._is_grasped & clearance_ok & (grasp_rel_lift >= cfg.lift_success_height) & slow
        self._success_steps = torch.where(
            success_inst, self._success_steps + 1, torch.zeros_like(self._success_steps)
        )
        held = self._success_steps >= cfg.success_hold_steps
        self._is_success = self._is_success | held

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        dropped = self.object_pos_w[:, 2] < (self.object_default_z - cfg.drop_height)
        if not cfg.terminate_on_drop:
            dropped = torch.zeros_like(dropped)

        # ANTI-HACK: arm pressed the table? any arm link center within arm_table_margin of the table plane
        # (env-local z = _object_bottom_ref). Ends the episode so the "press-up off the table" hack pays 0.
        arm_z_local = self.robot.data.body_pos_w[:, self._arm_body_ids, 2] - self.scene.env_origins[:, 2:3]
        arm_table_hit = (arm_z_local < (self._object_bottom_ref + cfg.arm_table_margin)).any(dim=1)
        if not cfg.terminate_on_arm_table_contact:
            arm_table_hit = torch.zeros_like(dropped)
        self.extras.setdefault("log", dict())["arm_table_hit_frac"] = arm_table_hit.float().mean()

        return dropped | self._is_success | arm_table_hit, time_out

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._grasp_baseline_lift[env_ids] = 1.0e6
        self._grasp_baseline_clearance[env_ids] = 1.0e6
        self._highest_rel_lift[env_ids] = 0.0
        self._success_steps[env_ids] = 0
        self._contact_steps[env_ids] = 0
        self._lost_contact_steps[env_ids] = 0
        self._is_grasped[env_ids] = False
        self._grasp_bonus_given[env_ids] = False  # "ever contacted" latch clears ONLY on episode reset
        self._grasp_age[env_ids] = 0
        self._success_paid[env_ids] = False
        self._lift_bonus_given[env_ids] = False

    # ------------------------------------------------------------------ reset object placement (P1-8)
    def _sample_non_overlapping_object_xy(self, env_ids, default_xy):
        """Bounding-sphere clearance of the whole object vs ALL hand points, with a safe fallback to
        the un-noised default xy (the base accepted the last -- possibly overlapping -- candidate)."""
        nx, ny = self.cfg.reset_object_pos_noise
        n = len(env_ids)
        # object bounding radius (half the AABB diagonal) + the base min-distance margin
        aabb_min = torch.tensor(self.cfg.object_aabb_min, device=self.device)
        aabb_max = torch.tensor(self.cfg.object_aabb_max, device=self.device)
        obj_radius = 0.5 * torch.norm(aabb_max - aabb_min).item()
        min_dist = self.cfg.reset_min_hand_object_dist + obj_radius

        hand_points = torch.cat(
            [self.finger_pad_w[env_ids], self.palm_center_w[env_ids].unsqueeze(1)], dim=1
        )
        xy = default_xy.clone()          # safe fallback = un-noised default position
        remaining = torch.ones(n, dtype=torch.bool, device=self.device)
        for _ in range(32):
            noise = sample_uniform(-1.0, 1.0, (n, 2), device=self.device)
            candidate = default_xy.clone()
            candidate[:, 0] += nx * noise[:, 0]
            candidate[:, 1] += ny * noise[:, 1]
            candidate_w = torch.cat(
                [
                    candidate + self.scene.env_origins[env_ids, 0:2],
                    self.object.data.default_root_state[env_ids, 2:3] + self.scene.env_origins[env_ids, 2:3],
                ],
                dim=-1,
            )
            min_hand = torch.norm(hand_points - candidate_w.unsqueeze(1), dim=-1).min(dim=-1).values
            accept = remaining & (min_hand > min_dist)
            xy[accept] = candidate[accept]
            remaining = remaining & ~accept
            if not remaining.any():
                break
        # envs that never found a clear candidate keep the safe default xy (NOT an overlapping one)
        return xy
