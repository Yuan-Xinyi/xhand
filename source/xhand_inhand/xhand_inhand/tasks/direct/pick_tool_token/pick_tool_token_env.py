# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""xArm7 + XHand hammer pickup with CrossDex token actions (Direct workflow).

The environment uses object-filtered singleton fingertip sensors, a calibrated real-handle
surface, and one shared robust grasp quality for phase confirmation, hold, lift and success.
Transport quality is evaluated entirely in center-of-mass frames.  Lift height is the minimum
world-Z of the real mesh convex hull relative to the table, so tipping or flinging the hammer
cannot masquerade as a successful lift.

Exploration uses signed near/close/wrap/lift potentials and five distal residual controls.  These
terms improve credit assignment without weakening contact, transport or true-clearance invariants.
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
from .grasp_signals import rigid_hold_quality, staged_close_quality, update_grasp_latch, wrap_quality
from .hybrid_action import apply_asymmetric_joint_residual
from .pick_tool_token_env_cfg import PickToolTokenEnvCfg


class PickToolTokenEnv(PickCubeTokenEnv):
    cfg: PickToolTokenEnvCfg

    def __init__(self, cfg: PickToolTokenEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev, N = self.device, self.num_envs

        # CrossDex leaves these five distal flexion joints nearly fixed.  Resolve them by name in
        # articulation-hand order; ee_names has a different order and must never be used here.
        _, hand_joint_names = self.robot.find_joints(self.cfg.hand_joint_names)
        distal_names = tuple(cfg.distal_residual_joint_names)
        if len(set(distal_names)) != len(distal_names):
            raise ValueError("distal_residual_joint_names contains duplicates")
        missing = [name for name in distal_names if name not in hand_joint_names]
        if missing:
            raise ValueError(f"Distal residual joints are absent from the XHand articulation: {missing}")
        self._distal_hand_ids = torch.tensor(
            [hand_joint_names.index(name) for name in distal_names], dtype=torch.long, device=dev
        )
        self._n_distal_residuals = len(distal_names) if cfg.enable_distal_residual else 0
        expected_actions = self._n_arm + self._n_tokens + self._n_distal_residuals
        if cfg.action_space != expected_actions:
            raise ValueError(
                f"PickTool action_space={cfg.action_space}, expected {expected_actions} "
                f"({self._n_arm} arm + {self._n_tokens} token + {self._n_distal_residuals} residual)."
            )
        self._last_token_hand_target = torch.zeros((N, len(hand_joint_names)), device=dev)
        self._last_distal_delta = torch.zeros((N, len(distal_names)), device=dev)
        self._last_raw_hand_target = torch.zeros((N, len(hand_joint_names)), device=dev)

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
        self._object_bottom_ref = corner_z_rest.min()  # scalar; LOOSE box bottom (kept for reference only)

        # ---- TRUE object clearance via the mesh CONVEX HULL (the local AABB box is ~12cm looser than the
        # mesh, so its rotated min-corner reads FAKE clearance when the object tips -> use real hull verts).
        from .tool_asset import TOOL_OBJ, TOOL_SCALE
        from scipy.spatial import ConvexHull
        _v = []
        for _ln in open(TOOL_OBJ):
            if _ln.startswith("v "):
                _p = _ln.split()
                _v.append((float(_p[1]), float(_p[2]), float(_p[3])))
        _v = torch.tensor(_v, dtype=torch.float, device=dev) * torch.tensor(TOOL_SCALE, device=dev)
        _hull = ConvexHull(_v.cpu().numpy()).vertices
        self._obj_hull_local = _v[torch.as_tensor(_hull, device=dev)]  # (M,3) hull verts, the min-z is always one
        # true table surface = the object's real lowest mesh point at the rest pose (it sits on the table);
        # yaw-invariant (reset only yaws), so a single constant.
        _rest_world_z = quat_apply(rest_quat.expand(self._obj_hull_local.shape[0], 4), self._obj_hull_local)[:, 2]
        self._table_surface_z = (_rest_world_z + rest_z).min()  # scalar, env-local

        # ---- dense analytic handle surface ----
        # The calibrated points form a cross-section of the handle. Build a convex polygon from a
        # thin slice of the real mesh, then use point-to-segment distance and edge normals at runtime.
        # This is both more accurate and much cheaper than a nearest search over all 44k mesh vertices.
        self._handle_center_local = torch.tensor(cfg.handle_center, dtype=torch.float, device=dev)
        self._handle_axis_local = self._normalize(torch.tensor(cfg.handle_axis, dtype=torch.float, device=dev))
        reference = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float, device=dev)
        if torch.abs(torch.dot(self._handle_axis_local, reference)) > 0.9:
            reference = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float, device=dev)
        self._handle_u_local = self._normalize(torch.cross(self._handle_axis_local, reference, dim=-1))
        self._handle_v_local = self._normalize(
            torch.cross(self._handle_axis_local, self._handle_u_local, dim=-1)
        )
        mesh_rel = _v - self._handle_center_local
        mesh_axial = mesh_rel @ self._handle_axis_local
        section_mask = mesh_axial.abs() <= cfg.handle_section_half_width
        section_rel = mesh_rel[section_mask]
        if section_rel.shape[0] < 3:
            raise RuntimeError("Handle mesh slice has fewer than three vertices; check handle frame calibration.")
        section_xy = torch.stack(
            (section_rel @ self._handle_u_local, section_rel @ self._handle_v_local), dim=-1
        )
        section_hull = ConvexHull(section_xy.cpu().numpy())
        polygon = section_xy[torch.as_tensor(section_hull.vertices, dtype=torch.long, device=dev)]
        self._handle_edge_start = polygon
        self._handle_edge_vector = torch.roll(polygon, shifts=-1, dims=0) - polygon
        edge_len = self._handle_edge_vector.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
        signed_twice_area = (
            polygon[:, 0] * torch.roll(polygon[:, 1], shifts=-1)
            - polygon[:, 1] * torch.roll(polygon[:, 0], shifts=-1)
        ).sum()
        polygon_radius = polygon.norm(dim=-1)
        if (
            signed_twice_area <= 2.0e-4
            or polygon_radius.min() < 0.005
            or polygon_radius.max() > 0.05
            or edge_len.min() < 1.0e-6
        ):
            raise RuntimeError(
                "Invalid handle cross-section polygon: "
                f"area={0.5 * signed_twice_area.item():.6g}, "
                f"radius=[{polygon_radius.min().item():.6g}, {polygon_radius.max().item():.6g}], "
                f"min_edge={edge_len.min().item():.6g}."
            )
        # scipy ConvexHull vertices are counter-clockwise: (dy, -dx) is the outward normal.
        self._handle_edge_normal_2d = torch.stack(
            (self._handle_edge_vector[:, 1], -self._handle_edge_vector[:, 0]), dim=-1
        ) / edge_len

        # Runtime truth, not the config or a cached default: fail early if mass authoring was ignored.
        self._runtime_object_masses = self.object.root_physx_view.get_masses().reshape(N, -1)
        expected_mass = torch.full_like(self._runtime_object_masses, cfg.expected_object_mass)
        if not torch.allclose(
            self._runtime_object_masses, expected_mass, atol=cfg.object_mass_tolerance, rtol=0.0
        ):
            mass_min = self._runtime_object_masses.min().item()
            mass_max = self._runtime_object_masses.max().item()
            raise RuntimeError(
                f"Runtime hammer mass is [{mass_min:.6f}, {mass_max:.6f}] kg; "
                f"expected {cfg.expected_object_mass:.6f} kg. Mass authoring did not reach PhysX."
            )

        # arm link body indices for the anti-hack "arm pressed the table" geometric termination
        arm_ids, _ = self.robot.find_bodies(list(cfg.arm_contact_bodies))
        self._arm_body_ids = torch.tensor(arm_ids, dtype=torch.long, device=dev)

        # ---- reward buffers (reset in _reset_idx) ----
        # contact-GATED lift with a per-grasp baseline (the ~299 version): lift is measured RELATIVE
        # to the height at which the contact grasp was formed, so knocking the hammer up earns nothing.
        self._grasp_baseline_lift = torch.full((N,), 1.0e6, device=dev)  # root lift at grasp (logging only now)
        self._grasp_baseline_clearance = torch.full((N,), 1.0e6, device=dev)  # object off-table clearance at grasp (P0-1)
        self._highest_rel_lift = torch.zeros(N, device=dev)              # (unused; old mvp20 ratchet)
        # object POSE at grasp formation -- the lift reward is gated on keeping this pose (no twisting, no
        # horizontal drift) so only a CLEAN vertical lift pays (arm "wiggle" that tumbles/drags the object -> 0).
        self._grasp_baseline_xy = torch.zeros((N, 2), device=dev)        # object xy at grasp (world; diff cancels origin)
        self._grasp_baseline_quat = torch.zeros((N, 4), device=dev)      # object orientation at grasp
        self._grasp_baseline_quat[:, 0] = 1.0                            # identity until first grasp locks it
        self._success_steps = torch.zeros(N, dtype=torch.long, device=dev)  # success hold counter (P0-3)
        # --- hysteresis grasp state machine: is_grasped is a STABLE latch, not raw flickering contact ---
        self._contact_steps = torch.zeros(N, dtype=torch.long, device=dev)       # consecutive valid-contact steps
        self._lost_contact_steps = torch.zeros(N, dtype=torch.long, device=dev)  # consecutive lost-contact steps
        self._is_grasped = torch.zeros(N, dtype=torch.bool, device=dev)          # confirmed stable grasp state
        self._grasp_bonus_given = torch.zeros(N, dtype=torch.bool, device=dev)   # safe stable-grasp bonus latch
        self._safe_grasp_steps = torch.zeros(N, dtype=torch.long, device=dev)    # consecutive low-impact latch steps
        self._grasp_age = torch.zeros(N, dtype=torch.long, device=dev)           # steps since is_grasped became True (unused mvp20)
        self._success_paid = torch.zeros(N, dtype=torch.bool, device=dev)        # one-shot stable-success bonus latch (unused mvp20)
        self._lift_bonus_given = torch.zeros(N, dtype=torch.bool, device=dev)    # mvp20: one-shot lift-off bonus latch (per episode)
        self._prev_close_quality = torch.zeros(N, device=dev)
        self._prev_wrap_quality = torch.zeros(N, device=dev)
        self._prev_lift_potential = torch.zeros(N, device=dev)
        self._potential_initialized = torch.zeros(N, dtype=torch.bool, device=dev)
        # task-space arm control (mvp27): palm EEF Jacobian row index (fixed base -> body_idx - 1) and a
        # cached 6x6 identity for the damped-least-squares pseudo-inverse.
        self._palm_jac_idx = self.palm_idx - 1
        self._eye6 = torch.eye(6, device=dev)

        # Contact tensors are ordered exactly like ee_names. Each fingertip has its own singleton
        # object-filtered sensor; Isaac Lab does not support filtering one sensor that matches many bodies.
        self._contact_thumb_idx = self.ee_names.index("thumb_rota_link2")
        self._contact_other_ids = torch.tensor(
            [i for i, name in enumerate(self.ee_names) if name != "thumb_rota_link2"],
            dtype=torch.long,
            device=dev,
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

        # Object-specific contact requires ONE sensor body per filtered ContactSensor. The previous
        # single regex sensor matched all five fingertips, an unsupported many-to-many configuration
        # that happened to work with one env and returned zero at training scale. Keep five singleton
        # sensors in ee_names order and filter each one against the object.
        self._object_contact_sensors: dict[str, ContactSensor] = {}
        for name in self.cfg.ee_body_names:
            sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{name}",
                    filter_prim_paths_expr=["/World/envs/env_.*/Object"],
                )
            )
            self._object_contact_sensors[name] = sensor
            self.scene.sensors[f"object_contact_{name}"] = sensor

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

    def _finger_object_force_magnitudes(self) -> torch.Tensor:
        """Object-filtered normal-force magnitudes in ``ee_names`` order, shape ``(N, 5)``."""

        magnitudes = []
        for name in self.ee_names:
            matrix = self._object_contact_sensors[name].data.force_matrix_w
            if matrix is None:
                raise RuntimeError(f"Filtered contact matrix is unavailable for singleton sensor {name!r}.")
            # (N, one sensor body, one-or-more filtered collision shapes, xyz)
            magnitudes.append(matrix.norm(dim=-1).sum(dim=(1, 2)))
        return torch.stack(magnitudes, dim=1)

    def _finger_net_force_magnitudes(self) -> torch.Tensor:
        """Unfiltered fingertip net forces, used only to audit filtered-contact coverage."""

        return torch.stack(
            [
                self._object_contact_sensors[name].data.net_forces_w[:, 0].norm(dim=-1)
                for name in self.ee_names
            ],
            dim=1,
        )

    def _finger_contact_state(self):
        """Return robust ``(thumb_contact, nonthumb_count)`` on the actual handle."""

        forces = self._finger_object_force_magnitudes()
        contact = (
            (forces > self.cfg.contact_force_thr)
            & self._handle_contact_region
            & (self._handle_side_distances < self.cfg.handle_contact_margin)
        )
        return contact[:, self._contact_thumb_idx], contact[:, self._contact_other_ids].sum(dim=1)

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

        # ---- real handle surface distance + outward normal ----
        # Transform pad centers to object local coordinates, project into the calibrated handle
        # cross-section, and find the nearest segment of the polygon cut from the true mesh.
        object_q_inv = quat_conjugate(self.object_quat_w)
        pad_rel_w = self.finger_pad_w - self.object_pos_w.unsqueeze(1)
        pad_local = quat_apply(
            object_q_inv.unsqueeze(1).expand(-1, len(self.ee_ids), -1).reshape(-1, 4),
            pad_rel_w.reshape(-1, 3),
        ).reshape(self.num_envs, len(self.ee_ids), 3)
        handle_rel = pad_local - self._handle_center_local
        handle_axial = (handle_rel * self._handle_axis_local).sum(dim=-1)
        handle_xy = torch.stack(
            (
                (handle_rel * self._handle_u_local).sum(dim=-1),
                (handle_rel * self._handle_v_local).sum(dim=-1),
            ),
            dim=-1,
        )
        edge_start = self._handle_edge_start.view(1, 1, -1, 2)
        edge_vector = self._handle_edge_vector.view(1, 1, -1, 2)
        point_delta = handle_xy.unsqueeze(2) - edge_start
        edge_len_sq = (edge_vector * edge_vector).sum(dim=-1).clamp_min(1.0e-12)
        edge_t = torch.clamp((point_delta * edge_vector).sum(dim=-1) / edge_len_sq, 0.0, 1.0)
        closest_xy = edge_start + edge_t.unsqueeze(-1) * edge_vector
        side_dist_sq = ((handle_xy.unsqueeze(2) - closest_xy) ** 2).sum(dim=-1)
        side_dist_sq, nearest_edge = side_dist_sq.min(dim=2)
        self._handle_side_distances = torch.sqrt(side_dist_sq)
        self._handle_contact_region = (
            (handle_axial >= self.cfg.handle_axial_min - self.cfg.handle_axial_margin)
            & (handle_axial <= self.cfg.handle_axial_max + self.cfg.handle_axial_margin)
        )
        axial_excess = torch.clamp(self.cfg.handle_axial_min - handle_axial, min=0.0) + torch.clamp(
            handle_axial - self.cfg.handle_axial_max, min=0.0
        )
        self._handle_surface_distances = torch.sqrt(side_dist_sq + axial_excess.square())
        self._handle_axial = handle_axial

        normal_2d = self._handle_edge_normal_2d[nearest_edge]
        handle_normal_local = (
            normal_2d[..., 0:1] * self._handle_u_local
            + normal_2d[..., 1:2] * self._handle_v_local
        )
        self._handle_surface_normal_w = quat_apply(
            self.object_quat_w.unsqueeze(1).expand(-1, len(self.ee_ids), -1).reshape(-1, 4),
            handle_normal_local.reshape(-1, 3),
        ).reshape(self.num_envs, len(self.ee_ids), 3)
        self.handle_center_w = self.object_pos_w + quat_apply(
            self.object_quat_w, self._handle_center_local.expand(self.num_envs, 3)
        )

        # Each pad should press into the nearest real handle surface. Unlike the old centroid-to-
        # keypoint directions, these normals have no spurious component along the handle axis.
        dot = (self.finger_pad_normal_w * self._handle_surface_normal_w).sum(dim=-1)
        self._finger_align = (1.0 - dot) * 0.5  # (N,5) in [0,1]

    # ------------------------------------------------------------------ clearance
    def _object_min_corner_z(self) -> torch.Tensor:
        """Env-local z of the object's lowest AABB corner (LOOSE box; kept for reference/logging only)."""
        N = self.num_envs
        cz = torch.stack(
            [
                (self.object_pos_w + quat_apply(self.object_quat_w, self._aabb_corners_local[k].expand(N, 3)))[:, 2]
                for k in range(8)
            ],
            dim=1,
        )
        return (cz - self.scene.env_origins[:, 2:3]).min(dim=1).values

    def _object_true_min_z(self) -> torch.Tensor:
        """Env-local z of the object's TRUE lowest point (min over mesh convex-hull verts, rotation-correct).
        Unlike the loose local-AABB box, this doesn't read fake clearance when the object tips."""
        N, M = self.num_envs, self._obj_hull_local.shape[0]
        q = self.object_quat_w.unsqueeze(1).expand(N, M, 4).reshape(-1, 4)
        v = self._obj_hull_local.unsqueeze(0).expand(N, M, 3).reshape(-1, 3)
        world = quat_apply(q, v).reshape(N, M, 3) + self.object_pos_w.unsqueeze(1)
        return (world[:, :, 2] - self.scene.env_origins[:, 2:3]).min(dim=1).values

    def _compute_grasp_signals(self) -> dict[str, torch.Tensor]:
        """Compute the single source of truth used by grasp, lift, release and success."""

        force_magnitude = self._finger_object_force_magnitudes()
        to_handle = self.handle_center_w - self.palm_center_w
        to_handle = to_handle / to_handle.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        palm_facing = 0.5 * (1.0 + (self.palm_normal_w * to_handle).sum(dim=-1))
        wrap = wrap_quality(
            force_magnitude,
            torch.where(
                self._handle_contact_region,
                self._handle_side_distances,
                torch.full_like(self._handle_side_distances, torch.inf),
            ),
            self._handle_surface_normal_w,
            self._finger_align,
            palm_facing,
            self._contact_thumb_idx,
            self._contact_other_ids,
            force_threshold=self.cfg.contact_force_thr,
            force_saturation=self.cfg.contact_force_saturation,
            surface_margin=self.cfg.handle_contact_margin,
            palm_facing_min=self.cfg.grasp_palm_facing_min,
            alignment_min=self.cfg.grasp_align_min,
            opposition_min=self.cfg.grasp_opposition_min,
        )
        wrap.update(
            staged_close_quality(
                force_magnitude,
                self._handle_surface_distances,
                self._handle_contact_region,
                self._handle_surface_normal_w,
                self._finger_align,
                wrap["palm_score"],
                self._contact_thumb_idx,
                self._contact_other_ids,
                alignment_min=self.cfg.grasp_align_min,
                opposition_min=self.cfg.grasp_opposition_min,
                proximity_scale_far=self.cfg.close_proximity_scale_far,
                proximity_scale_near=self.cfg.close_proximity_scale_near,
                force_saturation=self.cfg.contact_force_saturation,
            )
        )

        hold, slip_lin, slip_ang = rigid_hold_quality(
            self.robot.data.body_com_pos_w[:, self.palm_idx],
            self.robot.data.body_com_lin_vel_w[:, self.palm_idx],
            self.robot.data.body_com_ang_vel_w[:, self.palm_idx],
            self.object.data.root_com_pos_w,
            self.object.data.root_com_lin_vel_w,
            self.object.data.root_com_ang_vel_w,
            self.cfg.hold_v_scale,
            self.cfg.hold_w_scale,
        )
        # Wrap proves force-closure topology; hold quality proves transport. On the table a static
        # shallow touch can have hold=1, but cannot raise this minimum because wrap remains zero.
        wrap["hold_quality"] = hold
        wrap["slip_lin"] = slip_lin
        wrap["slip_ang"] = slip_ang
        palm_com_pos = self.robot.data.body_com_pos_w[:, self.palm_idx]
        palm_com_lin = self.robot.data.body_com_lin_vel_w[:, self.palm_idx]
        palm_com_ang = self.robot.data.body_com_ang_vel_w[:, self.palm_idx]
        palm_to_object = self.object.data.root_com_pos_w - palm_com_pos
        slip_lin_w = self.object.data.root_com_lin_vel_w - (
            palm_com_lin + torch.cross(palm_com_ang, palm_to_object, dim=-1)
        )
        slip_ang_w = self.object.data.root_com_ang_vel_w - palm_com_ang
        palm_inv = quat_conjugate(self.palm_quat)
        wrap["slip_lin_palm"] = quat_apply(palm_inv, slip_lin_w)
        wrap["slip_ang_palm"] = quat_apply(palm_inv, slip_ang_w)
        wrap["grasp_quality"] = torch.minimum(wrap["quality"], hold)
        wrap["force_magnitude"] = force_magnitude
        wrap["palm_facing"] = palm_facing
        return wrap

    # ------------------------------------------------------------------ reward
    def _get_observations(self) -> dict:
        # Preserve the old 87-dimensional observation as an exact prefix.  This permits an explicit
        # old-checkpoint migration without shifting its learned lift-feature column:
        # core70 | arm+token16 | lift1 | residual5 | close/contact/phase/transport23.
        d = super()._get_observations()
        clearance = self._object_true_min_z() - self._table_surface_z
        lift_progress = torch.clamp(clearance / self.cfg.lift_success_height, 0.0, 1.0).unsqueeze(-1)
        if not self.cfg.enable_grasp_observations:
            obs = torch.cat([d["policy"], lift_progress], dim=-1)
        else:
            base = d["policy"]
            core = base[:, : -self.cfg.action_space]
            old_action = self.actions[:, : self._n_arm + self._n_tokens]
            residual_action = self.actions[:, self._n_arm + self._n_tokens :]
            signals = self._compute_grasp_signals()
            latch = self._is_grasped.float().unsqueeze(-1)
            confirm_progress = torch.clamp(
                self._contact_steps.float() / self.cfg.grasp_confirm_steps, 0.0, 1.0
            ).unsqueeze(-1)
            release_progress = torch.clamp(
                self._lost_contact_steps.float() / self.cfg.grasp_release_steps, 0.0, 1.0
            ).unsqueeze(-1)
            phase = torch.cat(
                (
                    signals["legal_finger_proximity"],
                    signals["finger_force_strength"],
                    signals["close_quality"].unsqueeze(-1),
                    signals["quality"].unsqueeze(-1),
                    signals["hold_quality"].unsqueeze(-1),
                    1.0 - latch,
                    latch,
                    confirm_progress,
                    release_progress,
                    torch.clamp(signals["slip_lin_palm"] / self.cfg.hold_v_scale, -2.0, 2.0),
                    torch.clamp(signals["slip_ang_palm"] / self.cfg.hold_w_scale, -2.0, 2.0),
                ),
                dim=-1,
            )
            obs = torch.cat(
                (core, old_action, lift_progress, residual_action, phase), dim=-1
            )
        if obs.shape[1] != self.cfg.observation_space:
            raise RuntimeError(
                f"Built {obs.shape[1]} observations, cfg declares {self.cfg.observation_space}."
            )
        self._policy_obs_cache = obs
        return {"policy": obs, "critic": obs}

    def _get_states(self) -> torch.Tensor:
        return getattr(self, "_policy_obs_cache", None)

    def _decode_hand_action(self, hand_action: torch.Tensor) -> torch.Tensor:
        """Decode token9 plus five full-range, non-accumulating distal residuals."""

        if not self.cfg.enable_distal_residual:
            self._last_distal_delta.zero_()
            target = super()._decode_hand_action(hand_action)
            self._last_token_hand_target.copy_(target)
            self._last_raw_hand_target.copy_(target)
            return target
        expected_width = self._n_tokens + self._n_distal_residuals
        if hand_action.shape[1] != expected_width:
            raise ValueError(f"Expected {expected_width} hand actions, got {hand_action.shape[1]}.")
        token = hand_action[:, : self._n_tokens]
        residual = hand_action[:, self._n_tokens :]
        token_target = super()._decode_hand_action(token)
        hand_lower = self.dof_lower[:, self._hand_ids_t]
        hand_upper = self.dof_upper[:, self._hand_ids_t]
        target, delta = apply_asymmetric_joint_residual(
            token_target,
            hand_lower,
            hand_upper,
            residual,
            self._distal_hand_ids,
            validate_indices=False,
        )
        self._last_token_hand_target.copy_(
            torch.maximum(torch.minimum(token_target, hand_upper), hand_lower)
        )
        self._last_distal_delta.copy_(delta)
        self._last_raw_hand_target.copy_(target)
        return target

    def _get_rewards(self) -> torch.Tensor:
        # Staged close/wrap potentials bridge approach to force closure, while the strict shared
        # quality remains the only authority for latch, transport potential and success.
        cfg = self.cfg

        # Reach still targets the four calibrated cross-section points. Its alignment now comes from
        # the nearest true handle surface rather than fake centroid-radial keypoint directions.
        d = self._curr_fingertip_distances
        a = self._finger_align
        other_d = d[:, self._other_ee_idx]
        other_a = a[:, self._other_ee_idx]
        near_val, near_idx = torch.topk(other_d, k=2, dim=1, largest=False)  # thumb + 2 nearest others
        grasp_dist = (d[:, self._thumb_ee_idx] + near_val.sum(dim=-1)) / 3.0
        align = (a[:, self._thumb_ee_idx] + torch.gather(other_a, 1, near_idx).sum(dim=-1)) / 3.0
        signals = self._compute_grasp_signals()
        thumb_contact = signals["thumb_contact"]
        other_count = signals["other_contact_count"]
        valid_contact = thumb_contact & (other_count >= 2)
        palm_facing = signals["palm_facing"]
        hold_quality = signals["hold_quality"]
        slip_lin = signals["slip_lin"]
        slip_ang = signals["slip_ang"]
        q_wrap = signals["quality"]
        q_close = signals["close_quality"]
        q_contact = signals["contact_quality"]
        grasp_quality = signals["grasp_quality"]

        self._is_grasped, self._contact_steps, self._lost_contact_steps, _, _ = update_grasp_latch(
            grasp_quality,
            self._is_grasped,
            self._contact_steps,
            self._lost_contact_steps,
            high_threshold=cfg.grasp_quality_high,
            low_threshold=cfg.grasp_quality_low,
            confirm_steps=cfg.grasp_confirm_steps,
            release_steps=cfg.grasp_release_steps,
        )
        is_grasped_phase = self._is_grasped
        max_force = signals["force_magnitude"].max(dim=-1).values
        safe_grasp = (
            (grasp_quality >= cfg.grasp_quality_high)
            & (max_force <= cfg.grasp_bonus_max_force)
        )
        self._safe_grasp_steps = torch.where(
            safe_grasp, self._safe_grasp_steps + 1, torch.zeros_like(self._safe_grasp_steps)
        )
        first_stable_grasp = (
            is_grasped_phase
            & (self._safe_grasp_steps >= cfg.grasp_confirm_steps)
            & (~self._grasp_bonus_given)
        )
        self._grasp_bonus_given = self._grasp_bonus_given | first_stable_grasp

        clearance = self._object_true_min_z() - self._table_surface_z
        lift_fraction = torch.clamp(clearance / cfg.lift_success_height, 0.0, 1.0)

        # Gamma-correct potential shaping cannot be farmed by repeatedly approaching and retreating.
        # The first post-reset sample only initializes the potential and pays no state-only reset bonus.
        potential_ready = self._potential_initialized.float()
        close_delta = cfg.shaping_discount * q_close - self._prev_close_quality
        wrap_delta = cfg.shaping_discount * q_wrap - self._prev_wrap_quality
        r_close_progress = cfg.close_progress_scale * close_delta * potential_ready
        r_wrap_progress = cfg.wrap_progress_scale * wrap_delta * potential_ready
        # Keep the one-shot strict-latch event as the end of the close sequence.  Unlike the removed
        # contact/hold occupancies it cannot be farmed by waiting or by drop/regrasp cycles.
        r_grasp = cfg.grasp_bonus * first_stable_grasp.float()
        transport_x = torch.clamp(
            (grasp_quality - cfg.grasp_quality_low)
            / max(cfg.grasp_quality_high - cfg.grasp_quality_low, 1.0e-6),
            0.0,
            1.0,
        )
        force_safe = (max_force <= cfg.grasp_bonus_max_force).float()
        transport_gate = (
            is_grasped_phase.float()
            * transport_x.square()
            * (3.0 - 2.0 * transport_x)
            * force_safe
        )
        lift_potential = lift_fraction * transport_gate
        lift_delta = cfg.shaping_discount * lift_potential - self._prev_lift_potential
        r_lift_progress = cfg.lift_progress_scale * lift_delta * potential_ready
        newly_successful = self._is_success & (~self._success_paid)
        r_success = cfg.success_bonus * newly_successful.float()
        self._success_paid = self._success_paid | self._is_success

        force_excess = torch.clamp(
            (max_force - cfg.safe_contact_force) / cfg.contact_force_penalty_width, 0.0, 1.0
        )
        force_excess = force_excess.square() * (3.0 - 2.0 * force_excess)
        r_force_penalty = -cfg.force_excess_penalty_scale * force_excess
        residual_action = self.actions[:, self._n_arm + self._n_tokens :]
        r_residual_penalty = -cfg.distal_residual_penalty_scale * residual_action.square().sum(dim=-1)

        self._prev_close_quality.copy_(q_close)
        self._prev_wrap_quality.copy_(q_wrap)
        self._prev_lift_potential.copy_(lift_potential)
        self._potential_initialized.fill_(True)

        object_z = self.object_pos_w[:, 2] - self.scene.env_origins[:, 2]
        actual_lift = object_z - self.object_default_z

        if "log" not in self.extras:
            self.extras["log"] = dict()
        log = self.extras["log"]
        log["grasp_dist_mean"] = grasp_dist.mean()
        log["align_mean"] = align.mean()
        log["palm_facing_mean"] = palm_facing.mean()
        log["valid_contact_frac"] = valid_contact.float().mean()
        log["thumb_contact_frac"] = thumb_contact.float().mean()
        log["is_grasped_phase_frac"] = is_grasped_phase.float().mean()
        log["q_wrap_mean"] = q_wrap.mean()
        log["q_close_mean"] = q_close.mean()
        log["q_contact_mean"] = q_contact.mean()
        log["q_proximity_mean"] = signals["proximity_quality"].mean()
        log["grasp_quality_mean"] = grasp_quality.mean()
        log["palm_gate_mean"] = signals["palm_score"].mean()
        log["align_gate_mean"] = signals["alignment_score"].mean()
        log["opposition_mean"] = signals["opposition_raw"].mean()
        log["hold_quality_mean"] = hold_quality.mean()
        log["slip_lin_mean"] = slip_lin.mean()
        log["slip_ang_mean"] = slip_ang.mean()
        log["object_contact_force_mean"] = signals["force_magnitude"].mean()
        log["object_contact_force_max"] = signals["force_magnitude"].max()
        log["handle_surface_dist_mean"] = self._handle_surface_distances.mean()
        log["avg_ft_to_kp_mean"] = self._curr_fingertip_distances.mean()
        log["clearance_mean"] = clearance.mean()
        log["clearance_max"] = clearance.max()
        log["actual_lift_mean"] = actual_lift.mean()
        log["success_frac"] = self._is_success.float().mean()
        log["lift_ge_3cm_frac"] = (clearance >= 0.03).float().mean()
        log["lift_ge_5cm_frac"] = (clearance >= 0.05).float().mean()
        log["lift_ge_10cm_frac"] = (clearance >= 0.10).float().mean()
        log["lift_ge_20cm_frac"] = (clearance >= 0.20).float().mean()
        log["r_reach_mean"] = torch.zeros((), device=self.device)
        log["r_close_progress_mean"] = r_close_progress.mean()
        log["r_contact_mean"] = torch.zeros((), device=self.device)
        log["r_wrap_mean"] = r_wrap_progress.mean()
        log["r_grasp_mean"] = r_grasp.mean()
        log["r_hold_mean"] = torch.zeros((), device=self.device)
        log["hold_strength_mean"] = transport_x.mean()
        log["r_lift_mean"] = r_lift_progress.mean()
        log["r_lift_progress_mean"] = r_lift_progress.mean()
        log["lift_quality_mean"] = transport_gate.mean()
        log["lift_potential_mean"] = lift_potential.mean()
        log["r_success_mean"] = r_success.mean()
        log["force_excess_mean"] = force_excess.mean()
        log["r_force_penalty_mean"] = r_force_penalty.mean()
        log["r_residual_penalty_mean"] = r_residual_penalty.mean()
        if residual_action.shape[1] > 0:
            log["residual_action_abs_mean"] = residual_action.abs().mean()
            log["residual_action_sat_frac"] = (residual_action.abs() > 0.99).float().mean()
            log["distal_delta_abs_mean"] = self._last_distal_delta.abs().mean()
            log["distal_delta_abs_max"] = self._last_distal_delta.abs().max()

        return (
            r_close_progress
            + r_wrap_progress
            + r_grasp
            + r_lift_progress
            + r_success
            + r_force_penalty
            + r_residual_penalty
        )

    # ------------------------------------------------------------------ termination
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        cfg = self.cfg
        object_z = self.object_pos_w[:, 2] - self.scene.env_origins[:, 2]
        actual_lift = object_z - self.object_default_z

        # Success uses the same robust grasp quality as the phase latch and lift reward. A separate
        # thumb+one-contact condition here previously allowed back-of-hand and weak two-point presses.
        clearance = self._object_true_min_z() - self._table_surface_z
        signals = self._compute_grasp_signals()
        obj_lin = self.object.data.root_com_lin_vel_w.norm(dim=-1)
        obj_ang = self.object.data.root_com_ang_vel_w.norm(dim=-1)
        slow = (obj_lin < cfg.success_max_obj_lin_speed) & (obj_ang < cfg.success_max_obj_ang_speed)
        success_inst = (
            (clearance >= cfg.lift_success_height)
            & slow
            & self._is_grasped
            & (signals["grasp_quality"] >= cfg.grasp_quality_high)
            & (signals["force_magnitude"].max(dim=-1).values <= cfg.grasp_bonus_max_force)
        )
        self._success_steps = torch.where(
            success_inst, self._success_steps + 1, torch.zeros_like(self._success_steps)
        )
        held = self._success_steps >= cfg.success_hold_steps
        self._is_success = self._is_success | held

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        dropped = self.object_pos_w[:, 2] < (self.object_default_z - cfg.drop_height)
        if not cfg.terminate_on_drop:
            dropped = torch.zeros_like(dropped)

        return dropped | self._is_success, time_out

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._grasp_baseline_lift[env_ids] = 1.0e6
        self._grasp_baseline_clearance[env_ids] = 1.0e6
        self._highest_rel_lift[env_ids] = 0.0
        self._grasp_baseline_xy[env_ids] = 0.0
        self._grasp_baseline_quat[env_ids] = 0.0
        self._grasp_baseline_quat[env_ids, 0] = 1.0
        self._success_steps[env_ids] = 0
        self._contact_steps[env_ids] = 0
        self._lost_contact_steps[env_ids] = 0
        self._is_grasped[env_ids] = False
        self._grasp_bonus_given[env_ids] = False
        self._safe_grasp_steps[env_ids] = 0
        self._grasp_age[env_ids] = 0
        self._success_paid[env_ids] = False
        self._lift_bonus_given[env_ids] = False
        self._prev_close_quality[env_ids] = 0.0
        self._prev_wrap_quality[env_ids] = 0.0
        self._prev_lift_potential[env_ids] = 0.0
        self._potential_initialized[env_ids] = False

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
