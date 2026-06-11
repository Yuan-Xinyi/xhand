"""Config for the CSAC peg-in-hole insertion env (Isaac Lab 2.3.2).

All tunable parameters are centralized here. ``InsertionEnvCfg`` subclasses the
real Factory PegInsert cfg (confirmed name: ``FactoryTaskPegInsertCfg``) so we
inherit the contact physics, the 6D OSC delta-pose action, and the asset specs
unchanged — we only ADD reward/observation/safety/schedule parameters.
"""

from __future__ import annotations

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

# Confirmed real Factory cfg (see /disk2/IsaacLab/.../direct/factory/factory_env_cfg.py:192).
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg


@configclass
class RewardCfg:
    """Staged-reward weights and bands (all overridable)."""

    sparse: bool = False  # if True: only success bonus + force penalty (pi_e drives exploration)
    # reach_only: drop the insert/success terms -> reward only getting the peg
    # over the hole. The easiest "can it learn?" target for initial verification.
    reach_only: bool = False

    # --- engage-shaping reward (DEFAULT): a continuous potential that pulls the
    # peg from above the hole all the way down into the seated pose. Fixes the
    # "hover above the hole" plateau where the reach term saturates and depth
    # never turns on. r = -w_reach*xy - w_engage*engage_dist + r_success*succ - force_pen.
    use_engage_shaping: bool = True
    w_engage: float = 10.0     # -w_engage * ||peg_base - seated_target||_3d (down+in gradient)
    success_radius: float = 0.01  # m: engage_dist below this counts as success (looser than Factory)

    w_reach: float = 1.0       # -w_reach * xy_dist            (reach: center peg axis over hole)
    w_align: float = 0.5       # -w_align * angle_err in align band (legacy staged path only)
    w_insert: float = 5.0      # +w_insert * depth             (legacy staged path only)
    r_success: float = 10.0    # +r_success on success (lowered 50->10 to curb Q inflation)
    w_force: float = 0.01      # -w_force * max(0,|F|-force_safe)

    align_dist: float = 0.02   # m: xy band within which the align term turns on (legacy path)
    success_depth_frac: float = 0.04  # fraction of hole height counted as success (Factory default)
    force_safe: float = 5.0    # N: soft contact-force budget; penalize the excess


@configclass
class SafetyCfg:
    action_clip: float = 1.0          # clamp commanded action to [-1, 1] box (applied in train.py)
    force_abort_thresh: float = 50.0  # N

    # IMPORTANT: Factory's _reset_idx/randomize_initial_state assume ALL envs reset
    # at the SAME time (synchronous). Per-env termination crashes it (255 vs 256
    # shape mismatch). So we do NOT terminate the env per-env. Instead, success and
    # force-abort are surfaced as masks (env._success_edge / env._abort_mask) and
    # used ONLY as the replay-buffer bootstrap-done flag in train.py — episodes stay
    # fixed-horizon and reset synchronously on timeout. This grounds the value at the
    # insertion moment without violating Factory's reset assumption.
    ground_done_on_success: bool = True  # buffer done=1 at the rising edge of success
    ground_done_on_abort: bool = True    # buffer done=1 when |F| exceeds force_abort_thresh
    # Workspace box: Factory already clips the pos target to +/-5cm of the hole
    # frame (factory_env._apply_action). We keep that as the workspace constraint.


@configclass
class ScheduleCfg:
    """Mirror of sigma_tau_schedule.SigmaTauConfig (kept here so all knobs live in one cfg)."""

    mode: str = "by_phase"  # "by_phase" | "by_force" | "constant"
    sigma_by_phase: tuple = (0.20, 0.15, 0.08, 0.05)
    tau_by_phase: tuple = (0.10, 0.25, 0.50, 0.80)
    align_xy_thresh: float = 0.01
    contact_force_thresh: float = 1.0
    insert_depth_thresh: float = 0.002
    force_lo: float = 0.5
    force_hi: float = 10.0
    sigma_free: float = 0.20
    sigma_contact: float = 0.05
    tau_free: float = 0.10
    tau_contact: float = 0.80
    const_sigma: float = 0.20  # used by "constant" mode (force-signal-agnostic fallback)
    const_tau: float = 0.50


@configclass
class CSACHyperCfg:
    hidden: tuple = (256, 256)
    gamma: float = 0.99
    lr: float = 3e-4
    rho: float = 0.005           # soft target update rate
    buffer_capacity: int = 1_000_000
    batch_size: int = 1024
    learn_start_steps: int = 1000  # env-steps before learning begins
    updates_per_step: int = 1
    total_env_steps: int = 200_000


@configclass
class InsertionEnvCfg(FactoryTaskPegInsertCfg):
    """CSAC insertion env cfg. Inherits the real PegInsert task."""

    # --- our added sub-configs ---
    reward: RewardCfg = RewardCfg()
    safety: SafetyCfg = SafetyCfg()
    schedule: ScheduleCfg = ScheduleCfg()
    csac: CSACHyperCfg = CSACHyperCfg()

    # --- wrench / contact sensing ---
    # "commanded": Factory's commanded OSC wrench self.applied_wrench (6D). This
    #   is a genuine 6D force/torque and, during contact, reflects the contact
    #   reaction through the impedance law. DEFAULT: always available, never
    #   crashes -> guarantees `python train.py` runs out of the box.
    # "contact" : measured ContactSensor force on the peg (+ estimated moment).
    #   Opt-in. Requires the sensor prim_path below to match a peg body prim that
    #   carries the contact-reporter API. If you hit "could not find any bodies
    #   with contact reporter API", adjust held_contact_sensor.prim_path to your
    #   peg's rigid-body prim (the held asset already spawns with
    #   activate_contact_sensors=True, factory_tasks_cfg.py:163).
    wrench_source: str = "commanded"

    # ContactSensor on the held asset (the peg). The HeldAsset Xform itself
    # carries no physics API; its child rigid body does. For PegInsert that body
    # was verified (prim probe) to be ".../HeldAsset/forge_round_peg_8mm" with
    # PhysxContactReportAPI + RigidBodyAPI. For other Factory tasks the body name
    # differs (e.g. the gear/nut mesh) — adjust this path accordingly.
    # Only instantiated when wrench_source == "contact".
    held_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/HeldAsset/forge_round_peg_8mm",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
    )

    def __post_init__(self):
        # Smaller default env count keeps a single GPU comfortable; override via CLI.
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
