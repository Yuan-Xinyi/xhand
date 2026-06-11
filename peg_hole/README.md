# CSAC Peg-in-Hole Insertion (Isaac Lab Factory)

Conservative Soft Actor-Critic (CSAC) applied to a contact-rich peg-in-hole
insertion task, built **on top of Isaac Lab's Factory `PegInsert`** (the contact
physics, OSC impedance control, and assets are reused, not reimplemented).

```
peg_hole/
  csac.py                 # CSAC core (pure PyTorch, no Isaac Lab) + self-test
  sigma_tau_schedule.py   # contact-phase sigma/tau schedule (pure tensor) + self-test
  buffer.py               # GPU ring replay buffer (stores sigma/tau at s AND s') + self-test
  insertion_env.py        # subclass of FactoryEnv: 6D wrench obs + staged reward + contact_signals
  env_cfg.py              # @configclass: every tunable parameter
  train.py                # AppLauncher -> env -> schedule -> CSAC.update -> safety
  bc_pretrain.py          # scripted demos + BC -> demos/bc_actor.pt (pi_e prior)
  verify_signals.py       # foundation check: force signal -> sigma/tau switching
  demos/                  # bc_actor.pt, csac_ckpt.pt, metrics_<algo>.csv land here
```

## Recommended verification path (NO demonstrations needed)

BC is only an accelerator under sparse rewards; CSAC is self-consistent without it
(`tau` defaults to "pi_e = previous policy"). The goal of initial verification is
**"confirm the foundation didn't collapse"**, not "learn to insert". Do it in order:

```bash
conda activate env_isaaclab          # required (base python lacks isaacsim/torch)

# 0) decoupled core sanity (no simulator)
python csac.py ; python sigma_tau_schedule.py ; python buffer.py

# 1) force-signal -> schedule foundation (random actions, no RL)
python verify_signals.py --headless --num_envs 32 --steps 150   # writes /tmp/verify_signals.txt
#    expect: free space force_mag low, jumps on contact; phase shifts approach->contact/insert,
#    sigma drops ~0.2->~0.06, tau rises ~0.1->~0.7.

# 2/3) "can it learn?" on an EASY target, shaping reward (NOT sparse), no demos:
python train.py --headless --num_envs 128 --easy --reach_only --no_bc --algo csac --total_steps 100000
# 4) control arm: SAC = CSAC with tau->0, sigma=const (isolates the tau term)
python train.py --headless --num_envs 128 --easy --reach_only --no_bc --algo sac  --total_steps 100000
#    watch demos/metrics_<algo>.csv: reward (should rise), critic_loss (should converge),
#    q_mean (steady rise, not exploding). Diagnosis:
#      both flat       -> task/reward/action-space bug, not CSAC (check the env)
#      SAC jittery, CSAC stable -> tau term working, scale up
#      CSAC clearly worse        -> sigma/tau wiring or the -tau*log pi_e terms; recheck csac.py
```

`verify_signals.py` and the metrics CSV exist because the Isaac app swallows stdout;
the CSV also logs the numerical sanity checks (action-saturation fraction, NaN logp
guard, force-aborts-in-free-space count).

## Full training runs (after the foundation is green)

```bash
# (optional) collect scripted demos + behavior-clone the pi_e prior
python bc_pretrain.py --headless --num_envs 64 --demo_steps 6000

# full staged-reward CSAC (loads demos/bc_actor.pt into actor AND actor_prev if present)
python train.py --headless --num_envs 128

# sparse reward — only do this WITH a BC prior; sparse + no prior won't explore
python train.py --headless --num_envs 128 --sparse

# constant schedule (sigma=0.2, tau=0.5) — use if the force signal is untrustworthy
python train.py --headless --num_envs 128 --schedule_mode constant
```

Quick local checks (no simulator needed):

```bash
python csac.py               # runs CSAC.update on random data
python sigma_tau_schedule.py # prints phase/force schedule outputs
python buffer.py             # add/sample round-trip
```

## Step-0 API verification checklist (Isaac Lab 2.3.2, confirmed in source)

| Item | Confirmed value |
|---|---|
| Namespace | `isaaclab` (new), at `/disk2/IsaacLab` |
| Factory env / cfg | `FactoryEnv(DirectRLEnv)` / `FactoryEnvCfg`; PegInsert = `FactoryTaskPegInsertCfg` |
| Gym id | `Isaac-Factory-PegInsert-Direct-v0` |
| Direct hooks | `_get_observations` → `{"policy","critic"}`, `_get_rewards`→(N,), `_get_dones`→(terminated,truncated), `_pre_physics_step`, `_apply_action`, `_reset_idx`, `_setup_scene`, `_compute_intermediate_values`, `_get_factory_obs_state_dict`, `_get_curr_successes` |
| Action | **6D OSC delta-pose impedance** (pos×0.02 m, rot×0.097 rad → `ctrl_target_fingertip_midpoint_*` → `factory_control.compute_dof_torque`) |
| Peg pose | `self.held_pos`, `self.held_quat` |
| Hole pose | `self.fixed_pos`, `self.fixed_quat`, `self.fixed_pos_obs_frame` (hole tip) |
| Contact force | **No measured wrench in obs.** `self.applied_wrench` = *commanded* OSC wrench (6D). `activate_contact_sensors=True` on robot + held + fixed assets, but no sensor is read by Factory |
| Success | `_get_curr_successes`: `xy_dist < 0.0025` AND `z_disp < height·success_threshold` (PegInsert `success_threshold=0.04`, hole `height=0.05`) |
| `@configclass` | `isaaclab.utils.configclass`; `AppLauncher` from `isaaclab.app` |

## The three hard constraints — how each is honored

1. **OSC / impedance action (never raw torques).** Satisfied unchanged: Factory's
   6D action already maps to task-space delta-pose impedance control. We only add a
   unit-box action clamp in the safety layer. Entropy is therefore added in
   task space (probing approach directions near the hole), not in torque space.

2. **6D force/torque in the observation.** **Conflict:** Factory exposes no
   *measured* wrench. Resolution (`cfg.wrench_source`):
   - `"commanded"` (**default**) — uses `self.applied_wrench`, the 6D OSC wrench.
     It is a genuine force/torque and, during contact, reflects the contact
     reaction through the impedance law. Always available; guarantees
     `python train.py` runs out of the box.
   - `"contact"` (opt-in) — adds a `ContactSensor` on the peg body
     (`/World/envs/env_.*/HeldAsset/forge_round_peg_8mm`, verified via prim probe
     to carry `PhysxContactReportAPI`), reads `.data.net_forces_w` for the
     measured 3D force, and estimates the 3D moment as `(held_pos − fingertip_pos) × F`.
     **Known limitation in this build:** the PegInsert peg is a *single-link
     articulation* (the body is both the articulation root and the rigid body),
     and `ContactSensor`'s `create_rigid_body_view` resolves the wrong body count
     for that topology → `RuntimeError: Failed to initialize contact reporter`.
     So measured force is not available here without an Isaac-side fix; commanded
     is the working default.

   **Verified (verify_signals.py):** the commanded wrench cleanly separates the
   regimes — free space ≈ 0.4 N (OSC effort baseline, not 0), contact 1.5–1.9 N —
   and `contact_force_thresh = 1.0` sits between them, so the schedule switches
   correctly (σ 0.18→0.06, τ 0.15→0.67 on contact, reverting in free space).
   Because the free-space baseline isn't 0, if you ever distrust the signal use
   `--schedule_mode constant` (σ=0.2, τ=0.5) per the paper defaults.

   Either way the policy obs is `[fingertip_pos_rel_fixed, fingertip_quat,
   ee_linvel, ee_angvel, **wrench(6)**, fingertip_pos, held_pos_rel_fixed,
   held_quat, gripper_width] + prev_actions` = **36 dims** (verified).

3. **Low sigma during contact.** `sigma_tau_schedule.py` lowers `sigma` and raises
   `tau` as the peg moves approach → align → contact → insert, fed by
   `contact_signals()`. Defaults match the spec
   (`sigma_by_phase=(0.20,0.15,0.08,0.05)`, `tau_by_phase=(0.10,0.25,0.50,0.80)`).

## CSAC algorithm (csac.py)

Tanh-squashed diagonal-Gaussian policy (reparameterized), twin critics (min for
anti-overestimation). At the start of every `update`, the actor is snapshotted
into `actor_prev` = `pi_e` (Algorithm 1: `phi_p ← phi`), making the
relative-entropy term a one-step trust region; a BC prior initializes `pi_e`.

- **Critic target** (`a' ~ pi(·|s')`):
  `y = r + γ(1−d)[ min_j Q̄_j(s',a') − σ'·log π(a'|s') − τ'·(log π(a'|s') − log π_e(a'|s')) ]`
- **Critic loss:** `MSE(Q1(s,a), y) + MSE(Q2(s,a), y)`
- **Actor loss** (`ã ~ pi(·|s)`):
  `J = (τ+σ)·log π(ã|s) − τ·log π_e(ã|s) − min_i Q_i(s,ã)`

`σ`, `τ` are **per-sample** tensors from the schedule (every entropy /
relative-entropy term is weighted element-wise) — the only structural change vs.
vanilla SAC, plus the two `−τ·log π_e` anchor terms. Nets `(256,256)`, `lr=3e-4`,
`γ=0.99`, soft-target `ρ=0.005`. The replay buffer stores `σ,τ` at **both** `s`
(actor loss) and `s'` (critic target).

## Staged reward (insertion_env.py `_get_rewards`)

```
dense:  r = −w_reach·xy_dist
            − w_align·angle_err·[xy_dist < align_dist]
            + w_insert·depth
            + r_success·[success]
            − w_force·max(0, |F| − force_safe)
sparse: r = r_success·[success] − w_force·max(0, |F| − force_safe)
```
All weights/bands live in `env_cfg.RewardCfg`. `contact_signals()` returns
`(force_mag, depth, xy_err, ori_err)` (each `(num_envs,)`), reusing Factory's
peg/hole pose helpers.

## Safety layer (CSAC is conservative in *value*, not in *force*)

- **Action clamp** to the unit box (`train.py`).
- **Force-abort termination:** `|F| > cfg.safety.force_abort_thresh` ends the
  episode (`_get_dones`).
- **Workspace box:** Factory already clips the position target to ±5 cm of the
  hole frame (`_apply_action`); we keep that as the workspace constraint.

## Evaluation (operational metrics, not max-average-return)

Track: success rate, steps-to-insertion, peak contact force, force-limit
violations, sim-to-real drop. Ablations: (a) SAC = set all `tau`→0 (isolates the
relative-entropy term); (b) pure BC (isolates the RL contribution). For
sim-to-real, randomize friction / hole pose / peg size — the `tau`-term EMA
absorbs the resulting non-stationarity.

## Notes / assumptions

- The "attached paper" found in the workspace
  (`202604_Yuan/main.pdf`) is about **diffusion-based IK**, not CSAC, so the
  algorithm here is implemented strictly from the task spec.
- Stdout is swallowed by the Isaac Sim app launcher; training progress prints may
  not show in piped logs — they appear on an interactive terminal.
- Force-abort can reset a subset of envs; Factory's `_reset_idx` supports
  `env_ids` but was written assuming synchronous resets. Disable via
  `cfg.safety.force_abort=False` if you observe reset desync.
