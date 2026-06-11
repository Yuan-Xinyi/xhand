"""Contact-phase sigma / tau schedule — pure tensor, decoupled from Isaac Lab.

Mechanism (per the prompt spec):
  * sigma is the temperature of the Boltzmann factor exp(beta*Q): large sigma ->
    strong exploration, but it also smooths the bootstrapped value function.
  * tau is the weight of the memory anchor pi_e^alpha with alpha = tau/(sigma+tau):
    large tau -> small, conservative steps and an exponential moving average over
    the historical Q, i.e. noise suppression.

Therefore the desired contact-phase behavior:
  * Free space (approach / align): HIGH sigma, LOW tau -> explore, find the hole.
  * Contact / insert:              LOW sigma, HIGH tau -> precise, conservative,
    do NOT smooth away the value-function boundary of the "last millimeter".

Inputs (each shape (num_envs,)): contact force magnitude, insertion depth,
xy deviation, orientation error.  Output: per-env (sigma, tau), each (num_envs,).

Two mappings are provided:
  * ``by_phase``  : discrete phase (approach/align/contact/insert) lookup table.
  * ``by_force``  : smooth interpolation driven by contact-force magnitude.

All thresholds and per-phase (sigma, tau) values are configurable; defaults match
the spec: sigma_by_phase=(0.20,0.15,0.08,0.05), tau_by_phase=(0.10,0.25,0.50,0.80).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Phase indices (also used to index the per-phase tables).
APPROACH, ALIGN, CONTACT, INSERT = 0, 1, 2, 3


@dataclass
class SigmaTauConfig:
    # Per-phase tables, ordered [approach, align, contact, insert].
    sigma_by_phase: tuple = (0.20, 0.15, 0.08, 0.05)
    tau_by_phase: tuple = (0.10, 0.25, 0.50, 0.80)

    # Phase-classification thresholds.
    align_xy_thresh: float = 0.01     # m: within this xy error of hole axis -> aligned
    contact_force_thresh: float = 1.0  # N: above this contact force -> in contact
    insert_depth_thresh: float = 0.002  # m: peg tip this far below hole opening -> inserting

    # Smooth (by_force) mapping endpoints. force <= f_lo -> free-space values,
    # force >= f_hi -> contact values; linear blend in between.
    force_lo: float = 0.5    # N
    force_hi: float = 10.0   # N
    sigma_free: float = 0.20
    sigma_contact: float = 0.05
    tau_free: float = 0.10
    tau_contact: float = 0.80

    # Constant ("constant" mode) values. Use these when the force signal is not
    # trustworthy enough to drive phase switching (paper defaults): a wrong force
    # reading triggering the wrong phase is worse than not scheduling at all.
    const_sigma: float = 0.20
    const_tau: float = 0.50


class SigmaTauSchedule:
    def __init__(self, cfg: SigmaTauConfig | None = None, device: str = "cpu"):
        self.cfg = cfg or SigmaTauConfig()
        self.device = device
        self._sigma_tab = torch.tensor(self.cfg.sigma_by_phase, device=device)
        self._tau_tab = torch.tensor(self.cfg.tau_by_phase, device=device)

    # ----------------------------------------------------------- classify
    def classify_phase(self, force_mag, depth, xy_err, ori_err) -> torch.Tensor:
        """Return an integer phase index per env. Precedence: insert > contact > align > approach."""
        c = self.cfg
        n = force_mag.shape[0]
        phase = torch.full((n,), APPROACH, dtype=torch.long, device=force_mag.device)
        is_align = xy_err < c.align_xy_thresh
        is_contact = force_mag > c.contact_force_thresh
        is_insert = depth > c.insert_depth_thresh
        phase = torch.where(is_align, torch.full_like(phase, ALIGN), phase)
        phase = torch.where(is_contact, torch.full_like(phase, CONTACT), phase)
        phase = torch.where(is_insert, torch.full_like(phase, INSERT), phase)
        return phase

    # ----------------------------------------------------------- by phase
    def by_phase(self, force_mag, depth, xy_err, ori_err):
        """Discrete look-up: classify phase, then index the (sigma, tau) tables."""
        phase = self.classify_phase(force_mag, depth, xy_err, ori_err)
        sigma = self._sigma_tab.to(force_mag.device)[phase]
        tau = self._tau_tab.to(force_mag.device)[phase]
        return sigma, tau

    # ----------------------------------------------------------- by force
    def by_force(self, force_mag, depth=None, xy_err=None, ori_err=None):
        """Smooth interpolation by contact-force magnitude (depth/xy/ori unused here)."""
        c = self.cfg
        t = (force_mag - c.force_lo) / max(c.force_hi - c.force_lo, 1e-9)
        t = t.clamp(0.0, 1.0)
        sigma = c.sigma_free + t * (c.sigma_contact - c.sigma_free)
        tau = c.tau_free + t * (c.tau_contact - c.tau_free)
        return sigma, tau

    # ---------------------------------------------------------- constant
    def by_constant(self, force_mag, depth=None, xy_err=None, ori_err=None):
        """All-phase constant (sigma, tau). The force-signal-agnostic fallback."""
        n = force_mag.shape[0]
        sigma = torch.full((n,), self.cfg.const_sigma, device=force_mag.device)
        tau = torch.full((n,), self.cfg.const_tau, device=force_mag.device)
        return sigma, tau

    def __call__(self, signals: dict, mode: str = "by_phase"):
        """signals: dict with keys force_mag, depth, xy_err, ori_err (each (N,))."""
        fn = {"by_force": self.by_force, "constant": self.by_constant}.get(mode, self.by_phase)
        return fn(signals["force_mag"], signals["depth"], signals["xy_err"], signals["ori_err"])


if __name__ == "__main__":
    torch.manual_seed(0)
    sched = SigmaTauSchedule()
    N = 6
    sig = {
        "force_mag": torch.tensor([0.0, 0.2, 2.0, 5.0, 12.0, 0.1]),
        "depth": torch.tensor([0.0, 0.0, 0.0, 0.003, 0.01, 0.0]),
        "xy_err": torch.tensor([0.05, 0.005, 0.004, 0.001, 0.0005, 0.02]),
        "ori_err": torch.zeros(N),
    }
    for mode in ("by_phase", "by_force"):
        s, t = sched(sig, mode=mode)
        print(f"[{mode}] sigma={s.tolist()}")
        print(f"[{mode}] tau  ={t.tolist()}")
    print("phase =", sched.classify_phase(sig["force_mag"], sig["depth"], sig["xy_err"], sig["ori_err"]).tolist())
    print("sigma_tau_schedule self-test OK")
