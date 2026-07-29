# Full chain, zero IK (2026-07-29)

Random hammer pose on the table -> nudge+park (61) -> close (62) -> carry
takeoff policy (74) flies the held tool straight to the first elevated pose
goal (table + 16cm z-floor -- no goal can require table contact) and keeps
cruising through consecutive goals.  Four FlashSAC policies, no IK, no
scripted motion anywhere.  Vector eval: first-goal (takeoff) 95.7% / 98.0%
(256 envs x 2 seeds), ~3.7 further goals per successful env.
