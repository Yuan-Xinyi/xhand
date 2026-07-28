# Carry stage demos (2026-07-28)

SimToolReal-style consecutive pose goals: from a fresh stable latch, the
FlashSAC carry policy (66_carry_v1_s21, IK retired) moves the held hammer to
random pose goals (position box +-10/10/8cm, yaw full circle, roll/pitch
+-0.6rad; 4cm / 0.35rad tolerances).  Each reach resamples the goal in place;
only a drop ends the episode.  ~2.5 goals per 15s episode (128-env probe).

Clips: 5, 6 and 3 consecutive goals in single episodes (goal marker visible).
