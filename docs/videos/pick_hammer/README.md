# 048 hammer full-chain demo clips (2026-08-02)

Five-stage chain on the YCB 048 claw hammer, single policy per stage, no IK, no scripting
between stages: home spawn (full ±180° yaw) → nudge to grasp-ready pose → close/latch →
0.35 m lift → in-hand reorientation (index fingertip to the handle-neck functional point,
strike face down, deploy gates 2.0 cm / cos 0.85).

Checkpoints: 04 nudge / 05 close / 08 inhand (logs/flashsac/pick_hammer/).
Vector eval: 41.4% / 30.1% end-to-end (2×256 episodes).  The floating hammer is the
goal-pose marker.
