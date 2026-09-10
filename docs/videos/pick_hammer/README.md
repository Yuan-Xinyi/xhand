# 048 hammer full-chain demo clips (2026-08-02)

Five-stage chain on the YCB 048 claw hammer, single policy per stage, no IK, no scripting
between stages: home spawn (full ±180° yaw) → nudge to grasp-ready pose → close/latch →
0.35 m lift → in-hand reorientation (index fingertip to the handle-neck functional point,
strike face down, deploy gates 2.0 cm / cos 0.85).

Checkpoints: 04 nudge / 05 close / 08 inhand (logs/flashsac/pick_hammer/).
Vector eval: 41.4% / 30.1% end-to-end (2×256 episodes).  The floating hammer is the
goal-pose marker.

## Force-free nudge under real-pipeline degradation (2026-09-10)

`nudge_forcefree_{1,2,3}.mp4`: the sim2real-ready nudge policy (04c, trained with force
features zeroed + 0-3 step observation delay + FoundationPose-level pose noise) running
under the deployment-tier degradation (5 mm / 2 deg noise, 2-step delay, no force
features).  Ladder eval: 84.2% (512 eps) vs 55.8% for the force-reliant policy; doubled
degradation 64.1%.  Observation needs on real hardware: joints + FK + object pose only.
