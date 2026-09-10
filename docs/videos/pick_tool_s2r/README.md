# Sim2real-ready tool nudge (2026-09-10)

`nudge_forcefree_{1,2,3}.mp4`: the deployment-ready nudge policy for the original 19cm
tool (s2r_05, warm-chain 61 -> force-free -> +delay1 -> +delay2+noise), recorded UNDER the
deployment-tier degradation (5mm/2deg pose noise, 2-step obs delay, force features zeroed).

Observation needs on real hardware: joints + FK + FoundationPose object pose ONLY.

Staged sensory-deprivation ladder (each stage warm-starts the previous; direct combined
DR collapses the policy -- no gradient below ~10% initial success, timeout-penalty
equalization does not rescue it):
  61 baseline 99.6% clean / 47.9% force-free / 69.1% delay-2
  S1 force-free only          -> 97.1%
  S2 + delay 1                -> 98.0%
  S3 + delay 0-2 + pose noise -> 97.0% (training)
Final strict eval: 90.8% at deployment tier (512 eps), 53.3% at doubled degradation.
Factor note: the tool policy leaned on fingertip forces (2mm web pushing), the YCB hammer
policy on geometry -- same recipe, opposite sensory crutches.
