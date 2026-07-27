# Staged chain demos (2026-07-27)

End-to-end pick_tool chain, no home return, no mid-chain scripting:

  v9 nudge+park (61_nudge_v9_gateratchet_s16, 99.2/99.6% strict)
  -> close (62_close4_postv9_s17, retrained on 3140 real handoff states)
  -> min-jerk DLS-IK lift with contact-gated grip servo

End-to-end: 93.0% / 91.4% (256 envs x 2 seeds, home spawn, +-180 deg yaw).
Clips include the post-success hold fix (tool stays held after the 20cm
success latch) and the anti-sway lift conditioning (ik_gain 0.5 + deadbands,
zero-contact fingers excluded from the grip servo).
