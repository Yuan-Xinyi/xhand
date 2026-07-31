# Spindle drill: pure finger-driven axle rotation (2026-08-01)

The object is projected every control step onto a 1-DoF manifold (position
pinned, orientation restricted to the twist about the user-annotated handle
axis); arm channels zeroed.  Valve-turning: finger-driven rolling is the only
mechanism.  81_spindle_s36 (3x512, warm from the axial-roll policy): angle
ratchet maxed at PI with 15.7 rotations per 15s episode; scoring is
pose-only (the policy correctly adopts light-contact turning -- demanding a
full latch at confirm had stalled the ratchet at 1.0 rad).

Clips: 9 / 11 / 7 rotations per episode at the full-pi setting.
