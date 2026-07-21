# Floating XHand asset

The ablation uses the stock XHand URDF as a gravity-free, free-root articulation. Generate the
ignored USD before running the task:

```bash
TERM=xterm /disk2/IsaacLab/isaaclab.sh -p /disk2/IsaacLab/scripts/tools/convert_urdf.py \
  source/xhand_inhand/xhand_inhand/assets/xhand2R32/xhand_right.urdf \
  source/xhand_inhand/xhand_inhand/assets/floating_xhand/floating_xhand_free.usd \
  --joint-stiffness 3.0 --joint-damping 0.1 --headless
```

Do not pass `--fix-base`; the task applies bounded linear/angular root velocities directly.
