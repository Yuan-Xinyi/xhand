# xArm7 parallel-gripper asset

Build the generated URDF and convert it to the ignored USD before running the task:

```bash
python source/xhand_inhand/xhand_inhand/assets/xarm7_gripper/build_urdf.py
TERM=xterm /disk2/IsaacLab/isaaclab.sh -p /disk2/IsaacLab/scripts/tools/convert_urdf.py \
  source/xhand_inhand/xhand_inhand/assets/xarm7_gripper/xarm7_gripper.urdf \
  source/xhand_inhand/xhand_inhand/assets/xarm7_gripper/xarm7_gripper.usd \
  --fix-base --joint-stiffness 100.0 --joint-damping 15.0 --headless
```
