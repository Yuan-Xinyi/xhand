# PPO arm/hand ablation rollout videos

These are deterministic, close-up policy rollouts recorded from the two ablation tasks. Each video
contains one complete 1000-control-step episode (19.98 s at 50 FPS), rendered at 1280 x 720 with
one environment and seed 42. They are policy evaluations, not oracle demonstrations.

| Ablation | Checkpoint | Video | Observed behavior |
| --- | --- | --- | --- |
| Floating XHand | epoch 500 | [MP4](floating_xhand_policy_seed42.mp4) | The hand moves through the bounded workspace but does not establish a grasp or lift. |
| xArm7 + gripper | safe-reward best | [MP4](xarm7_gripper_policy_seed42.mp4) | The arm moves away/folds back without establishing persistent bilateral contact or lift. |

The videos are representative single-environment rollouts for visual inspection. Aggregate claims
in the main report come from the 256-environment deterministic evaluation rather than these two
clips.
