# pick_tool_token — grasp demo

xArm7 + XHand grasping and lifting the concave "pentagon" tool, trained with the
`Pick-Tool-Token-Direct-v0` task (CrossDex tokenized action pipeline + the pick_cube
staged lift reward). Click a file to view it in GitHub's video player.

| clip | description |
|------|-------------|
| [`pick_tool_token_grasp_solo.mp4`](pick_tool_token_grasp_solo.mp4) | single environment, close-up: reach → grasp → lift ~0.30 m |
| [`pick_tool_token_grasp_multi.mp4`](pick_tool_token_grasp_multi.mp4) | 9 environments (random tool poses/yaw), close-up on env 0 |

Checkpoint: mid-training (~epoch 3200 / 5000), best mean return ~610. Peak lift ~0.297–0.299 m
(the 0.30 m success bar is missed by ~1 mm at this checkpoint; the final checkpoint should
cross it). Recorded with `scripts/rl_games/record_tool.py`.
