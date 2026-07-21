# PPO arm/hand ablation (2026-07-21)

## Question

Is the failure of PPO on the reach -> grasp -> latch -> 20 cm lift sequence mainly caused by
xArm7 kinematics or by the XHand dexterous hand?

The comparison uses three physical tasks:

| Task | Robot action | Policy dimensions |
| --- | --- | --- |
| Baseline | xArm7 joints + XHand token/residual | action 21, obs/state 115 |
| Floating XHand | bounded free-root twist + the same XHand token/residual | action 20, obs/state 100 |
| xArm7 gripper | xArm7 joints + one symmetric jaw command | action 8, obs/state 54 |

The floating root is velocity-controlled and workspace-bounded; it is not teleported and contacts
remain dynamic. All tasks use the same hammer, table, randomized hammer reset distribution, true
mesh-clearance success test, and 20 second / 1000 control-step episode.

## Gripper feasibility and friction

The hammer/table retain the baseline friction coefficient 0.5. Only the rubber jaw pads use 2.0.
A deterministic reach-close-lift oracle at the canonical object pose passed before PPO training:

- final true mesh clearance: 0.20346 m;
- grasp latch held at the end;
- final bilateral forces: 1.85 N and 3.60 N;
- maximum contact force: 4.15 N;
- hammer displacement during closing: 0.00201 m.

Thus a PPO failure in this task is not evidence that the gripper is physically unable to hold the
hammer. The oracle is implemented in `scripts/rl_games/gripper_lift_oracle.py`.

## PPO protocol

- seed: 42;
- 1024 parallel environments;
- 500 epochs, horizon 32: 16,384,000 environment steps per task;
- about 16 complete 1000-step episode windows per environment, so late within-episode grasp/lift
  behavior is not truncated;
- identical PPO network and optimizer configuration;
- no BC, curriculum, demonstrations, or action shield;
- one seed by design: this is a bounded causal screen, not a variance study.

An initial gripper reward paid for unsafe impact contact and produced roughly 375 N late-training
force peaks. The final gripper protocol removes force from the geometric approach potential, fades
contact progress to zero between 60 N and 100 N, and makes an unsafe impact cost more than its
one-shot contact reward. This reduced the deterministic evaluation failures from 10,424 to 75 and
is the gripper result reported below.

## Results

Training-time maxima over the full 500 epochs:

| Task | Max true clearance | Any 5 cm lift | Any 20 cm lift | Any success |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.01940 m | no | no | no |
| Floating XHand | 0.000054 m | no | no | no |
| xArm7 gripper (safe reward) | 0.05854 m | yes, 1/1024 envs at one logged step | no | no |

The corrected gripper run still had no persistent latch in the final window and no success during
training. It did, however, discover rare short lifts that neither XHand task discovered.

Finite deterministic evaluation used seed 4242, 256 environments, and 2000 control steps (two full
episodes per environment when no early failure):

| Task/checkpoint | Successes | Timeouts | Failures | Max clearance | Grasped env-step fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline, epoch 500 | 0 | 512 | 0 | 0.000054 m | 0 |
| Floating XHand, epoch 500 | 0 | 512 | 0 | 0.000054 m | 0 |
| Gripper, best checkpoint | 0 | 449 | 75 | 0.06004 m | 0.00000586 |

The gripper produced one 5 cm-or-higher environment-step in evaluation, but it was not a stable
latched lift and never reached 20 cm.

## Conclusion

Removing the arm did not make PPO easier in this budget, so xArm7 kinematics are not the dominant
bottleneck. Replacing XHand with a gripper improved early contact and enabled rare short lifts, so
dexterous-hand complexity contributes to the difficulty, but the gripper still achieved zero stable
20 cm lifts despite proven physical feasibility.

The reliable conclusion is therefore: **the task is not failing solely because of either the arm or
the dexterous hand**. The larger remaining bottleneck is discovering and crediting the complete
approach -> safe bilateral contact -> persistent latch -> lift sequence. With one seed, this screen
does not establish asymptotic sample complexity; it does reject the stronger hypotheses that
removing the arm or swapping in a gripper makes the current plain-PPO task straightforward.

## Artifacts

- Branch: `ablation/ppo-arm-vs-hand` (base `ee5cae3`)
- Training runs: `logs/rl_games/pick_tool_token/0_ablation_{baseline,floating}_300ep_s42`
- Final corrected gripper run: `logs/rl_games/pick_tool_token/0_ablation_gripper_safe_500ep_s42`
- Finite evaluation support: `scripts/rl_games/play.py --max_steps ... --eval_json ...`
