# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""xArm7 + XHand multi-stage pick + in-hand reorientation on the YCB 048 claw hammer.

Object swap of ``pick_tool_token``: the same env class, staged grasp chain (nudge -> close ->
carry) and in-hand reorientation modes run against the google_16k 048_hammer scan; only the
config (mesh, rest pose, keypoints, handle frame, functional point) changes.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Pick-Hammer-Token-Direct-v0",
    entry_point="xhand_inhand.tasks.direct.pick_tool_token.pick_tool_token_env:PickToolTokenEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_hammer_token_env_cfg:PickHammerTokenEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
