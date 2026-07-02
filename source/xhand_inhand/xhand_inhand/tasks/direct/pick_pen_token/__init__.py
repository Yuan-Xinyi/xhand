# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""xArm7 + XHand pick-a-pen with CrossDex action tokenization (Direct workflow).

Grasp the pen off the table, lift it 10 cm, then re-orient it so the tip points
straight down (pen long axis within ~20 deg of the table normal) with the tip
protruding below the hand (unoccluded by the fingers).
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Pick-Pen-Token-Direct-v0",
    entry_point=f"{__name__}.pick_pen_token_env:PickPenTokenEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_pen_token_env_cfg:PickPenTokenEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
