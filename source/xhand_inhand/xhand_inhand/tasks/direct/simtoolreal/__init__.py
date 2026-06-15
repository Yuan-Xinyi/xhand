# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SimToolReal dexterous tool-manipulation task (xArm7 + XHand), Direct workflow.

Isaac Lab port of the SimToolReal Isaac Gym environment.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="SimToolReal-Direct-v0",
    entry_point=f"{__name__}.simtoolreal_env:SimToolRealEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simtoolreal_env_cfg:SimToolRealEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
