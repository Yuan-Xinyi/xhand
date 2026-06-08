# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FR3 + XHand pick-a-cube-and-reorient task (Direct workflow, lift-task-aligned)."""

import gymnasium as gym

from . import agents

gym.register(
    id="Pick-Cube-Direct-v0",
    entry_point=f"{__name__}.pick_cube_env:PickCubeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_cube_env_cfg:PickCubeEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
