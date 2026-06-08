# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FR3 + XHand pick-pen task (Direct workflow)."""

import gymnasium as gym

from . import agents

gym.register(
    id="Pick-Pen-Direct-v0",
    entry_point=f"{__name__}.pick_pen_env:PickPenEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_pen_env_cfg:PickPenEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
