# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FR3 + XHand grasp-and-reorient task (Direct workflow, dexsuite-aligned, cube)."""

import gymnasium as gym

from . import agents

gym.register(
    id="Fr3-Reorient-Cube-Direct-v0",
    entry_point=f"{__name__}.fr3_reorient_env:Fr3ReorientEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_reorient_env_cfg:Fr3ReorientEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
