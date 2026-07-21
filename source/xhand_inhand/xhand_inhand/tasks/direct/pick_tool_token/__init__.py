# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""xArm7 + XHand pick-a-tool with CrossDex action tokenization (Direct workflow).

Grasp the concave pentagon "tool" off the table and lift it, using the same tokenized
action pipeline and staged lift reward as ``pick_cube_token`` -- only the object changes.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Pick-Tool-Token-Direct-v0",
    entry_point=f"{__name__}.pick_tool_token_env:PickToolTokenEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_tool_token_env_cfg:PickToolTokenEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Pick-Tool-Floating-XHand-Direct-v0",
    entry_point=f"{__name__}.pick_tool_floating_env:PickToolFloatingXHandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pick_tool_floating_env_cfg:PickToolFloatingXHandEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Pick-Tool-Gripper-Direct-v0",
    entry_point=f"{__name__}.pick_tool_gripper_env:PickToolGripperEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_tool_gripper_env_cfg:PickToolGripperEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
