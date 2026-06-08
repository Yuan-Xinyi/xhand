# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XHand in-hand whiteboard-pen re-orientation task (axis / heading variants)."""

import gymnasium as gym

from . import agents

# the shared, hand-agnostic in-hand manipulation environment (copied into this project)
_inhand_entry = "xhand_inhand.tasks.direct.inhand_manipulation"

##
# Register Gym environments.
##

gym.register(
    id="Xhand-Pen-Direct-v0",
    entry_point=f"{_inhand_entry}.inhand_manipulation_env:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.xhand_pen_env_cfg:XHandPenEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

gym.register(
    id="Xhand-Pen-OpenAI-LSTM-Direct-v0",
    entry_point=f"{_inhand_entry}.inhand_manipulation_env:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.xhand_pen_env_cfg:XHandPenOpenAIEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)
