"""Registered fixed-intent specialist for OakInk ``flashlight_1``."""

import gymnasium as gym


gym.register(
    id="Functional-Pregrasp-Flashlight-Direct-v0",
    entry_point="xhand_inhand.tasks.direct.pick_tool_token.pick_tool_token_env:PickToolTokenEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.functional_pregrasp_flashlight_env_cfg:FunctionalPregraspFlashlightEnvCfg"
        )
    },
)
