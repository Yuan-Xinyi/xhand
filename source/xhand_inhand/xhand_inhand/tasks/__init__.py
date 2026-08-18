# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
# (simtoolreal.objects / .xarm7 are asset-build helpers, not task modules -- the former
# launches an Isaac Sim app at import time, the latter imports the optional `wrs` package).
_BLACKLIST_PKGS = [
    "utils",
    ".mdp",
    "simtoolreal.objects",
    "simtoolreal.xarm7",
    # asset-build helper (launches an Isaac Sim app in main(); import is safe but pointless)
    "functional_grasping.gen_hammer_nail_assets",
    # External OakInk asset conversion is intentionally lazy: importing unrelated tasks must
    # not require the optional UniDexFPM dataset.  The cfg entry point imports these only when
    # Functional-Pregrasp-Flashlight-Direct-v0 is selected.
    "functional_pregrasp_flashlight.flashlight_asset",
    "functional_pregrasp_flashlight.functional_pregrasp_flashlight_env_cfg",
]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
