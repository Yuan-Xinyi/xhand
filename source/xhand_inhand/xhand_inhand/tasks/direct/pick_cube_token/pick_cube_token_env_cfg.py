# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow config: xArm7 + XHand pick-a-cube with CrossDex action tokenization.

Identical task / reward / scene to ``pick_cube``, but the HAND is no longer driven by
12 raw relative joint deltas. Instead the policy outputs a 9-dim eigengrasp *token*
(CrossDex action tokenization). Each step the token is decoded to a MANO pose and
retargeted (offline-trained NN) to 12 absolute xhand joint targets.

Action layout (16 = 7 + 9):
  * [0:7]  arm joint deltas   (relative position control, same as pick_cube)
  * [7:16] hand eigengrasp token in [-1, 1] -> absolute hand joint targets

See ``tools/crossdex_retarget`` for the offline retargeting-network build.
"""

import os

from isaaclab.utils import configclass

from ..pick_cube.pick_cube_env_cfg import PickCubeEnvCfg

# repo root: .../tasks/direct/pick_cube_token/<this file>  ->  parents[6] == repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 6)))
_RETARGET_DIR = os.path.join(_REPO_ROOT, "tools", "crossdex_retarget", "models")


@configclass
class PickCubeTokenEnvCfg(PickCubeEnvCfg):
    # 16 = 7 arm joint deltas + 9 hand eigengrasp token
    action_space = 16
    n_hand_tokens = 9
    # obs = pick_cube obs (89) minus the 19-dim prev-action plus the 16-dim prev-action = 86
    observation_space = 86

    # offline-trained CrossDex retargeting network (eigengrasp token -> xhand joints)
    retarget_weights_path = os.path.join(_RETARGET_DIR, "retarget_nn_xhand.pt")
    retarget_meta_path = os.path.join(_RETARGET_DIR, "retarget_nn_xhand_meta.pkl")
