# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

import agents

import cfgs

from .anymal_c_env import AnymalCEnv

##
# Register Gym environments.
##

gym.register(
    id="Isaac-VisNav-Flat-Anymal-C-Direct-v0",
    entry_point="envs:AnymalCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{cfgs.__name__}:AnymalCFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-VisNav-Rough-Anymal-C-Direct-v0",
    entry_point="envs:AnymalCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{cfgs.__name__}:AnymalCRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
    },
)
