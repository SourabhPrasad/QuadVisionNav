# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
MorAL Quadruped Locomotion
"""

import gymnasium as gym

import agents

import cfgs

from .moral_env import MoralEnv

##
# Register Gym environments.
##

gym.register(
    id="Isaac-VisNav-Flat-Direct-v0",
    entry_point="envs:MoralEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{cfgs.__name__}:MoralFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_moral_cfg:MoralFlatRunnerCfg",
    },
)

gym.register(
    id="Isaac-VisNav-Direct-v0",
    entry_point="envs:MoralEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{cfgs.__name__}:MoralRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_moral_cfg:MoralRoughRunnerCfg",
    },
)

gym.register(
    id="Isaac-VisNav-Test-Direct-v0",
    entry_point="envs:MoralEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{cfgs.__name__}:MoralFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_moral_cfg:MoralTestRunnerCfg",
    },
)
