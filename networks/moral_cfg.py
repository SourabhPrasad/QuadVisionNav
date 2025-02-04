from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from dataclasses import MISSING

@configclass
class RslRlPpoMoralCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO MorAL network."""
    
    class_name = "MorAL"
    """The policy class name."""

    morph_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the morph network."""

    actor_critic_activation: str = MISSING
    """The activation function for the actor and critic networks."""

    morph_activations: str = MISSING
    """The activation function for the morph network."""

@configclass
class RslRlMoralPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Updated PPO algorithm for MorAL architecture."""
    class_name = "MorALPPO"

@configclass
class RslRlMoralRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Updated configuration of the runner for on-policy algorithms (MorAL)."""
    algorithm: RslRlPpoMoralCfg = MISSING
