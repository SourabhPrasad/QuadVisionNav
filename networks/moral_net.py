
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

class MorAL(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_morph_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        morph_hidden_dims=[256, 256, 256],
        actor_critic_activation="elu",
        morph_activations="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        actor_critic_activation = get_activation(actor_critic_activation)
        morph_activation = get_activation(morph_activations)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        mlp_input_dim_m = num_morph_obs
        
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(actor_critic_activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(actor_critic_activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(actor_critic_activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(actor_critic_activation)
        self.critic = nn.Sequential(*critic_layers)

        # Morph Net
        morph_layers = []
        print(f"Morp Input Dim: {mlp_input_dim_m} || Hidden Layer 1: {morph_hidden_dims[0]}")
        morph_layers.append(nn.Linear(mlp_input_dim_m, morph_hidden_dims[0]))
        morph_layers.append(morph_activation)
        for layer_index in range(len(morph_hidden_dims)):
            if layer_index == len(morph_hidden_dims) - 1:
                morph_layers.append(nn.Linear(morph_hidden_dims[layer_index], 12)) #Output: estimated velocity(R3) and morphology(R9)
            else:
                morph_layers.append(nn.Linear(morph_hidden_dims[layer_index], morph_hidden_dims[layer_index + 1]))
                morph_layers.append(morph_activation)
        self.morph = nn.Sequential(*morph_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Morph MLP: {self.morph}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, morph_observations):
        # Get morph-net output (R12: estimated velocity and morphology)
        morph_out = self.morph(morph_observations)
        updated_obs = torch.cat((observations, morph_out), dim=-1)
        mean = self.actor(updated_obs)

        # print(f"[TEST]: MORPH_NET OUTPUT: {morph_out.shape}")
        # print(f"[TEST]: UPDATED OBS: {updated_obs.shape}")
        # print(f"[TEST]: ACTOR OUTPUT: {mean.shape}")
        
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, morph_observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, morph_observations):
        # Get morph-net output (R12: estimated velocity and morphology)
        morph_out = self.morph(morph_observations)
        updated_obs = torch.cat((observations, morph_out), dim=-1)
        
        actions_mean = self.actor(updated_obs)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        # print(f"[TEST]: VALUE OUTPUT: {value.shape}")
        return value
    
    def morph_estimate(self, morph_observations):
        estimate = self.morph(morph_observations)
        return estimate


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid actor_critic_activation function!")
        return None

