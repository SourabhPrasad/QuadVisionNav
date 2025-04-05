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
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        morph_hidden_dims=[258, 128],
        actor_critic_activation="elu",
        morph_activations="relu",
        init_noise_std=1.0,
        **kwargs,
    ):
        # if kwargs:
        #     print(
        #         "MorAL_Net.__init__ got unexpected arguments, which will be ignored: "
        #         + str([key for key in kwargs.keys()])
        #     )
        super().__init__()
        print("[INFO] USING MORAL NETWORK")
        actor_critic_activation = get_activation(actor_critic_activation)
        morph_activation = get_activation(morph_activations)

        mlp_input_dim_a = num_actor_obs #+ 12 # Adding morph net output size
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
        morph_layers.append(nn.Linear(mlp_input_dim_m, morph_hidden_dims[0]))
        morph_layers.append(morph_activation)
        for layer_index in range(len(morph_hidden_dims)):
            if layer_index == len(morph_hidden_dims) - 1:
                # output: morphology(R9) and estimated velocity(R3)
                morph_layers.append(nn.Linear(morph_hidden_dims[layer_index], 12))
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

    def update_distribution(self, observations):
        # get morph-net estimate and concatenate with observations
        # estimate = self.morph(morph_observations).detach()
        # combined_obs = torch.cat((observations, estimate), dim=-1)
        combined_obs = observations
        mean = self.actor(combined_obs)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def morph_estimate(self, morph_observations):
        return self.morph(morph_observations)

    def act_inference(self, observations, **kwargs):
        # estimate = self.morph(morph_observations)
        # combined_obs = torch.cat((observations, estimate), dim=-1)
        combined_obs = observations
        actions_mean = self.actor(combined_obs)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


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
        print("invalid activation function!")
        return None
