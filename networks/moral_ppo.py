from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torchviz

from .moral_storage import MoralRolloutStorage
from .moral_net import MorAL


class MorALPPO:
    actor_critic: MorAL

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
    ):
        print("[INFO] USING NEW PPO")
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.morph_lr = learning_rate

        # PPO components
        self.beta = 0.1
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.transition = MoralRolloutStorage.Transition()

        # create optimizer param groups (actor-critic and morph-net)
        self.actor_critic_params = []
        self.morph_net_params = []
        for name, parameter in self.actor_critic.named_parameters():
            if "morph" in name:
                self.morph_net_params.append(parameter)
            else:
                print(name)
                self.actor_critic_params.append(parameter)

        print(f"[DEBUG] Actor-Critic params: {len(self.actor_critic_params)}")
        print(f"[DEBUG] Morph-Net params: {len(self.morph_net_params)}")

        # self.optimizer = optim.Adam(self.actor_critic_params, lr=learning_rate)
        # self.morph_net_optimizer = optim.Adam(self.morph_net_params, lr=learning_rate)
        
        # param_groups = [
        #     {"params": self.actor_critic_params, "lr": learning_rate},
        #     {"params": self.morph_net_params, "lr": learning_rate},
        # ]
        # self.optimizer = optim.Adam(param_groups, lr=learning_rate)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # DEBUG
        self._est_vel = None
        self._tar_vel = None

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        morph_obs_shape,
        morph_target_shape,
        action_shape
    ):
        self.storage = MoralRolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            morph_obs_shape,
            morph_target_shape,
            action_shape,
            self.device
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, morph_obs, morph_target):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        
        # Compute the actions and values
        morph_est = self.actor_critic.morph_estimate(morph_obs)
        # concatenate morph-net estimate with observations
        obs = torch.cat((obs, morph_est), dim=-1)
        
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        
        # need to record obs, critic_obs, morph_obs and morph_target before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.morph_observations = morph_obs
        self.transition.morph_targets = morph_target
        
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_regression_loss = 0
        
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        for (
            obs_batch,
            critic_obs_batch,
            morph_obs_batch,
            morph_target_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            # morph-net
            morph_batch = self.actor_critic.morph_estimate(morph_obs_batch)
            # concatenate morph-net estimate with observations
            # obs_batch = torch.cat((obs_batch, morph_batch), dim=-1)
            # actor
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            # critic
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )

            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    # elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    elif kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # update learning rate for actor_critic param group
                    # self.optimizer.param_groups[0]["lr"] = self.learning_rate
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
            

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            
            # Loss
            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Morph-net loss
            reg_loss_func = torch.nn.MSELoss()
            morph_loss = reg_loss_func(morph_batch[:, :9], morph_target_batch.detach()[:, :9])
            velocity_loss = reg_loss_func(morph_batch[:, 9:], morph_target_batch.detach()[:, 9:])
            reg_loss = morph_loss + velocity_loss
            
            # Total loss
            loss = self.beta * reg_loss + (1 - self.beta) * ppo_loss
            # torchviz.make_dot(loss, params=dict(self.actor_cri tic.named_parameters())).render("moral_ppo_graph_updated", format="png")

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic_params, self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_regression_loss += reg_loss.item()

            # DEBUG
            self._est_vel = morph_batch[0].detach().cpu().numpy()
            self._tar_vel = morph_target_batch[0].detach().cpu().numpy()
        
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_regression_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_regression_loss
