# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import json

import gymnasium as gym
import torch
import math

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster

from cfgs import MoralFlatEnvCfg, MoralRoughEnvCfg

class MoralEnv(DirectRLEnv):
    cfg: MoralFlatEnvCfg | MoralRoughEnvCfg

    def __init__(self, cfg: MoralFlatEnvCfg | MoralRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # get curriculum status
        self._curriculum_learning = False
        self._mean_terrain_level = None
        if cfg.terrain.terrain_type == "generator":
            if cfg.terrain.terrain_generator.curriculum:
                self._curriculum_learning = True

        self._temporal_observations = torch.zeros(
            self.num_envs,
            self.cfg.temporal_buffer_size,
            gym.spaces.flatdim(self.single_observation_space["policy"]) + gym.spaces.flatdim(self.single_action_space),
            device=self.device
        )

        # store error between robot velocity and command velocity
        self._velocity_error = torch.full((self.num_envs,), float('inf'), device=self.device)

        # get ground-truth morphology and actuator parameters
        self._morphology = None
        self._actuator_gains = None
        with open(
            os.path.join(self.cfg.asset_dir, 'morphology_params.json'), 'r'
        ) as file:
            # params = torch.Tensor(list((json.load(file).values()))[:self.num_envs]).to(device=self.device)
            params = torch.Tensor(list((json.load(file)['quadruped_87.usda']))).to(device=self.device)
            params = params.expand(self.num_envs, -1)
            # split morphology data and actuator gains
            self._morphologies = params[:, :9]
            self._actuator_gains = params[:, 9:]

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }
        
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies((".*THIGH", ".*HIP"))
        self._died_id, _ = self._contact_sensor.find_bodies(("base", ".*THIGH", ".*HIP"))

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, MoralRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        # self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        
        height_data = None
        if isinstance(self.cfg, MoralRoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        
        # feet data (contact and height)
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        feet_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._feet_ids], dim=-1), dim=1)[0] > 1.0
        ).to(self.device)
        feet_height = self._robot.data.body_pos_w[:, self._feet_ids][..., -1].to(self.device)

        # ground truth value for training morph-net
        morph_target = torch.cat(
            [
                tensor
                for tensor in(
                    self._morphologies,
                    self._robot.data.root_lin_vel_b
                )
                if tensor is not None
            ],
            dim=-1
        )

        # actor observations
        policy_obs = torch.cat(
            [
                tensor
                for tensor in (
                    # self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )

        # critic observation
        critic_obs = torch.cat(
            [
                tensor
                for tensor in (
                    policy_obs,
                    morph_target,
                    # height_data,
                    # feet_height,
                    # feet_contact,
                    # self._actuator_gains,
                )
                if tensor is not None
            ],
            dim=-1
        )

        # update temporal observation buffer with latest action and observation
        self._update_temporal_observations(torch.cat((policy_obs, self._actions), dim=-1))
        
        # calculate velocity error 
        # self._update_velocity_error(None)

        # print(f"[DEBUG] Policy: {policy_obs.shape}")
        # print(f"[DEBUG] Critic: {critic_obs.shape}")
        # print(f"[DEBUG] Target: {morph_target.shape}")
        # print(f"[DEBUG] Temporal: {self._temporal_observations.flatten(1, 2).shape}")

        observations = {
            "policy": policy_obs,
            "critic": critic_obs,
            "morph_obs": self._temporal_observations.flatten(1, 2),
            "morph_target": morph_target,
            "mean_level": self._mean_terrain_level,
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._died_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        
        # up_vector = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        # up_vector = math_utils.quat_rotate(self._robot.data.root_quat_w, up_vector)
        # tilt_angle = torch.acos(up_vector[:, 2])
        # died = tilt_angle > math.pi / 3  # 60 degrees

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        
        if isinstance(self.cfg, MoralRoughEnvCfg):
            self._set_curriculum_lvl(env_ids)
        self._robot.reset(env_ids)
        
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._temporal_observations[env_ids] = 0.0
        self._velocity_error[env_ids] = float('inf')
        
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def _update_temporal_observations(self, latest_observation: torch.Tensor) -> None:
        self._temporal_observations[:, :-1] = self._temporal_observations[:, 1:]
        self._temporal_observations[:, -1] = latest_observation

    def _set_curriculum_lvl(self, env_ids: torch.Tensor | None) -> None:
        move_up = self._velocity_error[env_ids] < self.cfg.velocity_cmd_threshold
        move_down = self._velocity_error[env_ids] > self.cfg.velocity_cmd_threshold
        move_down *= ~move_up
        # self._terrain.update_env_origins(env_ids, move_up, move_down)
        # if the robot solves the most difficult terrain, it stays there
        self._terrain.terrain_levels[env_ids] += 1 * move_up.int() - 1 * move_down.int()
        self._terrain.terrain_levels[env_ids] = torch.clamp(
            self._terrain.terrain_levels[env_ids],
            min=0,
            max=self._terrain.max_terrain_level-1,
        )
        self._terrain.env_origins[env_ids] = self._terrain.terrain_origins[
            self._terrain.terrain_levels[env_ids],
            self._terrain.terrain_types[env_ids]
        ]
        
        self._mean_terrain_level = torch.mean(self._terrain.terrain_levels.float())
    
    def _update_velocity_error(self) -> None:
        env_ids = self._robot._ALL_INDICES
        # get current linear and angular velocity
        current_lin_vel = self._robot.data.root_lin_vel_b[env_ids, :2]
        current_ang_vel = self._robot.data.root_ang_vel_b[env_ids, 2].unsqueeze(dim=-1)
        current_vel = torch.cat([current_lin_vel, current_ang_vel], dim=-1,)
    
        self._velocity_error = torch.sum(torch.square(self._commands - current_vel), dim=1)

