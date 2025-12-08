# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
import math
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
from lab.flamingo.tasks.manager_based.locomotion.velocity.sensors import LiftMask
from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Velocity-tracking rewards.
"""


def track_lin_vel_xy_link_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_link_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)

def track_ang_vel_z_link_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_link_ang_vel_b[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)

def lin_vel_z_link_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])

def ang_vel_xy_link_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)

def ang_vel_z_link_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_ang_vel_b[:, 2])

def track_pos_z_exp(
    env: ManagerBasedRLEnv,
    temperature: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,

) -> torch.Tensor:
    """Reward tracking of z position commands using an exponential kernel, considering relative height from wheels to base."""
    # Extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get the current z position of the robot's base
    current_pos_z = asset.data.root_link_pos_w[:, 2]

    # Get the command z position relative to wheels from the command manager
    command_pos_z = env.command_manager.get_command("base_velocity")[:, 3]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = command_pos_z + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = command_pos_z

    # Compute the error between the current height difference and the commanded height difference
    pos_z_error = torch.square(adjusted_target_height - current_pos_z)

    return torch.exp(-pos_z_error * temperature)

'''
def track_pos_z_exp_v2(
    env: ManagerBasedRLEnv,
    temperature: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,

) -> torch.Tensor:
    """Reward tracking of z position commands using an exponential kernel, considering relative height from wheels to base."""
    # Extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get the current z position of the robot's base
    current_pos_z = asset.data.root_link_pos_w[:, 2]

    # Get the command z position relative to wheels from the command manager
    command_pos_z = env.command_manager.get_command("base_velocity")[:, 3]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = command_pos_z + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = command_pos_z

    # Compute the error between the current height difference and the commanded height difference
    pos_z_error = torch.abs(adjusted_target_height - current_pos_z)

    return torch.exp(-pos_z_error * temperature)
'''

def track_base_roll_pitch_exp(
    env: ManagerBasedRLEnv,
    temperature: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),

) -> torch.Tensor:
    """Reward tracking of z position commands using an exponential kernel, considering relative height from wheels to base."""
    # Extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    r, p, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)

    # Map angles from [0, 2*pi] to [-pi, pi]
    roll = (r + math.pi) % (2 * math.pi) - math.pi
    pitch = (p + math.pi) % (2 * math.pi) - math.pi

    # Get the command z position relative to wheels from the command manager
    command = env.command_manager.get_command("roll_pitch")

    # Compute the error between the current height difference and the commanded height difference
    position_error = torch.norm(torch.square(command - torch.stack((roll, pitch), dim=1)), dim = 1)

    return torch.exp(-position_error * temperature)

def flat_euler_angle_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation in the environment frame as Euler angles (roll, pitch, yaw)."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    r, p, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)

    # Map angles from [0, 2*pi] to [-pi, pi]
    roll = (r + math.pi) % (2 * math.pi) - math.pi
    pitch = (p + math.pi) % (2 * math.pi) - math.pi

    rp = torch.stack((roll, pitch), dim=-1)
    return torch.sum(torch.square(rp), dim=1)

def flat_euler_angle_exp(env: ManagerBasedRLEnv, temperature: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation in the environment frame as Euler angles (roll, pitch, yaw).
    
     torch.exp(-temperature *  torch.sum(torch.abs(self.base_euler[:2]), dim=0))
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)

    # Map angles from [0, 2*pi] to [-pi, pi]
    roll = (roll + math.pi) % (2 * math.pi) - math.pi
    pitch = (pitch + math.pi) % (2 * math.pi) - math.pi

    rp = torch.stack((roll, pitch), dim=-1)
    return torch.exp(-temperature * torch.sum(torch.abs(rp), dim=1))

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def safe_landing_motion(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reward function to minimize air time and encourage smooth landings by ensuring wheel contact.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Check contact force to determine if wheels are touching the ground
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 0.1

    # Force minimization reward: penalize higher forces to encourage smooth landing
    force_magnitude = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)
    total_landing_force = torch.sum(force_magnitude, dim=-1)  # Sum over all contact points
    force_minimization_reward = total_landing_force[:, -1] # Encourage lower landing forces

    return force_minimization_reward

def feet_air_time_positive_biped(env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    # reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_air_time_lift_mask(env: ManagerBasedRLEnv,
                            sensor_cfg: SceneEntityCfg,
                            mask_sensor_cfg_left: SceneEntityCfg,
                            mask_sensor_cfg_right: SceneEntityCfg,
                            threshold: float) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds, encouraging locomotion when lift masks are active.

    This function rewards the agent for taking steps up to a specified threshold and also keeps one foot at
    a time in the air. Rewards are further enhanced when lift masks for the left or right foot are active.

    If the commands are small (i.e., the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    left_mask_sensor: LiftMask = env.scene[mask_sensor_cfg_left.name]
    right_mask_sensor: LiftMask = env.scene[mask_sensor_cfg_right.name]

    # Lift mask sensors
    left_lift_mask = left_mask_sensor.data.mask
    right_lift_mask = right_mask_sensor.data.mask

    # Compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1

    # Base reward for alternating stance and air time
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)

    # Apply lift mask to enhance rewards for locomotion
    lift_mask_bonus = left_lift_mask + right_lift_mask  # Shape: [N]
    reward *= lift_mask_bonus  # Amplify reward if lift masks are active

    # No reward for zero command
    reward *= torch.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1) > 0.1

    return reward

def foot_clearance_lift_mask(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg_left: SceneEntityCfg,
    height_sensor_cfg_right: SceneEntityCfg,
    mask_sensor_cfg_left: SceneEntityCfg,
    mask_sensor_cfg_right: SceneEntityCfg,
    target_height: float = 0.3,
) -> torch.Tensor:
    """
    Reward the swinging feet for clearing a dynamically calculated target height off the ground,
    with separate rewards for left and right feet based on lift_mask.

    Args:
        env (ManagerBasedRLEnv): Simulation environment.
        asset_cfg (SceneEntityCfg): Configuration for the asset (feet).
        sensor_cfg_left (SceneEntityCfg): Configuration for the left foot sensor.
        sensor_cfg_right (SceneEntityCfg): Configuration for the right foot sensor.
        target_height (float): Target height for foot clearance relative to the sensor.
        tanh_mult (float): Multiplication factor for velocity term in tanh.

    Returns:
        torch.Tensor: Reward values for the current step.
    """
    # Extract asset and sensor data
    asset: RigidObject = env.scene[asset_cfg.name]
    left_height_sensor: RayCaster = env.scene[height_sensor_cfg_left.name]
    right_height_sensor: RayCaster = env.scene[height_sensor_cfg_right.name]
    left_mask_sensor: LiftMask = env.scene[mask_sensor_cfg_left.name]
    right_mask_sensor: LiftMask = env.scene[mask_sensor_cfg_right.name]

    # Lift mask sensor
    left_lift_mask = left_mask_sensor.data.mask
    right_lift_mask = right_mask_sensor.data.mask

    # Compute dynamic target heights (sensor-based)
    dynamic_target_height_left = target_height + torch.mean(left_height_sensor.data.ray_hits_w[..., 2], dim=1)  # Shape: [N]
    dynamic_target_height_right = target_height + torch.mean(right_height_sensor.data.ray_hits_w[..., 2], dim=1)  # Shape: [N]

    # Compute foot height errors
    left_foot_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], 2]  # Left foot height
    right_foot_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids[1], 2]  # Right foot height

    left_foot_error = torch.square(left_foot_height - dynamic_target_height_left)  # Shape: [N]
    right_foot_error = torch.square(right_foot_height - dynamic_target_height_right)  # Shape: [N]

    # left_foot_reward = torch.exp(-1.5 * left_foot_error)  # Shape: [N]
    # right_foot_reward = torch.exp(-1.5 * right_foot_error)  # Shape: [N]

    # Combine rewards for left and right feet
    total_reward = left_lift_mask * left_foot_error + right_lift_mask * right_foot_error  # Shape: [N]

    # No reward for zero command
    total_reward *= torch.norm(env.command_manager.get_command("base_velocity")[:, :4], dim=1) > 0.1

    return total_reward


class Trajectory_reward(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg,
                 env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.left_height_sensor: RayCaster = env.scene[cfg.params["height_sensor_cfg_left"].name]
        self.right_height_sensor: RayCaster = env.scene[cfg.params["height_sensor_cfg_right"].name]
        self.left_mask_sensor: LiftMask = env.scene[cfg.params["mask_sensor_cfg_left"].name]
        self.right_mask_sensor: LiftMask = env.scene[cfg.params["mask_sensor_cfg_right"].name]

        self.rad_2_deg = 57.2958
        self.phi = torch.pi / 2

        """
            from base coordinate
        """
        # base to hip
        self.left_joint1 = np.array([-0.02305, 0.08, 0.034], dtype=np.float64)
        self.right_joint1 = np.array([-0.02305, -0.08, 0.034], dtype=np.float64)
        # hip to shoulder
        self.left_joint2 = np.array([-0.083025, 0.08, 0.034], dtype=np.float64)
        self.right_joint2 = np.array([-0.083025, -0.08, 0.034], dtype=np.float64)
        # leg lengths
        self.l_1 = np.float64(0.183551 - 0.08)
        self.l_2 = np.sqrt((-0.083025 + 0.221034) ** 2 + (0.034 + 0.137321) ** 2)
        self.l_3 = np.sqrt((-0.221034 + 0.0569727) ** 2 + (-0.137321 + 0.28389) ** 2)
        self.l_4 = np.float64(0.24355 - 0.183551)

        # absolute foot point
        x = 0.1
        y = self.l_1
        z = 0.35
        v_x = 0.15
        v_y = 0.05
        v_z = 0.3


        self.start_point = np.array([[-x, y, -z],
                                     [x/10, -y, -z]], dtype=np.float64)
        self.end_point = np.array([[x/10, y, -z],
                                   [-x, -y, -z]], dtype=np.float64)
        self.start_vel1 = np.array([[v_x, -v_y, v_z/5 ],
                                    [-v_x, -v_y, 0 ]], dtype=np.float64)
        self.end_vel1 = np.array([[-v_x , v_y, -v_z / 10],
                                  [-v_x , v_y, 0]], dtype=np.float64)
        self.start_vel2 = np.array([[-v_x, v_y, 0],
                                    [v_x, v_y, v_z /5 ]], dtype=np.float64)
        self.end_vel2 = np.array([[-v_x , -v_y, 0],
                                  [-v_x , -v_y, -v_z / 10]], dtype=np.float64)


        self.T = 1
        self.async_flag = 0
        self.is_left = 1
        self.is_right = 1 - self.is_left

        #print(env.step_dt) : 0.02

        # 0.005
        self.dt = env.physics_dt
        self.step = 0
        self.last_step = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        height_sensor_cfg_left: SceneEntityCfg,
        height_sensor_cfg_right: SceneEntityCfg,
        mask_sensor_cfg_left: SceneEntityCfg,
        mask_sensor_cfg_right: SceneEntityCfg,
    ) -> torch.Tensor:

        # Height sensor
        left_height = torch.mean(self.left_height_sensor.data.ray_hits_w[..., 2], dim=1)
        right_height = torch.mean(self.right_height_sensor.data.ray_hits_w[..., 2], dim=1)
        # Lift mask sensor
        left_lift_mask = self.left_mask_sensor.data.mask
        right_lift_mask = self.right_mask_sensor.data.mask


        # [4096]
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        # [4096, 6]

        current_joint_pos = self.asset.data.joint_pos[:, asset_cfg.joint_ids]

        self.step = env.episode_length_buf
        t = self.step * self.dt #* env.decimation

        mask2 = t < 2 * self.T
        t[~mask2] = 0
        mask = t < self.T  # mask: [4096], True/False
        t[~mask] -= self.T
        t_normalized = t / self.T

        left_P = torch.empty((t.shape[0], 3), dtype=torch.float32, device=t.device)  # [4096, 3]
        right_P = torch.empty((t.shape[0], 3), dtype=torch.float32, device=t.device)  # [4096, 3]

        # t < self.T
        left_P[mask] = self.cubic_hermite_spline(
            self.start_point[0], self.end_point[0], self.start_vel1[0], self.end_vel1[0], t_normalized[mask]
        ).to(left_P.dtype)

        # t >= self.T
        left_P[~mask] = self.cubic_hermite_spline(
            self.end_point[0], self.start_point[0], self.start_vel2[0], self.end_vel2[0], t_normalized[~mask]
        ).to(left_P.dtype)

        # t < self.T
        right_P[mask] = self.cubic_hermite_spline(
            self.start_point[1], self.end_point[1], self.start_vel1[1], self.end_vel1[1], t_normalized[mask]
        ).to(right_P.dtype)

        # t >= self.T
        right_P[~mask] = self.cubic_hermite_spline(
            self.end_point[1], self.start_point[1], self.start_vel2[1], self.end_vel2[1], t_normalized[~mask]
        ).to(right_P.dtype)

        left_th1, left_th2, left_th3 = self.IK_3dof_leg(left_P[:,0], left_P[:,1], left_P[:,2], self.is_left)
        right_th1, right_th2, right_th3 = self.IK_3dof_leg(right_P[:,0], right_P[:,1], right_P[:,2],self.is_right)
        # compensate init joint pos
        left_th2 -= torch.pi/4
        left_th3 += torch.pi/2
        right_th2 -= torch.pi/4
        right_th3 += torch.pi/2

        left_target_joint_pos = torch.stack([left_th1, left_th2, left_th3], dim = 1).to(current_joint_pos.device)
        right_target_joint_pos = torch.stack([right_th1, right_th2, right_th3], dim = 1).to(current_joint_pos.device)

        left_error_joint_pos = torch.square(current_joint_pos[:, :3] - left_target_joint_pos)
        right_error_joint_pos = torch.square(current_joint_pos[:, 3:] - right_target_joint_pos)

        left_reward = right_lift_mask * torch.exp(-torch.sum(left_error_joint_pos, dim=1) / 0.25**2)
        right_reward = left_lift_mask * torch.exp(-torch.sum(right_error_joint_pos, dim=1) / 0.25**2)

        reward = left_reward + right_reward

        return reward

    def cubic_hermite_spline(self,
                             A,
                             D,
                             U,
                             V,
                             t):
        """
            A : start point
            D : end point
            U : start velocity
            V : end velocity
            U = 3*(B-A)
            V = 3*(D-C)
        """

        t = t.unsqueeze(-1)
        A = torch.from_numpy(A).to(t.device)
        D = torch.from_numpy(D).to(t.device)
        U = torch.from_numpy(U).to(t.device)
        V = torch.from_numpy(V).to(t.device)

        h00 = (2 * t ** 3) - (3 * t ** 2) + 1
        h10 = t ** 3 - (2 * t ** 2) + t
        h01 = (-2 * t ** 3) + (3 * t ** 2)
        h11 = t ** 3 - t ** 2

        return (
                h00 * A + h10 * U +
                h01 * D + h11 * V
        )


    def Rotation_X(self, theta):
        # rotation matrix
        theta = theta.to(dtype=torch.float64)  # float64

        # cos(theta), sin(theta)
        cos_theta = torch.cos(theta)  # [4096]
        sin_theta = torch.sin(theta)  # [4096]

        # row to batch
        R_x = torch.zeros((theta.shape[0], 3, 3), dtype=torch.float64, device=theta.device)  # [4096, 3, 3]

        R_x[:, 0, 0] = 1.0  # first row
        R_x[:, 1, 1] = cos_theta  # second row
        R_x[:, 1, 2] = -sin_theta
        R_x[:, 2, 1] = sin_theta
        R_x[:, 2, 2] = cos_theta

        return R_x

    def IK_3dof_leg(self, x, y, z, is_left: bool):

        """
            z-y plane
        """
        # [num_envs]
        d3 = torch.sqrt(y ** 2 + z ** 2)

        gamma_2 = torch.arcsin((self.l_1 / d3)) #* torch.sin(self.phi))=1
        gamma_3 = torch.pi - gamma_2 - self.phi
        gamma_1 = torch.arctan2(z, y)

        """
            x-z' plane j2, j4
        """

        if is_left == 1:
            theta_1 = -(gamma_3 + gamma_1)
            R = -theta_1 + self.phi - torch.pi / 2
            c = torch.stack([
                torch.full((theta_1.shape[0],), 0.0, dtype=torch.float64, device=theta_1.device),
                self.l_1 * torch.cos(-theta_1),
                self.l_1 * torch.sin(-theta_1)], dim=1)
            j2 = torch.tensor(self.left_joint2).unsqueeze(0).to(theta_1.device) + c
        else:
            theta_1 = gamma_3 - (torch.pi + gamma_1)
            R = theta_1 - self.phi + torch.pi / 2
            c = torch.stack([
                torch.full((theta_1.shape[0],), 0.0, dtype=torch.float64, device=theta_1.device)
                , -self.l_1 * torch.cos(theta_1),
                self.l_1 * torch.sin(theta_1)], dim=1 )
            j2 = torch.tensor(self.right_joint2).unsqueeze(0).to(theta_1.device) + c

        j4 = torch.stack([x, y, z], dim=1)
        # [num_envs, 3]
        j4_2_vec = j4 - j2

        # [num_envs, 3, 1]
        p_2 = torch.matmul(self.Rotation_X(R), j4_2_vec.unsqueeze(-1))

        x_2, z_2 = p_2[:,0], p_2[:,2]
        # [num_envs]
        x_2 = x_2.squeeze(1)
        z_2 = z_2.squeeze(1)

        theta_3 = torch.arccos((x_2 ** 2 + z_2 ** 2 - self.l_2 ** 2 - self.l_3 ** 2) / (2 * self.l_2 * self.l_3)) - torch.pi

        alpha = torch.arctan(self.l_3 * torch.sin(abs(theta_3)) / (self.l_2 + self.l_3 * torch.cos(abs(theta_3))))
        beta = torch.arctan2(z_2, -x_2) + torch.pi / 2
        theta_2 = alpha + beta

        return theta_1, theta_2, theta_3

def adaptive_terrain_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg_left: SceneEntityCfg,
    height_sensor_cfg_right: SceneEntityCfg,
    mask_sensor_cfg_left: SceneEntityCfg,
    mask_sensor_cfg_right: SceneEntityCfg,
    target_clearance: float = 0.3,
    clearance_margin: float = 0.05,
    smoothness_penalty_weight: float = 0.1,
    success_reward: float = 10.0,
) -> torch.Tensor:
    """
    Reward function for adaptive locomotion to handle stairs and flat terrain.

    Args:
        env (ManagerBasedRLEnv): Simulation environment.
        asset_cfg (SceneEntityCfg): Configuration for the asset (feet and wheels).
        height_sensor_cfg_left (SceneEntityCfg): Configuration for the left foot sensor.
        height_sensor_cfg_right (SceneEntityCfg): Configuration for the right foot sensor.
        mask_sensor_cfg_left (SceneEntityCfg): Configuration for the left lift mask sensor.
        mask_sensor_cfg_right (SceneEntityCfg): Configuration for the right lift mask sensor.
        target_clearance (float): Desired height clearance for stairs.
        clearance_margin (float): Allowable error margin for clearance.
        smoothness_penalty_weight (float): Weight for penalizing erratic motions.
        success_reward (float): Reward for successfully clearing an obstacle.

    Returns:
        torch.Tensor: Reward values for the current step.
    """
    # Extract asset and sensor data
    asset: RigidObject = env.scene[asset_cfg.name]
    left_height_sensor: RigidObject = env.scene[height_sensor_cfg_left.name]
    right_height_sensor: RigidObject = env.scene[height_sensor_cfg_right.name]
    left_mask_sensor: LiftMask = env.scene[mask_sensor_cfg_left.name]
    right_mask_sensor: LiftMask = env.scene[mask_sensor_cfg_right.name]

    # Lift mask sensor
    left_lift_mask = left_mask_sensor.data.mask
    right_lift_mask = right_mask_sensor.data.mask

    # Calculate dynamic target heights
    dynamic_target_height_left = target_clearance + torch.mean(left_height_sensor.data.ray_hits_w[..., 2], dim=1)
    dynamic_target_height_right = target_clearance + torch.mean(right_height_sensor.data.ray_hits_w[..., 2], dim=1)

    # Calculate foot height errors
    left_foot_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], 2]
    right_foot_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids[1], 2]

    left_foot_error = torch.abs(left_foot_height - dynamic_target_height_left)
    right_foot_error = torch.abs(right_foot_height - dynamic_target_height_right)

    # Reward for maintaining clearance within margin
    left_clearance_reward = torch.exp(-((left_foot_error - clearance_margin) ** 2))
    right_clearance_reward = torch.exp(-((right_foot_error - clearance_margin) ** 2))

    # Smoothness penalty for erratic z-velocity
    left_velocity_z = torch.abs(asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0], 2])
    right_velocity_z = torch.abs(asset.data.body_lin_vel_w[:, asset_cfg.body_ids[1], 2])
    smoothness_penalty = smoothness_penalty_weight * (left_velocity_z + right_velocity_z)

    # Lift mask activation bonus (encourage proper activation)
    lift_mask_bonus = left_lift_mask * left_clearance_reward + right_lift_mask * right_clearance_reward

    # Success reward for clearing the obstacle
    clearance_success = (
        (left_foot_error < clearance_margin) & (right_foot_error < clearance_margin)
    ).float()
    success_bonus = clearance_success * success_reward

    # Combine rewards
    total_reward = lift_mask_bonus + success_bonus - smoothness_penalty

    return total_reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


class FlamingoAirTimeReward(ManagerTermBase):
    """Reward for longer feet air and contact time with stuck detection and reward for locomotion."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.stuck_threshold: float = cfg.params.get("stuck_threshold", 0.1)
        self.stuck_duration: int = cfg.params.get("stuck_duration", 5)
        self.threshold: float = cfg.params.get("threshold", 0.2)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.stuck_counter = torch.zeros(self.asset.data.root_lin_vel_b.shape[0], device=self.asset.device)

        if not self.contact_sensor.cfg.track_air_time:
            raise RuntimeError("Activate ContactSensor's track_air_time!")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        stuck_threshold: float,
        stuck_duration: int,
        threshold: float,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward calculates the air-time for the feet and applies a reward when the robot is stuck.

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # Extract the necessary sensor data
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

        # Compute the base movement command and its progress
        base_velocity_tensor = env.command_manager.get_command("base_velocity")[:, :3]
        progress = torch.norm(base_velocity_tensor - self.asset.data.root_lin_vel_b, dim=1)
        is_stuck = progress > self.stuck_threshold  # Detect lack of progress

        # Manage the stuck counter and determine stuck status
        self.stuck_counter = torch.where(is_stuck, self.stuck_counter + 1, torch.zeros_like(self.stuck_counter))
        stuck = self.stuck_counter >= self.stuck_duration
        stuck = stuck.unsqueeze(1)

        # Compute the reward based on air time and first contact when stuck
        stuck_air_time_reward = torch.sum((last_air_time - self.threshold) * first_contact * stuck.float(), dim=1)
        # Ensure no reward is given if there is no movement command
        stuck_air_time_reward *= torch.norm(base_velocity_tensor[:, :2], dim=1) > 0.1

        # # Foot clearance reward
        # foot_z_target_error = torch.square(self.asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - self.target_height)
        # foot_velocity_tanh = torch.tanh(
        #     tanh_mult * torch.norm(self.asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
        # )
        # foot_clearance_reward = (
        #     torch.exp(-torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1) / std) * stuck.float()
        # )

        # Final reward: Encourage lifting legs when stuck
        reward = stuck_air_time_reward  # + foot_clearance_reward

        return reward


def stand_origin_base(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize linear velocity on x or y when the command is zero, encouraging the robot to stand still."""
    # Extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the command and check if it's zero
    command = env.command_manager.get_command(command_name)[:, :]
    is_zero_command = torch.all(command == 0.0, dim=1)  # Check per item in batch if command is zero

    # Calculate linear and angular velocity errors
    lin_vel = asset.data.root_lin_vel_b[:, :2]
    lin_vel_error = torch.sum(torch.square(lin_vel), dim=1)

    ang_vel = asset.data.root_ang_vel_b[:, 2]
    ang_vel_error = torch.square(ang_vel)

    # Penalize the linear and angular velocity errors
    velocity_penalty = lin_vel_error + ang_vel_error

    # Calculate deviation from origin position
    current_pos = asset.data.root_pos_w[:, :2]
    position_error = torch.sum(torch.square(current_pos - env.scene.env_origins[:, :2]), dim=1)

    # Penalize the deviation from the origin position
    position_penalty = position_error

    # Apply the penalty only when the command is zero
    penalty = (velocity_penalty + position_penalty) * is_zero_command.float()

    return penalty


def stand_still_base(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize linear velocity on x or y when the command is zero, encouraging the robot to stand still."""
    # Extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the command and check if it's zero
    command = env.command_manager.get_command(command_name)[:, :2]
    is_zero_command = torch.all(command == 0.0, dim=1)  # Check per item in batch if command is zero

    # Calculate linear and angular velocity errors
    lin_vel = asset.data.root_lin_vel_b[:, :2]
    lin_vel_error = torch.sum(torch.square(lin_vel), dim=1)

    ang_vel = asset.data.root_ang_vel_b[:, :2]
    ang_vel_error = torch.sum(torch.square(ang_vel), dim=1)

    # Penalize the linear and angular velocity errors
    velocity_penalty = (lin_vel_error + ang_vel_error) / std**2

    # Calculate deviation from origin position
    current_pos = asset.data.root_pos_w[:, :2]
    position_error = torch.sum(torch.square(current_pos - env.scene.env_origins[:, :2]), dim=1)

    # Penalize the deviation from the origin position
    position_penalty = position_error / std**2

    # Apply the penalty only when the command is zero
    penalty = (velocity_penalty) * is_zero_command.float()

    return penalty


def stand_still(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the robot for standing still when the command is zero, penalizing movement, especially backward movement."""
    # Extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Compute the command and check if it's zero
    command = env.command_manager.get_command(command_name)[:, :3]
    is_zero_command = torch.all(command == 0.0, dim=1)  # Check per item in batch if command is zero

    # Calculate wheel velocity error
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    wheel_vel_error = torch.sum(torch.abs(wheel_vel), dim=1)

    # Penalize backward movement by adding a higher penalty for negative velocities
    # backward_movement_penalty = torch.sum(torch.clamp(wheel_vel, max=0), dim=1)

    # Calculate the reward
    reward = wheel_vel_error / std**2

    # Make sure to only give non-zero reward where command is zero
    reward = reward * is_zero_command.float()

    return reward


def joint_align_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_threshold: float = -1.0,
) -> torch.Tensor:
    """Penalize joint mis-alignments.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)

    if cmd_threshold != -1.0:
        mis_aligned = torch.where(
            cmd <= cmd_threshold,
            torch.abs(
                asset.data.joint_pos[:, asset_cfg.joint_ids[0]] - asset.data.joint_pos[:, asset_cfg.joint_ids[1]]
            ),
            torch.tensor(0.0),
        )
    else:
        mis_aligned = torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids[0]] - asset.data.joint_pos[:, asset_cfg.joint_ids[1]]
        )

    return mis_aligned


def joint_soft_pos_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0] * soft_ratio
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1] * soft_ratio
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def action_smoothness_hard(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using smoothing term."""
    sm1 = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    sm2 = torch.sum(
        torch.square(env.action_manager.action + env.action_manager.prev_action - 2 * env.action_manager.prev2_action),
        dim=1,
    )
    sm3 = 0.05 * torch.sum(torch.abs(env.action_manager.action), dim=1)

    return sm1 + sm2 + sm3


def force_action_zero(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    velocity_threshold: float = -1.0,
    cmd_threshold: float = -1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)

    if cmd_threshold != -1.0 or velocity_threshold != -1.0:
        force_action_zero = torch.where(
            torch.logical_or(cmd.unsqueeze(1) <= cmd_threshold, body_vel.unsqueeze(1) <= velocity_threshold),
            torch.tensor(0.0),
            torch.abs(env.action_manager.action[:, asset_cfg.joint_ids]),
        )
    else:
        force_action_zero = torch.abs(env.action_manager.action[:, asset_cfg.joint_ids])
    return torch.sum(force_action_zero, dim=1)

def base_height_adaptive_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_link_pos_w[:, 2] - adjusted_target_height)

def base_target_range_height_v2(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        min_target_height=0.33126,
        max_target_height=0.37126,
        minimum_height=0.2607,
        sharpness=2.0,
    ):
    # Ensure reward is zero when current height is at or below the minimum height
    asset: RigidObject = env.scene[asset_cfg.name]
    current_height = asset.data.root_link_pos_w[:,2]

    # print(current_height)
    reward = torch.where(min_target_height <= current_height , torch.ones_like(current_height), 1 * (current_height - minimum_height) / (min_target_height - minimum_height))
    reward = torch.where(current_height <= max_target_height, reward, 1 * (1.0 - (current_height - max_target_height) / (min_target_height - minimum_height)))

    return (reward.clamp(min=0.0) ** sharpness)

def track_pos_z(
    env: ManagerBasedRLEnv,
    sharpness: float,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    current_height = asset.data.root_link_pos_w[:, 2]

    target_height = env.command_manager.get_command("base_velocity")[:, 3]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        current_height_rel = current_height - torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        current_height_rel = current_height

    # Compute reward based on current height
    reward = torch.zeros_like(current_height_rel)

    # If below minimum height, return zero reward
    below_minimum = current_height_rel <= minimum_height
    reward[below_minimum] = 0.0

    # If below target height but above minimum height
    below_target = current_height_rel <= target_height
    reward[below_target] = (current_height_rel[below_target] - minimum_height) / (target_height[below_target] - minimum_height)

    # If above target height
    above_target = current_height_rel > target_height
    reward[above_target] = 1.0 - (current_height_rel[above_target] - target_height[above_target]) / (target_height[above_target] - minimum_height)

    # Ensure reward is non-negative and apply sharpness
    reward = torch.clamp(reward, min=0.0) ** sharpness

    return reward

def base_height_range_l2(
    env: ManagerBasedRLEnv,
    min_height: float,
    max_height: float,
    in_range_reward: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Provide a fixed reward when the asset height is within a specified range and penalize deviations."""
    asset: RigidObject = env.scene[asset_cfg.name]
    root_pos_z = asset.data.root_link_pos_w[:, 2]

    # Check if the height is within the specified range
    in_range = (root_pos_z >= min_height) & (root_pos_z <= max_height)

    # Calculate the absolute deviation from the nearest range limit when out of range
    out_of_range_penalty = torch.square(root_pos_z - torch.where(root_pos_z < min_height, max_height, min_height))

    # Assign a fixed reward if in range, and a negative penalty if out of range
    reward = torch.where(in_range, in_range_reward * torch.ones_like(root_pos_z), -out_of_range_penalty)

    return reward

def base_height_range_relative_l2(
    env: ManagerBasedRLEnv,
    min_height: float,
    max_height: float,
    in_range_reward: float,
    root_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Provide a fixed reward when the asset height is within a specified range and penalize deviations."""
    root_asset: RigidObject = env.scene[root_cfg.name]
    wheel_asset: RigidObject = env.scene[wheel_cfg.name]

    root_pos_z = root_asset.data.root_link_pos_w[:, 2]
    # Get the mean z position of the wheels
    # wheel_pos_z = wheel_asset.data.body_pos_w[:, wheel_cfg.body_ids, 2].mean(dim=1)
    # Get the minimum z position of the wheels
    wheel_pos_z = wheel_asset.data.body_pos_w[:, wheel_cfg.body_ids, 2].max(dim=1).values

    # Calculate the height difference
    height_diff = root_pos_z - wheel_pos_z

    # Check if the height difference is within the specified range
    in_range = (height_diff >= min_height) & (height_diff <= max_height)

    # Calculate the absolute deviation from the nearest range limit when out of range
    out_of_range_penalty = torch.square(height_diff - torch.where(height_diff < min_height, max_height, min_height))

    # Assign a fixed reward if in range, and a negative penalty if out of range
    reward = torch.where(in_range, in_range_reward * torch.ones_like(height_diff), -out_of_range_penalty)

    return reward


def base_height_dynamic_wheel_l2(
    env: ManagerBasedRLEnv,
    min_height: float,
    max_height: float,
    in_range_reward: float,
    root_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Provide a fixed reward when the asset height relative to the furthest wheel is within a specified range and penalize deviations."""
    root_asset: RigidObject = env.scene[root_cfg.name]
    wheel_asset: RigidObject = env.scene[wheel_cfg.name]

    root_pos_z = root_asset.data.root_link_pos_w[:, 2]
    # Get the z positions of all the wheels
    wheel_pos_z = wheel_asset.data.body_pos_w[:, wheel_cfg.body_ids, 2]

    # Calculate the height differences for all wheels
    height_diffs = root_pos_z.unsqueeze(1) - wheel_pos_z

    # Find the maximum height difference for each instance (both positive and negative)
    max_height_diff, _ = torch.max(height_diffs, dim=1)
    min_height_diff, _ = torch.min(height_diffs, dim=1)

    # Choose the larger absolute value between max and min height differences
    furthest_height_diff = torch.where(
        torch.abs(max_height_diff) > torch.abs(min_height_diff), max_height_diff, min_height_diff
    )

    # Check if the furthest height difference is within the specified range
    in_range = (furthest_height_diff >= min_height) & (furthest_height_diff <= max_height)

    # Calculate the absolute deviation from the nearest range limit when out of range
    out_of_range_penalty = torch.square(
        furthest_height_diff - torch.where(furthest_height_diff < min_height, max_height, min_height)
    )

    # Assign a fixed reward if in range, and a negative penalty if out of range
    reward = torch.where(in_range, in_range_reward * torch.ones_like(furthest_height_diff), -out_of_range_penalty)

    return reward

def link_x_vel_deviation_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    weel_vel_b = torch.mean(quat_rotate_inverse(asset.data.body_link_quat_w[:, asset_cfg.body_ids], asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]), dim=1)
    root_vel_b = asset.data.root_lin_vel_b
    
    # minimize difference between root and wheel velocities
    return torch.square(weel_vel_b[:, 0] - root_vel_b[:, 0])

def joint_target_deviation_range_l1(
    env: ManagerBasedRLEnv,
    min_angle: float,
    max_angle: float,
    in_range_reward: float,
    cmd_threshold: float = -1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Provide a fixed reward when the joint angle is within a specified range and penalize deviations."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)

    # Get the current joint positions
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # Check if the joint angles are within the specified range
    in_range = (current_joint_pos >= min_angle) & (current_joint_pos <= max_angle)

    # Calculate the absolute deviation from the nearest range limit when out of range
    out_of_range_penalty = torch.abs(current_joint_pos - max_angle)

    if cmd_threshold != -1.0:
        joint_deviation_range = torch.where(
            cmd.unsqueeze(1) <= cmd_threshold,
            torch.where(in_range, in_range_reward * torch.ones_like(current_joint_pos), -out_of_range_penalty),
            torch.tensor(0.0),
        )
    else:
        # Assign a fixed reward if in range, and a negative penalty if out of range
        joint_deviation_range = torch.where(
            in_range, in_range_reward * torch.ones_like(current_joint_pos), -out_of_range_penalty
        )

    # Sum the rewards over all joint ids
    return torch.sum(joint_deviation_range, dim=1)


def joint_target_deviation_range_l1_inv(
    env: ManagerBasedRLEnv,
    min_angle: float,
    max_angle: float,
    in_range_reward: float,
    cmd_threshold: float = -1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Provide a fixed reward when the joint angle is within a specified range and penalize deviations."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)

    # Get the current joint positions
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # Check if the joint angles are within the specified range
    in_range = (current_joint_pos >= min_angle) & (current_joint_pos <= max_angle)

    # Calculate the absolute deviation from the nearest range limit when out of range
    out_of_range_penalty = torch.abs(current_joint_pos - min_angle)

    if cmd_threshold != -1.0:
        joint_deviation_range = torch.where(
            cmd.unsqueeze(1) <= cmd_threshold,
            torch.where(in_range, in_range_reward * torch.ones_like(current_joint_pos), -out_of_range_penalty),
            torch.tensor(0.0),
        )
    else:
        # Assign a fixed reward if in range, and a negative penalty if out of range
        joint_deviation_range = torch.where(
            in_range, in_range_reward * torch.ones_like(current_joint_pos), -out_of_range_penalty
        )

    # Sum the rewards over all joint ids
    return torch.sum(joint_deviation_range, dim=1)


def joint_deviation_zero_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_vel), dim=1)


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for bipeds.

    This reward penalizes contact timing differences between the two feet to bias the policy towards a natural walking gait.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.cmd_threshold: float = cfg.params["cmd_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]

        # Parse and validate synced feet pair names
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if len(synced_feet_pair_names) != 2:
            raise ValueError("This reward requires exactly two pairs of feet for bipedal walking.")

        # Convert foot names to body IDs
        self.foot_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        self.foot_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        cmd_threshold: float = 0.0,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward enforces that one foot is in the air while the other is in contact with the ground.

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # Calculate the asynchronous reward for the two feet
        async_reward = self._async_reward_func(self.foot_0, self.foot_1)

        # only enforce gait if the command velocity or body velocity is above a certain threshold
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
        return torch.where(
            torch.logical_or(cmd > self.cmd_threshold, body_vel > self.velocity_threshold),
            async_reward,
            torch.tensor(0.0),
        )

    """
    Helper functions.
    """

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time

        # Ensure the tensors are properly broadcasted by selecting only the relevant dimensions
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)

        # Summing over the appropriate axis to reduce to the correct size
        return torch.exp(-(se_act_0 + se_act_1) / self.std).squeeze()


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    cmd_threshold: float = -1.0,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    if cmd_threshold != -1.0:
        foot_clearance = torch.where(
            cmd <= cmd_threshold,
            torch.tensor(0.0),
            torch.exp(-torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1) / std),
        )
    else:
        foot_clearance = torch.exp(-torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1) / std)

    return foot_clearance

def stay_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for staying alive."""
    return torch.ones(env.num_envs, device=env.device)

def no_fly(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Reward if only one foot is in contact with the ground."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    latest_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]

    contacts = latest_contact_forces > threshold
    single_contact = torch.sum(contacts.float(), dim=1) == 1

    return 1.0 * single_contact

def pitch_limit_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["base_link"]),
) -> torch.Tensor:
    """Penalize pitch angle (forward/backward tilt) using L2.

    - base_link 쿼터니언을 오일러로 변환
    - pitch를 [-pi, pi] 범위로 매핑
    - pitch^2 를 반환해서 크게 숙일수록 더 큰 penalty
    """
    # base_link 가져오기
    asset: RigidObject = env.scene[asset_cfg.name]

    # world frame에서의 roll, pitch, yaw
    r, p, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)

    # [0, 2*pi] 범위를 [-pi, pi]로 매핑
    pitch = (p + math.pi) % (2 * math.pi) - math.pi

    # 0(rad)에서의 L2 penalty
    return torch.square(pitch)


def body_lin_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """몸(링크)들의 선형 가속도가 클수록 페널티를 주는 항.

    - asset_cfg.body_ids 에 지정된 링크들의 world 기준 선형 가속도 사용
    - 각 링크의 ||a|| 를 더해서 배치별 스칼라 값으로 반환
    """
    # 로봇 articulation 가져오기
    asset: Articulation = env.scene[asset_cfg.name]

    # [num_envs, num_bodies, 3] : 각 링크의 선형 가속도 (world frame)
    lin_acc = asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :]

    # 각 링크의 가속도 norm → env마다 전부 합산 → [num_envs]
    return torch.sum(torch.norm(lin_acc, dim=-1), dim=1)

def stand_still(
    env, lin_threshold: float = 0.05, ang_threshold: float = 0.05, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    penalizing linear and angular motion when command velocities are near zero.
    """

    asset = env.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_w[:, :2]
    base_ang_vel = asset.data.root_ang_vel_w[:, -1]

    commands = env.command_manager.get_command("base_velocity")

    lin_commands = commands[:, :2]
    ang_commands = commands[:, 2]

    reward_lin = torch.sum(
        torch.abs(base_lin_vel) * (torch.norm(lin_commands, dim=1, keepdim=True) < lin_threshold), dim=-1
    )

    reward_ang = torch.abs(base_ang_vel) * (torch.abs(ang_commands) < ang_threshold)

    total_reward = reward_lin + reward_ang
    return total_reward

def leg_symmetry(env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Reward regulate abad joint position."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_rotate_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    leg_symmetry_err = torch.abs(feet_pos_b[:, 0, 1]) - torch.abs(feet_pos_b[:, 1, 1])

    return torch.exp(-leg_symmetry_err ** 2 / std**2)

def com_over_support_xy_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    support_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["left_wheel_static_link", "right_wheel_static_link"],
    ),
):
    """CoM가 두 바퀴(지지선) 중심에서 얼마나 멀리 떨어져 있는지 XY 평면에서 L2 penalty.

    - env: ManagerBasedRLEnv
    - asset_cfg: 로봇 전체 (기본값 "robot")
    - support_cfg: 지지점(여기서는 두 바퀴 링크들을 사용)
    반환:
        shape: (num_envs,)  — 각 env마다 (CoM_xy - support_center_xy)^2 의 합
    """
    # 로봇 articulation
    asset = env.scene[asset_cfg.name]

    # 루트 CoM world position: [x, y, z, qw, qx, qy, qz] 중 pos 부분만 사용
    # root_com_pose_w: (num_envs, 7)
    root_com_pos_w = asset.data.root_com_pos_w
    com_xy = root_com_pos_w[:, :2]  # (num_envs, 2)

    # 바퀴 링크들의 world pose: (num_envs, num_bodies, 7)
    body_poses = asset.data.body_link_pos_w
    wheel_ids = support_cfg.body_ids  # 두 바퀴의 body index들
    wheel_pos = body_poses[:, wheel_ids, :3]           # (num_envs, num_wheels, 3)
    support_xy = wheel_pos[:, :, :2].mean(dim=1)       # (num_envs, 2) 두 바퀴 중심

    diff_xy = com_xy - support_xy                      # (num_envs, 2)
    # XY 거리의 제곱합 -> L2 penalty
    return torch.sum(diff_xy * diff_xy, dim=-1)

def lin_speed_over_limit_l2(env, limit: float = 1.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    base_lin_vel = env.scene[asset_cfg.name].data.root_lin_vel_w[:, 0]  # x 방향 속도
    excess = torch.clamp(torch.abs(base_lin_vel) - limit, min=0.0)
    return excess * excess

def feet_distance(env: ManagerBasedRLEnv,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                  feet_links_name: list[str]=["foot_[RL]_Link"],
                  min_feet_distance: float = 0.1,
                  max_feet_distance: float = 1.0,)-> torch.Tensor:
    # Penalize base height away from target
    asset: Articulation = env.scene[asset_cfg.name]
    feet_links_idx = asset.find_bodies(feet_links_name)[0]
    feet_pos = asset.data.body_link_pos_w[:,feet_links_idx]
    # feet distance on x-y plane
    feet_distance = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=-1)
    reward = torch.clip(min_feet_distance - feet_distance, 0, 1)
    reward += torch.clip(feet_distance - max_feet_distance, 0, 1)
    return reward

def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

def stable_single_leg_stance(
    env: ManagerBasedRLEnv,
    # 로봇 자세(orientation)용
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # 접지 여부 확인용 (왼/오른 바퀴)
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=["left_wheel_link", "right_wheel_link"]
    ),
    contact_threshold: float = 5.0,   # N 정도, 상황에 맞게 조정
    ori_margin: float = 0.4,         # rad 정도. bad_orientation limit(0.7) 보다 훨씬 작게
) -> torch.Tensor:
    """한쪽 바퀴만 닿아 있고 + 로봇이 비교적 수평일 때 큰 보상을 주는 term."""
    # 로봇 / 센서 핸들
    robot = env.scene[asset_cfg.name]
    contact_sensor = env.scene[sensor_cfg.name]

    # 접촉 힘: (num_envs, B, 3)  -- B=2 (left/right wheel) 이라고 가정
    forces = contact_sensor.data.net_forces_w      # [N, 2, 3]
    # z 축(normal) 힘만 사용
    fz = forces[..., 2].abs()                      # [N, 2]

    # 접지 여부 (threshold 이상이면 contact로 간주)
    contact = (fz > contact_threshold).float()     # [N, 2]
    num_contacts = contact.sum(dim=-1)             # [N]

    # 정확히 한쪽만 닿아있는 경우만 1, 나머지는 0
    single_support = (num_contacts == 1.0).float() # [N]

    # 자세 안정성: flat_orientation_l2 결과(roll^2 + pitch^2 같은 것)를 가져다가 soft하게 스케일링
    # 이미 mdp.flat_orientation_l2(env, asset_cfg=SceneEntityCfg("robot")) 함수가 존재한다고 가정
    ori_error = flat_orientation_l2(env, asset_cfg=asset_cfg)  # [N]
    # ori_margin 근처까지는 거의 1, 그 이상 틀어지면 급격히 줄어드는 형태
    # (ori_error가 rad^2 스케일이라고 보면, ori_margin^2 로 나눠줌)
    safe_ori_score = torch.exp(-(ori_error / (ori_margin ** 2)))  # [N], 0~1

    # single_support 이면서 자세가 안정적일수록 큰 값
    reward = single_support * safe_ori_score

    return reward