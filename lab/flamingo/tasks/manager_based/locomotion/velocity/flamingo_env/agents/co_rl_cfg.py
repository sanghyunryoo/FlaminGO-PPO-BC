# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from scripts.co_rl.core.wrapper import (
    # EncoderCfg,
    CoRlPolicyRunnerCfg,
    CoRlPpoActorCriticCfg,
    CoRlPpoAlgorithmCfg,
)

######################################## [ PPO CONFIG] ########################################


@configclass
class FlamingoPPORunnerCfg(CoRlPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "FlamingoStand-v0"
    experiment_description = "test"
    empirical_normalization = False
    policy = CoRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = CoRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    # encoder = EncoderCfg(
    #     output_detach = True,
    #     num_output_dim = 3,
    #     hidden_dims = [256, 128],
    #     activation = "elu",
    #     orthogonal_init = False,
    # )

@configclass
class FlamingoRoughPPORunnerCfg_Stand_Drive(FlamingoPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 15000
        self.experiment_name = "Flamingo_Rough_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.class_name = 'PPO'

@configclass
class FlamingoRoughPPOBCRunnerCfg_Stand_Drive(FlamingoPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 15000
        self.experiment_name = "Flamingo_Rough_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.class_name = 'PPO_BC'
        self.algorithm.bc_coef = 1.0
        self.algorithm.rl_coef = 0.5
        self.algorithm.low_rl_coef_ratio = 0.15
        self.algorithm.high_bc_coef_ratio = 2.5
        self.algorithm.annealing_factor = 0.001
        self.algorithm.train_joint_idx = [0, 1, 2, 3, 4, 5, 6, 7]

@configclass
class FlamingoRoughPPORunnerCfg_Stand_Drive_Play(FlamingoPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.max_iterations = 15000
        self.experiment_name = "Flamingo_Rough_Stand_Drive"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.train_joint_idx = [0, 1, 2, 3, 4, 5, 6, 7]

###############################################################################################

