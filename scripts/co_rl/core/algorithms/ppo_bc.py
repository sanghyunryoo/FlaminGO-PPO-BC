#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from scripts.co_rl.core.modules import ActorCritic
from scripts.co_rl.core.storage import BC_RolloutStorage

class PPO_BC:
    actor_critic: ActorCritic
    
    def __init__(
        self,
        actor_critic,
        teacher_actor_critic=None,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        est_learning_rate=1.0e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",

        bc_ckpt: str | None = None,
        bc_coef: float = 1.0,
        rl_coef: float = 1.0,

        annealing_factor: float = 0.001,
        low_rl_coef_ratio: float = 0.25,
        high_bc_coef_ratio: float = 10.0,

        train_joint_idx: list = None,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = BC_RolloutStorage.Transition()


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

        self.cfg = None

        self.teacher = teacher_actor_critic
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
                
        self.teacher = None
        self.bc_ckpt = bc_ckpt
        self.bc_coef = bc_coef
        self.rl_coef = rl_coef   

        self.annealing_factor = annealing_factor
        self.rl_coef_initial = rl_coef
        self.bc_coef_initial = bc_coef

        self.low_rl_coef = low_rl_coef_ratio * rl_coef
        self.high_bc_coef = high_bc_coef_ratio * bc_coef
        self.train_joint_idx = train_joint_idx

    def init_storage(self, cfg, num_envs, num_transitions_per_env, actor_obs_shape, teacher_obs_shape, critic_obs_shape, action_shape):
        self.update_late_cfg(cfg)
        self.storage = BC_RolloutStorage(
            cfg , num_envs, num_transitions_per_env, actor_obs_shape, teacher_obs_shape, critic_obs_shape, action_shape, self.device
        )
    
        # === Teacher 초기화 ===
        if self.bc_ckpt is not None and self.teacher is None:
            self._init_teacher()

    def _init_teacher(self):
        """
        ActorCritic 초기화 및 체크포인트 로드
        """
        from scripts.co_rl.core.modules import ActorCritic

        ckpt = torch.load(self.bc_ckpt, map_location=self.device)
        
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state_ac = ckpt["model_state_dict"]
            elif "model" in ckpt:
                state_ac = ckpt["model"]
            else:
                state_ac = ckpt
        else:
            state_ac = ckpt

        # === ActorCritic 구조 추론 ===
        actor_in_dim = state_ac["actor.0.weight"].shape[1]
        critic_in_dim = state_ac["critic.0.weight"].shape[1]
        actor_layers = [w.shape[0] for k, w in state_ac.items() if k.startswith("actor.") and "weight" in k]
        hidden_dims = actor_layers[:-1]
        num_actions_teacher = actor_layers[-1]

        print(f"[BC] Teacher AC: actor_in={actor_in_dim}, critic_in={critic_in_dim}, "
            f"hidden={hidden_dims}, action_dim={num_actions_teacher}")

        self.teacher = ActorCritic(
            num_actor_obs=actor_in_dim,
            num_critic_obs=critic_in_dim,
            num_actions=num_actions_teacher,
            actor_hidden_dims=hidden_dims,
            critic_hidden_dims=hidden_dims,
        ).to(self.device)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        missing, unexpected = self.teacher.load_state_dict(state_ac, strict=False)

        print(f"[BC] Teacher Load Done: {self.bc_ckpt}")
        if missing:
            print(f"     missing AC keys: {missing}")
        if unexpected:
            print(f"     unexpected AC keys: {unexpected}")

    def update_late_cfg(self, cfg):
        self.cfg = cfg

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, teacher_obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values

        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.teacher_observations = teacher_obs
        self.transition.critic_observations = critic_obs
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

    def update(self, expert_data=None, bc_coef=0.1):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_bc_loss = 0

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        for (
            obs_batch,
            teacher_obs_batch,
            critic_obs_batch,
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

            # -----------------------------
            # PPO 기본
            # -----------------------------
            self.actor_critic.act(obs_batch, dim=-1)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            dist_now  = self.actor_critic.dist(mu_batch[..., self.train_joint_idx],  sigma_batch[..., self.train_joint_idx])
            dist_old  = self.actor_critic.dist(old_mu_batch[..., self.train_joint_idx], old_sigma_batch[..., self.train_joint_idx])
            actions_wheel = actions_batch[..., self.train_joint_idx]

            actions_log_prob_w_now = dist_now.log_prob(actions_wheel).sum(dim=-1)
            actions_log_prob_w_old = dist_old.log_prob(actions_wheel).sum(dim=-1).detach()

            entropy_batch = dist_now.entropy().sum(dim=-1)
            
            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(dist_now.stddev / (dist_old.stddev + 1.0e-5) + 1.0e-8)
                        + (dist_old.variance + (dist_old.mean - dist_now.mean).pow(2))
                        / (2.0 * dist_now.variance + 1.0e-8)
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # ratio 및 surrogate loss
            ratio = torch.exp(actions_log_prob_w_now - actions_log_prob_w_old)
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # value loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # -----------------------------
            # Behavior Cloning Loss (teacher)
            bc_loss = torch.tensor(0.0, device=self.device)
            if self.teacher is not None and self.bc_coef > 0.0:
                # 1) teacher 쪽은 no_grad
                with torch.no_grad():
                    t_in  = teacher_obs_batch
                    mu_t = self.teacher.act_inference(t_in)[..., :6]   # teacher mean (no grad)

                # 2) student mean은 grad가 흘러야 함 → no_grad 밖에서
                mu_s = self.actor_critic.action_mean[..., :6]

                # 3) MSE BC loss
                bc_loss = ((mu_s - mu_t).pow(2).mean(dim=-1)).mean()
            # -----------------------------
            # 최종 Loss
            # -----------------------------
            rl_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            loss = self.rl_coef * rl_loss + self.bc_coef * bc_loss

            # gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_bc_loss += bc_loss.item()

        num_updates_extra = 0
        mean_extra_loss = 0

        # 결과 리턴
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        if num_updates_extra > 0:
            mean_extra_loss /= num_updates
        self.storage.clear()
        self.rl_coef = max(self.low_rl_coef, self.rl_coef * (1 - self.annealing_factor))  # 점점 줄어들게
        self.bc_coef = min(self.bc_coef * (1 + self.annealing_factor), self.high_bc_coef)  # 최대 2.5배까지 증가)
        return mean_value_loss, mean_extra_loss, mean_surrogate_loss, mean_bc_loss