# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCriticRMA
from rsl_rl.storage import RolloutStorage
import wandb
from rsl_rl.utils import unpad_trajectories


class RMS(object):
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape, device=device)
        self.S = torch.ones(shape, device=device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs + (delta**2) * self.n * bs / (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S

class PPO:
    actor_critic: ActorCriticRMA
    def __init__(self,
                 actor_critic,
                 estimator,
                 estimator_paras,
                 depth_encoder,
                 depth_encoder_paras,
                 depth_actor,
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
                 device='cpu',
                 dagger_update_freq=20,
                 priv_reg_coef_schedual = [0, 0, 0],
                 **kwargs
                 ):

        
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

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

        # Adaptation
        self.hist_encoder_optimizer = optim.Adam(self.actor_critic.actor.history_encoder.parameters(), lr=learning_rate)
        self.priv_reg_coef_schedual = priv_reg_coef_schedual
        self.counter = 0

        # Estimator
        self.estimator = estimator
        self.priv_states_dim = estimator_paras["priv_states_dim"]
        self.num_prop = estimator_paras["num_prop"]
        self.num_scan = estimator_paras["num_scan"]
        self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=estimator_paras["learning_rate"])
        self.train_with_estimated_states = estimator_paras["train_with_estimated_states"]

        # Depth encoder
        self.if_depth = depth_encoder != None
        if self.if_depth:
            self.depth_encoder = depth_encoder
            self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(), lr=depth_encoder_paras["learning_rate"])
            self.depth_encoder_paras = depth_encoder_paras
            self.depth_actor = depth_actor
            self.depth_actor_optimizer = optim.Adam([*self.depth_actor.parameters(), *self.depth_encoder.parameters()], lr=depth_encoder_paras["learning_rate"])

        # GradNorm
        if self.if_depth:
            self.gradnorm_alpha = 0.5
            self.task_weights = torch.nn.Parameter(torch.tensor([1.0, 1.0], device=self.device)) # initialize task weights for 2 tasks
            self.task_weight_optimizer = torch.optim.Adam([self.task_weights], lr=1e-3)
            self.initial_losses = None
            self.gradnorm_shared_params = (
                        list(self.depth_encoder.g_a.parameters()) +
                        list(self.depth_encoder.g_b.parameters())
                    )
            self.ema_decay = 0.99
            self.loss_ema = {
                "recon": None,
                "action": None
            }
            # self.gradnorm_shared_params = [next(self.depth_encoder.parameters())]
            # self.gradnorm_shared_params = list(self.depth_encoder.parameters())

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape,  critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, info, hist_encoding=False):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values, use proprio to compute estimated priv_states then actions, but store true priv_states
        if self.train_with_estimated_states:
            obs_est = obs.clone()
            priv_states_estimated = self.estimator(obs_est[:, :self.num_prop])
            obs_est[:, self.num_prop+self.num_scan:self.num_prop+self.num_scan+self.priv_states_dim] = priv_states_estimated
            self.transition.actions = self.actor_critic.act(obs_est, hist_encoding).detach()
        else:
            self.transition.actions = self.actor_critic.act(obs, hist_encoding).detach()

        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        rewards_total = rewards.clone()

        self.transition.rewards = rewards_total.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

        return rewards_total
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimator_loss = 0
        mean_discriminator_loss = 0
        mean_discriminator_acc = 0
        mean_priv_reg_loss = 0
        orig_hidden = None

        if self.actor_critic.is_recurrent:
            orig_hidden = self.actor_critic.get_hidden_states()
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        # update over mini-batches
        for (
            obs_batch, critic_obs_batch, actions_batch,
            target_values_batch, advantages_batch, returns_batch,
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
            hid_states_batch, masks_batch
        ) in generator:

            if self.actor_critic.is_recurrent:
                # === RECURRENT PATH ===
                # obs_batch: (T, B, F), masks_batch: (T, B) or (T, B, 1)
                # masks_batch: 1 = valid data, 0 = padding

                # init hidden state
                hid_a, hid_c = hid_states_batch
                if hid_a is not None:
                    if hid_a.dim() == 3 and hid_a.size(0) == 1:
                        hid_a = hid_a.squeeze(0)
                    self.actor_critic.actor.hidden_state = hid_a
                else:
                    self.actor_critic.actor.hidden_state = None

                T, B, F = obs_batch.shape

                log_probs_list = []
                entropies_list = []
                values_list = []
                mus_list = []

                for t in range(T):
                    # forward actor
                    mu_t = self.actor_critic.actor(obs_batch[t], hist_encoding=False)
                    std = self.actor_critic.std
                    dist = torch.distributions.Normal(mu_t, std)

                    log_prob_t = dist.log_prob(actions_batch[t]).sum(dim=-1)  # (B,)
                    entropy_t = dist.entropy().sum(dim=-1)  # (B,)

                    # forward critic
                    value_t = self.actor_critic.critic(critic_obs_batch[t]).squeeze(-1)  # (B,)

                    log_probs_list.append(log_prob_t)
                    entropies_list.append(entropy_t)
                    values_list.append(value_t)
                    mus_list.append(mu_t)

                # stack: (T, B) or (T, B, A)
                actions_log_prob_batch_new = torch.stack(log_probs_list, dim=0)  # (T, B)
                entropy_batch = torch.stack(entropies_list, dim=0)  # (T, B)
                value_batch = torch.stack(values_list, dim=0)  # (T, B)
                mu_batch = torch.stack(mus_list, dim=0)  # (T, B, A)
                sigma_batch = self.actor_critic.std.expand_as(mu_batch)

                # masks for loss computation
                # masks_batch: (T, B) or (T, B, 1), 1=valid, 0=padding
                if masks_batch is not None:
                    masks_flat = masks_batch.reshape(-1).float()  # (T*B,)
                    valid_count = masks_flat.sum().clamp(min=1)
                else:
                    masks_flat = torch.ones(T * B, device=obs_batch.device)
                    valid_count = T * B

                # flatten for loss
                old_actions_log_prob_flat = old_actions_log_prob_batch.reshape(-1)  # (T*B,)
                advantages_flat = advantages_batch.reshape(-1)  # (T*B,)
                returns_flat = returns_batch.reshape(-1)  # (T*B,)
                target_values_flat = target_values_batch.reshape(-1)  # (T*B,)
                old_mu_flat = old_mu_batch.reshape(-1, old_mu_batch.shape[-1])  # (T*B, A)
                old_sigma_flat = old_sigma_batch.reshape(-1, old_sigma_batch.shape[-1])  # (T*B, A)

                actions_log_prob_flat = actions_log_prob_batch_new.reshape(-1)  # (T*B,)
                entropy_flat = entropy_batch.reshape(-1)  # (T*B,)
                value_flat = value_batch.reshape(-1)  # (T*B,)
                mu_flat = mu_batch.reshape(-1, mu_batch.shape[-1])  # (T*B, A)
                sigma_flat = sigma_batch.reshape(-1, sigma_batch.shape[-1])  # (T*B, A)

                # for priv_reg
                obs_flat = obs_batch.reshape(-1, F)  # (T*B, F)

            else:
                # === NON-RECURRENT PATH ===
                # obs_batch: (B, F), no time dimension

                # forward actor
                mu_batch = self.actor_critic.actor(obs_batch, hist_encoding=False)
                std = self.actor_critic.std
                dist = torch.distributions.Normal(mu_batch, std)

                actions_log_prob_flat = dist.log_prob(actions_batch).sum(dim=-1)  # (B,)
                entropy_flat = dist.entropy().sum(dim=-1)  # (B,)

                # forward critic
                value_flat = self.actor_critic.critic(critic_obs_batch).squeeze(-1)  # (B,)

                mu_flat = mu_batch
                sigma_flat = std.expand_as(mu_batch)

                # flatten references (already flat for non-recurrent)
                old_actions_log_prob_flat = old_actions_log_prob_batch.reshape(-1)
                advantages_flat = advantages_batch.reshape(-1)
                returns_flat = returns_batch.reshape(-1)
                target_values_flat = target_values_batch.reshape(-1)
                old_mu_flat = old_mu_batch
                old_sigma_flat = old_sigma_batch

                obs_flat = obs_batch
                masks_flat = torch.ones(obs_batch.shape[0], device=obs_batch.device)
                valid_count = obs_batch.shape[0]

            # PPO Surrogate Loss
            ratio = torch.exp(actions_log_prob_flat - old_actions_log_prob_flat)
            surr1 = ratio * advantages_flat
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_flat
            surrogate_loss = -torch.sum(torch.min(surr1, surr2) * masks_flat) / valid_count

            # Value Loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_flat + (value_flat - target_values_flat).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_flat - returns_flat).pow(2)
                value_losses_clipped = (value_clipped - returns_flat).pow(2)
                value_loss = torch.sum(torch.max(value_losses, value_losses_clipped) * masks_flat) / valid_count
            else:
                value_loss = torch.sum((value_flat - returns_flat).pow(2) * masks_flat) / valid_count

            # Entropy Loss
            entropy_loss = -torch.sum(entropy_flat * masks_flat) / valid_count

            # Priv Reg Loss (Adaptation)
            valid_mask = masks_flat > 0
            obs_valid = obs_flat[valid_mask]

            priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_valid)
            with torch.inference_mode():
                hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_valid)
            priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()

            priv_reg_stage = min(
                max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1
            )
            priv_reg_coef = (
                priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0])
                + self.priv_reg_coef_schedual[0]
            )

            # Estimator Loss
            priv_states_predicted = self.estimator(obs_valid[:, :self.num_prop])
            gt_priv = obs_valid[
                :, self.num_prop + self.num_scan : self.num_prop + self.num_scan + self.priv_states_dim
            ]
            estimator_loss = (priv_states_predicted - gt_priv).pow(2).mean()

            self.estimator_optimizer.zero_grad()
            estimator_loss.backward()
            nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
            self.estimator_optimizer.step()

            # KL Adaptive Learning Rate
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = (
                        torch.log(sigma_flat / old_sigma_flat + 1e-5)
                        + (old_sigma_flat.pow(2) + (old_mu_flat - mu_flat).pow(2)) / (2.0 * sigma_flat.pow(2))
                        - 0.5
                    ).sum(dim=-1)
                    kl_mean = torch.sum(kl * masks_flat) / valid_count

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for pg in self.optimizer.param_groups:
                        pg['lr'] = self.learning_rate

            # Total Loss & Backward
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                + self.entropy_coef * entropy_loss
                + priv_reg_coef * priv_reg_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Accumulate metrics
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimator_loss += estimator_loss.item()
            mean_priv_reg_loss += priv_reg_loss.item()
            mean_discriminator_loss += 0
            mean_discriminator_acc += 0

        # Finalize
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimator_loss /= num_updates
        mean_priv_reg_loss /= num_updates
        mean_discriminator_loss /= num_updates
        mean_discriminator_acc /= num_updates

        # Restore original hidden state
        if self.actor_critic.is_recurrent:
            if orig_hidden is None or orig_hidden == (None, None):
                self.actor_critic.actor.hidden_state = None
            else:
                hid = orig_hidden[0]  # orig_hidden is (h_actor, h_critic)
                if hid is None:
                    self.actor_critic.actor.hidden_state = None
                elif hid.dim() == 3 and hid.size(0) == 1:
                    self.actor_critic.actor.hidden_state = hid.squeeze(0)  # (1,B,H) -> (B,H)
                else:
                    self.actor_critic.actor.hidden_state = hid  # already (B,H)
        self.storage.clear()
        self.update_counter()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_estimator_loss,
            mean_discriminator_loss,
            mean_discriminator_acc,
            mean_priv_reg_loss,
            priv_reg_coef,
        )

    def update_dagger(self):
        mean_hist_latent_loss = 0
        orig_hidden = None

        if self.actor_critic.is_recurrent:
            orig_hidden = self.actor_critic.get_hidden_states()
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        # update over mini-batches
        for (
            obs_batch, critic_obs_batch, actions_batch,
            target_values_batch, advantages_batch, returns_batch,
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
            hid_states_batch, masks_batch
        ) in generator:

            if self.actor_critic.is_recurrent:
                # === RECURRENT PATH ===
                # obs_batch: (T, B, F), masks_batch: (T, B) or (T, B, 1)
                # masks_batch: 1 = valid, 0 = padding
                
                hid_a, hid_c = hid_states_batch
                if hid_a is not None:
                    if hid_a.dim() == 3 and hid_a.size(0) == 1:
                        hid_a = hid_a.squeeze(0)
                    self.actor_critic.actor.hidden_state = hid_a
                else:
                    self.actor_critic.actor.hidden_state = None
                
                T, B, F = obs_batch.shape

                with torch.inference_mode():
                    for t in range(T):
                        _ = self.actor_critic.actor(obs_batch[t], hist_encoding=True)
                
                # flatten obs for latent inference
                if masks_batch is not None:
                    masks_flat = masks_batch.reshape(-1).float()
                    valid_mask = masks_flat > 0
                else:
                    valid_mask = torch.ones(T * B, dtype=torch.bool, device=obs_batch.device)
                
                obs_flat = obs_batch.reshape(-1, F)
                obs_valid = obs_flat[valid_mask]
            else:
                # === NON-RECURRENT PATH ===
                # obs_batch: (B, F)
                obs_valid = obs_batch
            
            # Adaptation module update (hist_encoder distillation)
            with torch.inference_mode():
                priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_valid)
            hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_valid)
            hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
            
            self.hist_encoder_optimizer.zero_grad()
            hist_latent_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.actor.history_encoder.parameters(), self.max_grad_norm)
            self.hist_encoder_optimizer.step()
            
            mean_hist_latent_loss += hist_latent_loss.item()

        # Finalize
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        
        # Restore original hidden state
        if self.actor_critic.is_recurrent:
            if orig_hidden is None or orig_hidden == (None, None):
                self.actor_critic.actor.hidden_state = None
            else:
                hid = orig_hidden[0]  # orig_hidden is (h_actor, h_critic)
                if hid is None:
                    self.actor_critic.actor.hidden_state = None
                elif hid.dim() == 3 and hid.size(0) == 1:
                    self.actor_critic.actor.hidden_state = hid.squeeze(0)
                else:
                    self.actor_critic.actor.hidden_state = hid

        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss

    def update_depth_encoder(self, depth_latent_batch, scandots_latent_batch):
        # Depth encoder ditillation
        if self.if_depth:
            # TODO: needs to save hidden states
            depth_encoder_loss = (scandots_latent_batch.detach() - depth_latent_batch).norm(p=2, dim=1).mean()

            self.depth_encoder_optimizer.zero_grad()
            depth_encoder_loss.backward()
            nn.utils.clip_grad_norm_(self.depth_encoder.parameters(), self.max_grad_norm)
            self.depth_encoder_optimizer.step()
            return depth_encoder_loss.item()
        
    def update_belief_actor(self, scandots_batch, actions_student_batch=None, actions_teacher_batch=None, recon_batch=None):
        if self.if_depth:
            device = self.device
            # Reconstruction loss
            if recon_batch is not None and scandots_batch is not None:
                recon_loss_raw = nn.functional.mse_loss(
                    recon_batch,
                    scandots_batch.detach()
                )
            else:
                recon_loss_raw = torch.tensor(0.0, device=device)
            # Action imitation loss
            if actions_student_batch is not None and actions_teacher_batch is not None:
                l2_per_sample = torch.norm(
                    actions_student_batch - actions_teacher_batch.detach(),
                    p=2,
                    dim=-1
                )
                action_loss_raw = l2_per_sample.mean()
            else:
                action_loss_raw = torch.tensor(0.0, device=device)
            # Task loss vector L = [L1, L2]
            self._update_ema("recon", recon_loss_raw)
            self._update_ema("action", action_loss_raw)
            recon_loss = recon_loss_raw / (self.loss_ema["recon"] + 1e-8)
            action_loss = action_loss_raw / (self.loss_ema["action"] + 1e-8)
            L = torch.stack([recon_loss, action_loss])    # shape [2]
            # Initialize reference losses (GradNorm)
            if self.initial_losses is None:
                # store a detached copy (no grad)
                self.initial_losses = L.detach().clone()

            # GradNorm: first compute G_i (gradient norms) with create_graph=True
            # Expectation: self.gradnorm_shared_params is an iterable of tensors (shared params)
            shared_params = self.gradnorm_shared_params
            # Ensure shared_params is a tuple/list as required by autograd.grad
            # Note: autograd.grad returns a tuple of grads corresponding to inputs
            G_list = []
            for i in range(L.numel()):  # two tasks
                # compute grads of (w_i * L_i) wrt shared params; keep graph so we can backprop through these norms
                # allow_unused=True in case some params do not contribute to this task (safer)
                grads = torch.autograd.grad(
                    outputs=(self.task_weights[i] * L[i]),
                    inputs=shared_params,
                    retain_graph=True,    # we will need the graph again later (safe to keep until shared update)
                    create_graph=True,    # IMPORTANT: allow gradients of G_i w.r.t task_weights
                    allow_unused=True
                )

                # grads may be tuple of tensors (or single tensor). compute flattened norm:
                # replace None grads with zeros of appropriate shape
                flat_grads = []
                for g, p in zip(grads, shared_params):
                    if g is None:
                        # create zero tensor with same shape as param p, on same device
                        flat_grads.append(torch.zeros_like(p).view(-1))
                    else:
                        flat_grads.append(g.contiguous().view(-1))
                if len(flat_grads) == 0:
                    # defensive: if no shared params (shouldn't happen), treat norm as zero
                    g_norm = torch.tensor(0.0, device=device)
                else:
                    all_flat = torch.cat(flat_grads)
                    g_norm = torch.norm(all_flat, p=2)
                G_list.append(g_norm)

            G = torch.stack(G_list)  # shape [K]
            # detach G_avg used as scalar baseline for target; but keep G itself attached for gradnorm_loss backward
            G_avg = G.mean().detach()

            # Compute target gradient magnitudes target_G
            # loss_ratio = L(t) / L(0)
            loss_ratio = L.detach() / (self.initial_losses + 1e-12)
            # normalized rates r_i
            r_i = loss_ratio / (loss_ratio.mean() + 1e-12)
            # target gradient magnitudes
            target_G = G_avg * (r_i ** self.gradnorm_alpha)

            # GradNorm loss and update task weights
            # L_grad = sum_i |G_i - target_G_i|
            gradnorm_loss = torch.abs(G - target_G).sum()

            # update task weights (these should be optimized by self.task_weight_optimizer)
            self.task_weight_optimizer.zero_grad()
            gradnorm_loss.backward(retain_graph=True)
            self.task_weight_optimizer.step()

            # Prevent weights from diverging: normalize to sum K (K=number of tasks)
            with torch.no_grad():
                w = self.task_weights
                K = float(L.numel())
                # ensure positivity if desired (paper doesn't strictly enforce positivity but common to keep >0)
                # Here we keep raw values but normalize their sum to K.
                w_min, w_max = 0.5, 2.0   # clip
                w[:] = torch.clamp(w, w_min, w_max)
                w[:] = K * w / (w.sum() + 1e-12)

            # Finally update encoder + actor using the (updated) task weights
            weighted_loss = (self.task_weights * L).sum()
            self.depth_actor_optimizer.zero_grad()
            weighted_loss.backward()
            # Clip gradients for encoder + actor
            nn.utils.clip_grad_norm_(list(self.depth_encoder.parameters()) + list(self.depth_actor.parameters()), self.max_grad_norm)
            self.depth_actor_optimizer.step()

            # return scalar numbers
            return recon_loss.item(), action_loss.item(), weighted_loss.item(), self.task_weights.detach().cpu().clone(), self.loss_ema
        
    def _update_ema(self, name, value):
        v = value.detach()
        if self.loss_ema[name] is None:
            self.loss_ema[name] = v.clone()
        else:
            self.loss_ema[name] = (
                self.ema_decay * self.loss_ema[name] +
                (1 - self.ema_decay) * v
            )

    def update_belief_actor_ema(self, scandots_batch, actions_student_batch=None, actions_teacher_batch=None, recon_batch=None):
        if self.if_depth:
            device = self.device
            # Reconstruction loss
            if recon_batch is not None and scandots_batch is not None:
                recon_loss_raw = nn.functional.mse_loss(
                    recon_batch,
                    scandots_batch.detach()
                )
            else:
                recon_loss_raw = torch.tensor(0.0, device=device)
            # Action imitation loss
            if actions_student_batch is not None and actions_teacher_batch is not None:
                l2_per_sample = torch.norm(
                    actions_student_batch - actions_teacher_batch.detach(),
                    p=2,
                    dim=-1
                )
                action_loss_raw = l2_per_sample.mean()
            else:
                action_loss_raw = torch.tensor(0.0, device=device)

            # Update EMA
            self._update_ema("recon", recon_loss_raw)
            self._update_ema("action", action_loss_raw)
            # Normalized losses
            recon_loss = recon_loss_raw / (self.loss_ema["recon"] + 1e-8)
            action_loss = action_loss_raw / (self.loss_ema["action"] + 1e-8)
            # final loss
            weighted_loss = recon_loss + action_loss

            # optimize
            self.depth_actor_optimizer.zero_grad()
            weighted_loss.backward()

            nn.utils.clip_grad_norm_(
                list(self.depth_encoder.parameters()) +
                list(self.depth_actor.parameters()),
                self.max_grad_norm
            )

            self.depth_actor_optimizer.step()

            return (
                recon_loss.item(),
                action_loss.item(),
                weighted_loss.item(),
                None,                # no task weights
                self.loss_ema.copy() # return your EMA
            )

    def update_depth_actor(self, actions_student_batch, actions_teacher_batch, yaw_student_batch, yaw_teacher_batch):
        if self.if_depth:
            depth_actor_loss = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1).mean()
            yaw_loss = (yaw_teacher_batch.detach() - yaw_student_batch).norm(p=2, dim=1).mean()

            # no yaw_loss
            # loss = depth_actor_loss + yaw_loss
            loss = depth_actor_loss

            self.depth_actor_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.depth_actor.parameters(), self.max_grad_norm)
            self.depth_actor_optimizer.step()
            return depth_actor_loss.item(), yaw_loss.item()
    
    def update_depth_both(self, depth_latent_batch, scandots_latent_batch, actions_student_batch, actions_teacher_batch):
        if self.if_depth:
            depth_encoder_loss = (scandots_latent_batch.detach() - depth_latent_batch).norm(p=2, dim=1).mean()
            depth_actor_loss = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1).mean()

            depth_loss = depth_encoder_loss + depth_actor_loss

            self.depth_actor_optimizer.zero_grad()
            depth_loss.backward()
            nn.utils.clip_grad_norm_([*self.depth_actor.parameters(), *self.depth_encoder.parameters()], self.max_grad_norm)
            self.depth_actor_optimizer.step()
            return depth_encoder_loss.item(), depth_actor_loss.item()
    
    def update_counter(self):
        self.counter += 1
    
    def compute_apt_reward(self, source, target):

        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        # sim_matrix = torch.norm(source[:, None, ::2].view(b1, 1, -1) - target[None, :, ::2].view(1, b2, -1), dim=-1, p=2)
        # sim_matrix = torch.norm(source[:, None, :2].view(b1, 1, -1) - target[None, :, :2].view(1, b2, -1), dim=-1, p=2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)

        reward, _ = sim_matrix.topk(self.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)

        if not self.knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            if self.rms:
                moving_mean, moving_std = self.disc_state_rms(reward)
                reward = reward / moving_std
            reward = torch.clamp(reward - self.knn_clip, 0)  # (b1, )
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            if self.rms:
                moving_mean, moving_std = self.disc_state_rms(reward)
                reward = reward / moving_std
            reward = torch.clamp(reward - self.knn_clip, 0)
            reward = reward.reshape((b1, self.knn_k))  # (b1, k)
            reward = reward.mean(dim=1)  # (b1,)
        reward = torch.log(reward + 1.0)
        return reward