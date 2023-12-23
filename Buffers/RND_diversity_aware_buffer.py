import argparse
import copy
import math
import random
import statistics
import time
from distutils.util import strtobool
from typing import Dict, Generator, NamedTuple, Optional, Union
import gymnasium as gym
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import BaseBuffer, RolloutBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv
import sys
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union
from torch.distributions.categorical import Categorical
import numpy as np
import torch as th
from gymnasium import spaces
import heapq
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

from exploration_on_policy.Networks.ByolEncoder import BYOLEncoder
from exploration_on_policy.Networks.ACNetwork import ACNetwork
from exploration_on_policy.Networks.BufferEncoder import RNDBufferEncoder
from exploration_on_policy.Networks.UcbCountMultipleNets import UcbNets
from exploration_on_policy.utils.Buffer_add_helper import Buffer_add_helper
from exploration_on_policy.utils.Test_util import get_test_arg


class Traj:
    def __init__(self, nd_value: int,
                 obs_idx,
                 dependent_idx: int,
                 rnd_v: int
                 ):
        self.nd_value = nd_value
        self.obs_idx = obs_idx
        self.dependent_idx = dependent_idx
        self.rnd_v = rnd_v

    def getKey(self):
        return self.nd_value

    def __lt__(self, other):
        return self.nd_value < other.nd_value


class RNDReplayBuffer:
    def __init__(
            self,
            args,
            buffer_size: int,
            action_num: int,
            ema_beta: float,
            dim_x: int,
            dim_y: int,
            n_envs: int,
            initial_traj_len_times: int,
            roll_out_len: int,
            emb_dim: int,
            encoder_learning_rate: float = 0.001,
            store_device: Union[th.device, str] = "cpu",
            cuda_device="cuda:0",
            initial_encoder_train_epoches: int = 10,
    ):
        assert buffer_size > n_envs, "buffer_size > n_envs"
        self.actions_num = action_num
        self.store_device = store_device
        self.cuda_device = cuda_device
        self.initial_encoder_train_epoches = initial_encoder_train_epoches
        self.args = args
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.traj_len_times = initial_traj_len_times
        traj_len = initial_traj_len_times * roll_out_len
        self.roll_out_len = roll_out_len
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.traj_len = traj_len

        # Once the buffer is full, it remains full forever
        self.buffer_ah = Buffer_add_helper(self.traj_len_times, roll_out_len, buffer_size, n_envs)

        self.observations = torch.zeros((buffer_size, traj_len, dim_x, dim_y), device="cpu")
        self.log_probs = torch.zeros((buffer_size, traj_len), device="cpu")
        self.actions = torch.zeros((buffer_size, traj_len), dtype=torch.long, device="cpu")
        self.dones = torch.zeros((buffer_size, traj_len), dtype=torch.long, device="cpu")

        self.frontier = torch.zeros((buffer_size, dim_x, dim_y), device="cpu")
        self.frontier_idx = torch.zeros((buffer_size,), dtype=torch.long, device="cpu")
        # Under the assumption that RND novelty values only decrease,
        # we only recalculate the novelty_diversity value for the frontier state, to save computation
        self.f_novelties = torch.zeros((buffer_size,), dtype=torch.float, device="cpu")
        self.f_novelty_diversity = torch.zeros((buffer_size,), dtype=torch.float,
                                               device="cpu")  # only care the novelty of THE one frontier state
        self.dependency_indices = torch.zeros((buffer_size,), dtype=torch.long, device="cpu")
        # self.rewards = torch.zeros((buffer_size, traj_len), device="cpu")

        # min_heap
        # self.min_heap = None
        # self.min_heap = [Traj(-1, i) for i in range(buffer_size)]
        # heapq.heapify(self.min_heap)

        # cache
        self.cache = torch.zeros((initial_traj_len_times, roll_out_len, n_envs, dim_x, dim_y), dtype=torch.float,
                                 device="cpu")
        self.cache_actions = torch.zeros((initial_traj_len_times, roll_out_len, n_envs,), dtype=torch.long,
                                         device="cpu")
        self.cache_dones = torch.zeros((initial_traj_len_times, roll_out_len, n_envs,), dtype=torch.long, device="cpu")
        self.cache_log_probs = torch.zeros((initial_traj_len_times, roll_out_len, n_envs,), dtype=torch.float,
                                           device="cpu")
        self.cache_top = 0

        # encoder
        self.b_encoder = RNDBufferEncoder(emb_dim, ema_beta, action_num,
                                          encoder_learning_rate, device=self.cuda_device)

        if psutil is not None:
            total_memory_usage = sys.getsizeof(self.observations) + sys.getsizeof(self.actions) + sys.getsizeof(
                self.f_novelties) \
                                 + sys.getsizeof(self.frontier) + sys.getsizeof(self.frontier_idx) + sys.getsizeof(
                self.log_probs) \
                                 + sys.getsizeof(self.f_novelty_diversity) + sys.getsizeof(self.dones) + sys.getsizeof(
                self.dependency_indices) \
                                 + sys.getsizeof(self.cache)

            # Convert to GB
            total_memory_usage /= 1024 ** 3
            mem_available /= 1024 ** 3
            print(f"total_memory_usage is {total_memory_usage:.2f}GB, while mem_available is {mem_available:.2f}GB")

    # @torch.no_grad()
    # def update_encoder_target(self):
    #     self.b_encoder.encoder_ema.update_moving_average()

    def train_encoder(self,
                      states: torch.Tensor,  # (B, rollout, dim x, dim y)
                      actions: torch.Tensor,  # (B,rollout,
                      # done: torch.Tensor,
                      intrinsic_rewards: torch.Tensor = None,  # sample based on this
                      minibatch_num: int = 60,
                      epochs: int = 1,
                      ):
        self.b_encoder.train_encoder(states, actions, intrinsic_rewards, minibatch_num, epochs)

    def add(
            self,
            obs: torch.Tensor,  # (rollout len, envs, dim x, dim y)
            rnd_values: torch.Tensor,  # (rollout len, envs) calcualted during the training of RND
            rnd_values_min: float,
            actions: torch.Tensor,  # (rollout len, envs,
            # reward: torch.Tensor,  # (rollout len, envs,
            done: torch.Tensor,  # (rollout len, envs,
            log_probs: torch.Tensor,  # (rollout len, envs)
            rnd_module: UcbNets,
            policy_net: nn.Module = None,
            policy_net_optimizer=None,
    ) -> None:
        if not self.buffer_ah.die:
            if self.buffer_ah.less_than_full():
                # the buffer is not yet full, so just fill in new trajectories
                add_envs, q, w, e, r = self.buffer_ah.get_indices_and_update(obs.shape[1])
                self.observations[q:w, e:r, :, :] = obs[:, 0:add_envs, :, :].transpose(0, 1).cpu()
                self.actions[q:w, e:r] = actions[:, 0:add_envs].t().cpu()
                self.dones[q:w, e:r] = done[:, 0:add_envs].t().cpu()
                self.log_probs[q:w, e:r] = log_probs[:, 0:add_envs].t().cpu()
                if self.buffer_ah.no_add_cache():
                    return

            # update cache
            self.cache[self.cache_top] = obs.detach().cpu()
            self.cache_dones[self.cache_top] = done.detach().cpu()
            self.cache_actions[self.cache_top] = actions.detach().cpu()
            self.cache_log_probs[self.cache_top] = log_probs.detach().cpu()
            self.cache_top = (self.cache_top + 1) % self.traj_len_times

            if self.buffer_ah.adding_cache():
                return

            if self.buffer_ah.is_full():
                self.buffer_ah.die = True
                # train encoder
                print("buffer full, train encoder for the first time")
                b_inds = np.arange(self.buffer_size)
                for ep in range(self.initial_encoder_train_epoches):
                    np.random.shuffle(b_inds)
                    for idx in range(self.buffer_size):
                        self.train_encoder(self.observations[b_inds[idx]].unsqueeze(0),
                                           self.actions[b_inds[idx]].unsqueeze(0))

                print("buffer full, calculating the frontier info")
                # calculate all the info about the buffer
                # use the one with the largetst
                emb = None
                for i, obv_i in enumerate(self.observations):
                    rnd_v = rnd_module(obv_i).mean(1)  # (traj len,)
                    r_max, r_index = torch.max(rnd_v, dim=0)
                    self.frontier[i] = obv_i[r_index].cpu().detach().clone()
                    self.frontier_idx[i] = r_index
                    self.f_novelties[i] = r_max
                    emb = self.b_encoder(obv_i.unsqueeze(1)).mean(0) if emb is None else self.b_encoder(
                        obv_i.unsqueeze(1)).mean(0) + emb
                emb /= self.buffer_size
                self.b_encoder.center = emb.to(self.b_encoder.device)
                diversities = self.b_encoder.get_diversity(self.frontier,
                                                           self.frontier,
                                                           tau=2,
                                                           )  # ( frontier size, frontier size)
                diversities.fill_diagonal_(float('inf'))
                diversities, d_indices = torch.min(diversities, dim=1)  # (frontier size,), (frontier size,)
                self.f_novelty_diversity = (
                        diversities.cpu() * self.f_novelties).cpu().detach().clone()  # (frontier size,)
                self.dependency_indices = d_indices.to(torch.long).cpu().detach().clone()  # (frontier size,)

                # calculate the heap
                # self.min_heap = [Traj(nd_val.item(), i,self.dependency_indices[i],self.f_novelties[i]) for i, nd_val in enumerate(self.f_novelty_diversity)]
                # heapq.heapify(self.min_heap)

        # add 时候整块加进来，
        # 缓存前n块。
        # 先判断是否准入，如果最大的novelty足够了，可以就加入。
        rnd_values[rnd_values < rnd_values_min] = 0.0
        # train encoder first
        self.train_encoder(obs.transpose(0, 1), actions.transpose(0, 1))  # add other parameters

        # TODO -- add length shrinkage, within-in episode diversity
        roll_len, envs, dim_x, dim_y = obs.shape
        f_rnd_vals, f_indices = torch.max(rnd_values, dim=0)  # (envs,), (envs,)
        f_indices_ = f_indices + self.roll_out_len * torch.arange(envs, device=f_indices.get_device())
        diversities = self.b_encoder.get_diversity(obs.reshape((-1, dim_x, dim_y))[f_indices_, :, :],
                                                   self.frontier)  # (n_envs, frontier size)
        diversities, depen_idx = torch.min(diversities, dim=1)  # (n_envs,), (n_envs,)
        # print(diversities.mean(), "diversities mean")
        novelties_diversities = diversities * rnd_values.reshape(-1)[f_indices_]  # (n_envs,)

        # heaps for comparing
        f_indices = torch.cat([f_indices.unsqueeze(1), torch.arange(envs, device=f_indices.get_device()).unsqueeze(1)],
                              dim=1)  # (n_envs,2)
        f_heap = [Traj(nd_value=nd, obs_idx=f_indices[idx, :], dependent_idx=depen_idx[idx], rnd_v=f_rnd_vals[idx]) for
                  idx, nd in enumerate(novelties_diversities)]  # obs_idx is [ frontier idx, env idx]
        heapq.heapify(f_heap)

        all_heap = [
            Traj(nd_value=nd, obs_idx=idx, dependent_idx=self.dependency_indices[idx], rnd_v=self.f_novelties[idx]) for
            idx, nd in enumerate(self.f_novelty_diversity)]
        heapq.heapify(all_heap)
        n_small = heapq.nsmallest(self.n_envs, all_heap, key=lambda pp: pp.nd_value)  # (n_envs,)
        heapq.heapify(n_small)

        add_li, delete_li = [], []
        while len(f_heap) > 0 and len(n_small) > 0:
            if f_heap[0].getKey() > n_small[0].getKey():
                f_traj = heapq.heappop(f_heap)
                add_li.append(f_traj)
                old_traj = heapq.heappop(all_heap)
                delete_li.append(old_traj)

                heapq.heappush(all_heap, Traj(nd_value=f_traj.nd_value,
                                              obs_idx=old_traj.obs_idx, dependent_idx=f_traj.dependent_idx,
                                              rnd_v=f_traj.rnd_v))  #
                # replace the old traj with a new one in the heap TODO 其实这一步引发的变化没有考虑
            else:
                _ = heapq.heappop(f_heap)
        del all_heap

        buffer_idx = torch.tensor([traj.obs_idx for traj in delete_li], dtype=torch.long)  # (add num,)

        if buffer_idx.shape[0] > 0:
            # 删除掉的states 给policy重新训练一次
            if policy_net is not None and policy_net_optimizer is not None:
                print("retrain policy by deletion...")
                self.train_policy(epoch=3, rnd_module=rnd_module, policy_net=policy_net,
                                  policy_net_optimizer=policy_net_optimizer,
                                  sample_from_buffer=False,
                                  obs=self.observations[buffer_idx].to("cuda:0"),
                                  actions=self.actions[buffer_idx].to("cuda:0"),
                                  dones=self.dones[buffer_idx].to("cuda:0"),
                                  logprob_og=self.log_probs[buffer_idx].to("cuda:0"))

            # 同理，再训练一次
            self.train_encoder(self.observations[buffer_idx], self.actions[buffer_idx], epochs=2)

            # update the buffer
            envs_idx = torch.tensor([traj.obs_idx[1] for traj in add_li], dtype=torch.long)  # (add num,)
            tmp = [(self.cache_top - prev) % self.traj_len_times for prev in range(0, self.traj_len_times)]
            self.observations[buffer_idx] = torch.cat([self.cache[i, :, envs_idx, :, :] for i in tmp],
                                                      dim=0).transpose(0, 1).cpu()
            self.actions[buffer_idx] = torch.cat([self.cache_actions[i, :, envs_idx] for i in tmp], dim=0).transpose(0,
                                                                                                                     1).cpu()
            self.dones[buffer_idx] = torch.cat([self.cache_dones[i, :, envs_idx] for i in tmp], dim=0).transpose(0,
                                                                                                                 1).cpu()
            self.log_probs[buffer_idx] = torch.cat([self.cache_log_probs[i, :, envs_idx] for i in tmp],
                                                   dim=0).transpose(0, 1).cpu()
            self.f_novelty_diversity[buffer_idx] = torch.tensor([traj.nd_value for traj in add_li]).cpu()

            frontier_indices = torch.tensor(
                [traj.obs_idx[0] + self.roll_out_len * traj.obs_idx[1] for traj in add_li])
            obs_v = obs.reshape((-1, self.dim_x, self.dim_y))
            self.frontier[buffer_idx] = obs_v[frontier_indices, :, :].cpu().clone()
            self.f_novelties[buffer_idx] = torch.tensor([t.rnd_v for t in add_li]).cpu().clone()
            self.frontier_idx[buffer_idx] = (self.traj_len_times - 1) * self.roll_out_len + torch.tensor(
                [traj.obs_idx[0] for traj in add_li]).cpu().clone()
            self.dependency_indices[buffer_idx] = torch.tensor([t.dependent_idx for t in add_li],
                                                               dtype=torch.long).cpu().clone()

            # the deletion may cause other frontiers's diversities to change！
            update_idx = torch.nonzero(torch.isin(self.dependency_indices, buffer_idx)).squeeze(
                dim=1).cpu()  # (update nums, )
            if update_idx.shape[0] > 0:
                new_div = self.b_encoder.get_diversity(self.frontier[update_idx],
                                                       self.frontier)  # (update nums, buffer size )
                new_div, new_dpt = torch.min(new_div, dim=1)  # (update nums,), (update nums,)
                self.f_novelty_diversity[update_idx] = (
                            new_div.squeeze().cpu() * self.f_novelties[update_idx].squeeze()).cpu()
                self.dependency_indices[update_idx] = new_dpt.to(torch.long).cpu().clone()

    def train_policy(self,
                     epoch: int,
                     rnd_module: UcbNets,
                     policy_net: ACNetwork,
                     policy_net_optimizer,
                     sample_from_buffer: bool = True,
                     batch_size: int = None,
                     minibatch_size=80,
                     train_logratio_clip= 10,
                     obs: torch.Tensor = None,  # (B, traj len, dim x, dim y)
                     actions: torch.Tensor = None,  # (B, traj len,
                     dones: torch.Tensor = None,  # (B, traj len,
                     logprob_og: torch.Tensor = None,  # (B, traj len,
                     ):

        with torch.no_grad():
            if sample_from_buffer:
                _, obs, actions, dones, logprob_og, _ = self.sample(batch_size,
                                                                    rnd_module.device)  # obs (B, traj len, dim x, dim y)
            batch_size, traj_len = actions.shape[0], actions.shape[1]
            dim_x, dim_y = obs.shape[2], obs.shape[3]
            curiosity_rewards = rnd_module(obs[:, 1:, :, :].reshape((-1, dim_x, dim_y))).mean(1).reshape(
                (batch_size, traj_len - 1))  # (B,  traj_len -1)
            curiosity_rewards = torch.cat(
                [curiosity_rewards, torch.zeros((batch_size,)).unsqueeze(1).to(curiosity_rewards.get_device())],
                dim=1)  # (B,  traj_len)

            curiosity_rewards[:, 1:-1] = curiosity_rewards[:, 1:-1] - self.args.novelD_alpha * curiosity_rewards[:,
                                                                                               0:-2]
            curiosity_rewards[:, -1] = curiosity_rewards[:, -2]  # The last reward  (B,  traj_len)
            curiosity_rewards[:, 0] = curiosity_rewards[:, 1]  # The first reward  (B,  traj_len)
            curiosity_rewards[
                curiosity_rewards < self.args.clip_intrinsic_reward_min] = 0

            assert curiosity_rewards.shape[1] == traj_len
            policy_net.eval()
            policy, critics = policy_net(
                obs.reshape(-1, self.dim_x, self.dim_y).unsqueeze(1))  # (B * traj_len,action num) , (B*traj_len,)
            policy, critics = policy.reshape((batch_size, traj_len, self.actions_num)), critics.reshape(
                (batch_size, traj_len))  # (B, traj_len, action num) , (B, traj_len)
            assert policy.size(-1) == self.actions_num, "Categorical "
            p = Categorical(logits=policy)
            log_prob1 = p.log_prob(actions).clone()  # (B,traj_lens)
            int_returns, int_advantages = self._calculate_off_policy_gae(policy_net,
                                                                         last_obs=obs[:, -1, :, :],
                                                                         curiosity_rewards=curiosity_rewards,
                                                                         device=logprob_og.get_device(),
                                                                         log_probS_og=logprob_og,
                                                                         logprobs=log_prob1,  # (B,traj_lens)
                                                                         int_values=critics,
                                                                         dones=dones,
                                                                         )
            del policy, critics, p, curiosity_rewards

        obs = obs.reshape((-1, self.dim_x, self.dim_y))
        actions = actions.reshape(-1)
        log_prob1 = log_prob1.reshape(-1)
        int_advantages = int_advantages.reshape(-1)
        int_returns = int_returns.reshape(-1)

        policy_net.train()

        total_size = obs.shape[0]
        clipfracs = []
        b_inds = np.arange(total_size)
        for epoch in range(epoch):
            np.random.shuffle(b_inds)
            for start in range(0, total_size, minibatch_size):
                end = start + minibatch_size
                end = end if end < total_size else total_size
                if start >= end:
                    break
                mb_inds = b_inds[start:end]
                policy, critic = policy_net(obs[mb_inds].unsqueeze(1))
                assert policy.size(-1) == self.actions_num, "Categorical "
                p = Categorical(logits=policy)
                new_log_probs = p.log_prob(actions[mb_inds])
                logratio = new_log_probs - log_prob1[mb_inds]
                logratio = torch.clip(logratio, min=-train_logratio_clip, max=train_logratio_clip, )
                ratio = logratio.exp()
                print(logratio.mean().item(), ",", ratio.mean().item(), "ratio is influenced by BN", mb_inds.shape)

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                mb_advantages = int_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_int_values = critic.view(-1)
                v_loss = ((new_int_values - int_returns[mb_inds]) ** 2).mean()

                loss = pg_loss + v_loss * self.args.vf_coef
                policy_net_optimizer.zero_grad()
                loss.backward()

                if self.args.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        policy_net.parameters(),
                        self.args.max_grad_norm,
                    )
                policy_net_optimizer.step()

            if self.args.target_kl is not None:
                if approx_kl > self.args.target_kl:
                    break

    @torch.no_grad()
    def _calculate_off_policy_gae(self,
                                  policy_net: ACNetwork,
                                  last_obs,  # (B, dim x, dim y)
                                  curiosity_rewards,  # (B,lens)
                                  device,
                                  log_probS_og,  # (B,lens)
                                  logprobs,  # (B,lens)
                                  int_values,  # (B,lens)
                                  dones,  # (B,lens)
                                  ):
        lens = curiosity_rewards.shape[1]
        last_obs = last_obs.unsqueeze(1)  # (B,1, dim x, dim y)
        assert last_obs.shape[1] == 1 and len(last_obs.shape) == 4, "the roll out dim is used as the channel dim" + str(
            last_obs.shape[1]) + ", " + str(len(last_obs.shape))
        next_value_int = policy_net.get_critic(last_obs)
        next_value_int = next_value_int.reshape(1, -1)
        int_advantages = torch.zeros_like(curiosity_rewards, device=device)
        int_lastgaelam = 0
        for t in reversed(range(lens)):
            importance = torch.exp(log_probS_og[:, t] - logprobs[:, t])  # (B,)
            # print(importance.mean(),"importance")
            importance = torch.clip(importance, max=10, min=0.01)  # (B,)
            print(importance.mean(), "importance")
            if t == lens - 1:
                nextnonterminal = 1.0
                int_nextvalues = next_value_int
                int_nextvalues *= importance
            else:
                nextnonterminal = 1.0 - dones[:, t + 1]
                int_nextvalues = int_values[:, t + 1]

            int_delta = curiosity_rewards[:, t] + self.args.int_gamma * nextnonterminal * int_nextvalues - int_values[:,
                                                                                                           t]
            int_advantages[:, t] = int_lastgaelam = (
                    int_delta + self.args.int_gamma * self.args.gae_lambda * nextnonterminal * int_lastgaelam * importance
            )
        int_returns = int_advantages + int_values
        return int_returns, int_advantages

    def sample(self,
               batch_size: int,
               device,
               ):
        sorted, indices = torch.sort(self.f_novelty_diversity, descending=False)
        indices = indices[-batch_size:].clone()
        indices = indices[torch.randperm(indices.shape[0])].to(device)
        s_obs = self.observations[indices, :, :, :].to(device)
        s_actions = self.actions[indices, :].to(device)
        s_dones = self.dones[indices, :].to(device)
        s_logprob = self.log_probs[indices, :].to(device)
        s_f_indices = self.frontier_idx[indices, :].to(device)
        return indices, \
               s_obs, \
               s_actions, \
               s_dones, \
               s_logprob, \
               s_f_indices


if __name__ == "__main__":
    buffer_size = 12
    action_space = 3
    ema_beta = 0.3
    dim_x = 44
    dim_y = 33
    n_envs = 3
    initial_traj_len_times = 3
    roll_out_len = 12
    emb_dim = 9
    # batch_size = 13
    args = get_test_arg()
    buffer = RNDReplayBuffer(args, buffer_size, action_space, ema_beta, dim_x, dim_y,
                             n_envs, initial_traj_len_times, roll_out_len, emb_dim, encoder_learning_rate=0.01,
                             initial_encoder_train_epoches=1, )
    device = "cuda:0"
    ac_emb_dim = 23
    # TODO 把byol_encoder里面的 BN完全去掉，完全放在AC net里面
    byol_encoder = BYOLEncoder(in_channels=1, out_size=32, emb_dim=ac_emb_dim).to(device)  # output size 600
    ac_network = ACNetwork(device, action_space,byol_encoder=byol_encoder, indim=ac_emb_dim).to(device)
    policy_net = ac_network.to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

    rnd_module = UcbNets(device, args, action_space, dim_x, dim_y, net_num=args.net_num,
                         filter_size=3, ae_sin_size=12, hidden_units=10, in_dim=1530).to(
        device)

    for _ in range(7777):
        obs = torch.randn((roll_out_len, n_envs, dim_x, dim_y), device=device)/2  # TODO 这个device真实吗？
        rnd_values = rnd_module(obs.reshape(-1, dim_x, dim_y)).sum(1).reshape(roll_out_len, n_envs)  # * math.sqrt(i)
        rnd_values[1:] -= 0.5 * rnd_values[:-1]  # *  math.sqrt(i**2)
        # rnd_values = torch.randn((roll_out_len, n_envs), device=device) #* math.sqrt(i)
        rnd_values_min = 0.01  # * math.sqrt(i)
        print("round", math.sqrt(i), "*****************************************************")
        actions = torch.randint(high=action_space, size=(roll_out_len, n_envs), device=device)
        done = torch.randint(high=2, size=(roll_out_len, n_envs), device=device)
        obs_v = obs.reshape(-1, dim_x, dim_y).unsqueeze(1)
        policy_net.eval()
        logits, _ = policy_net(obs_v)
        logits = logits.reshape(roll_out_len, n_envs, -1)

        p = Categorical(logits=logits)
        log_probs = p.log_prob(actions)

        # 这个测试有问题
        buffer.train_policy(epoch=3, rnd_module=rnd_module,
                            policy_net=policy_net,
                            policy_net_optimizer=optimizer,
                            sample_from_buffer=False,
                            obs=obs.transpose(0, 1),
                            actions=actions.t(),
                            dones=done.t(),
                            logprob_og=log_probs.t())

        buffer.add(obs, rnd_values, rnd_values_min, actions, done, log_probs, rnd_module,
                   policy_net, optimizer
                   )
