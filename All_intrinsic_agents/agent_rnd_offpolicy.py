import argparse
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Any, Dict, List, Optional
from torch.distributions.categorical import Categorical
import numpy as np
from ..Networks.ACNetwork import ACNetwork
from ..Networks.UcbCountMultipleNets import UcbNets
from ..BYOL import BYOLEncoder
from ..utils.Multi_dim_gumbel_softmax import multi_dim_softmax
from ..utils.utils import obvs_preprocess, update_info_buffer
from ..utils.normalizer import TorchRunningMeanStd


class IntrinsicAgent(nn.Module):
    def __init__(self, envs,
                 args: argparse.Namespace,
                 device,
                 dim_x: int,
                 dim_y: int,
                 learning_rate=1e-3,
                 I_action_gumbel_max_tau: float = 1,
                 I_ubc_gumbel_max_tau: float = 1,
                 pseudo_ucb_coef: float = 1,

                 # RND networks parameters
                 ae_sin_size: int = 128,
                 filter_size: int = 3,
                 hidden_units: int = 512,
                 ac_emb_dim: int = 300,
                 use_only_UBC_exploration_threshold: float = 0.7,
                 ):
        super(IntrinsicAgent, self).__init__()
        self.envs = envs
        self.device = device
        self.clip_intrinsic_reward_min = args.clip_intrinsic_reward_min

        byol_encoder = BYOLEncoder(in_channels=1, out_size=192, emb_dim=ac_emb_dim).to(device)  # output size 600
        ac_network = ACNetwork(device, envs, indim=ac_emb_dim).to(device)
        self.policy_net = nn.Sequential(byol_encoder, ac_network).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.action_space = envs.action_space.n

        # used for sample action
        self.I_action_gumbel_max_tau = I_action_gumbel_max_tau
        # self.obs_stats =  TorchRunningMeanStd((dim_x, dim_y), device=device,min_size=args.num_steps) # cuda

        # pseudo count for the (s,a) pairs
        self.pseudo_ucb_coef = pseudo_ucb_coef  # the scale of the ubc count, related to empowerment
        self.I_ubc_gumbel_max_tau = I_ubc_gumbel_max_tau
        self.pseudo_ucb_nets = UcbNets(device, args, self.action_space, dim_x, dim_y, net_num=args.net_num,
                                       filter_size=filter_size, ae_sin_size=ae_sin_size, hidden_units=hidden_units).to(
            device)

        self.ubc_statistics = TorchRunningMeanStd((args.num_envs, envs.action_space.n), device=device,
                                                  min_size=args.num_steps)  # shape of (args.num_envs, action num)

        self.use_only_UBC_exploration_threshold = use_only_UBC_exploration_threshold

    def get_action_train(self,
                         obs_  # (batch size,  dim x, dim y)
                         , action_given  # (batch size, )
                         , ):
        # TODO 加入batch size 维度
        obs_ = obs_.to(self.device).to(torch.float32)
        policy, critic = self.policy_net(obs_)  # (batch size, action_nums),  (batch size, )
        probs = Categorical(logits=policy)
        return probs.log_prob(action_given), \
               probs.entropy(), \
               critic  # todo -- entropy 这一项也要根据ucb调节，训练时候

    @torch.no_grad()
    def get_action(self,
                   obs_,  # (envs, dim x, dim y)
                   ):
        # TODO
        self.use_only_UBC_exploration_threshold = 0.7

        obs_ = obs_.to(self.device).to(torch.float32)
        # pseudo count of UBC
        ubc_values = self.pseudo_ucb_nets(obs_)  # (env,action_nums)
        self.ubc_statistics.update_single(ubc_values)
        ubc_values = self.ubc_statistics.normalize_by_var(ubc_values)
        ubc_values_ = torch.sqrt(ubc_values)  # sqrt() similar to UCB
        intrinsic_reward_ = ubc_values_.mean(dim=-1)  # (env,)
        msk = intrinsic_reward_ > self.use_only_UBC_exploration_threshold  # (envs,) the envs that only use Ucb as the policy

        policy_og, critic = self.policy_net(obs_)  # (env,action_nums),  (env,)

        tau_add_one = intrinsic_reward_.clone()  # (env,)
        tau_add_one[tau_add_one < self.clip_intrinsic_reward_min] = 0  # (env,)
        policy = multi_dim_softmax(logits=policy_og, tau_add_one=tau_add_one)  # (env,action_nums)
        policy = policy + self.pseudo_ucb_coef * ubc_values_  # (env,action_nums) # TODO set self.pseudo_ucb_coef here
        policy[msk] = ubc_values_[msk]  # the envs that only use Ucb as the policy
        probs = Categorical(probs=policy)  # Attention！it is probs here，not logits！ no softmax here！
        actions_arg = probs.sample()  # (env, ),  instead of torch.argmax(policy, dim=1)
        return actions_arg, \
               probs.log_prob(actions_arg), \
               probs.entropy(), \
               intrinsic_reward_, \
               critic, \
               ubc_values_, \
               Categorical(logits=policy_og).log_prob(actions_arg)

    @torch.no_grad()
    def calculate_off_policy_gae(self,
                                 last_obs, curiosity_rewards, args, device, log_probS_og, logprobs, int_values
                                 ):
        # next_obs = torch.from_numpy(next_obs)
        _, _, _, _, next_value_int, _, _ = self.get_action(last_obs)
        next_value_int = next_value_int.reshape(1, -1)
        int_advantages = torch.zeros_like(curiosity_rewards, device=device)
        int_lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            int_nextnonterminal = 1.0
            importance = torch.exp(log_probS_og[t] - logprobs[t])  # (envs,)
            importance = torch.clip(importance, max=10, min=0.001)  # (envs,)

            if t == args.num_steps - 1:
                int_nextvalues = next_value_int
                int_nextvalues *= importance
            else:
                int_nextvalues = int_values[t + 1]

            int_delta = curiosity_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - \
                        int_values[t]

            int_advantages[t] = int_lastgaelam = (
                    int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam * importance
            )
        int_returns = int_advantages + int_values
        return int_returns, int_advantages

    def rollout_step(self, args, device, dim_x, dim_y, envs,
                     ep_info_buffer, ep_success_buffer, rb, global_step: int,
                     next_obs,
                     ):
        if global_step == 0:
            print("Start to initialize observation normalization parameter.....")
            envs.reset()
            # tmp_rnd = copy.deepcopy(self.pseudo_ucb_nets)
            for step in range(1000):
                acs = np.random.randint(0, envs.action_space.n, size=(args.num_envs,))
                s, r, d, _ = envs.step(acs)
                t_obs = obvs_preprocess(s, device=device)
                # self.obs_stats.update_single(t_obs)
                rew = self.pseudo_ucb_nets(t_obs)
                self.ubc_statistics.update_single(rew)
            # self.obs_stats.count = 80
            self.ubc_statistics.count = args.num_steps
            print("End to initialize...")

        obs = torch.zeros((args.num_steps, args.num_envs) + (dim_x, dim_y)).to(device)  # (steps, env nums, dimx,dimy)
        actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
        extrinsic_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        log_probS_og = torch.zeros((args.num_steps, args.num_envs)).to(device)
        curiosity_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        # rnd_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
        # ubc_values_all = torch.zeros((args.num_steps, args.num_envs, self.action_space)).to(device)
        previous_intrinsic_rewards = torch.zeros(args.num_envs)  # (num envs,)

        next_done = torch.zeros(args.num_envs).cpu().numpy()
        self.eval()

        for step in range(args.num_steps):
            with torch.no_grad():
                # get action
                actions_arg, action_log_prob, entropy, rnd_intrinsic_reward, int_v, ubc_values, log_prob_og = self.get_action(
                    next_obs)

            logprobs[step] = action_log_prob
            log_probS_og[step] = log_prob_og
            obs[step] = next_obs  # .to(device)
            dones[step] = torch.from_numpy(next_done).to(device)
            actions[step] = actions_arg.to(device)
            int_values[step] = int_v.flatten()
            # ubc_values_all[step] = ubc_values
            # rnd_rewards[step] = rnd_intrinsic_reward
            if step > 0:
                # see novelD
                novelD_reward = rnd_intrinsic_reward.clone()
                novelD_reward = novelD_reward - args.novelD_alpha * previous_intrinsic_rewards
                novelD_reward[
                    novelD_reward < args.clip_intrinsic_reward_min] = 0  # clip small rewards to 0, to avoid cumulative small rewards

                # total curiosity rewards
                curiosity_rewards[step - 1] = novelD_reward
            # update for novelD
            previous_intrinsic_rewards = rnd_intrinsic_reward.clone()

            # env step() call
            real_next_obs, rewards, next_done, infos = envs.step(actions_arg)
            envs.render("human")
            real_next_obs = obvs_preprocess(real_next_obs, device=device)  # ,obs_stats=self.obs_stats)

            extrinsic_rewards[step] = torch.from_numpy(rewards).to(device)

            update_info_buffer(ep_info_buffer, ep_success_buffer, infos, next_done)
            # real_next_obs = real_next_obs.clone() TODO -- what is this?
            rb.add(next_obs.cpu().detach().numpy(), real_next_obs.cpu().detach().numpy(), actions_arg.cpu().numpy(),
                   rewards, next_done, infos)
            next_obs = real_next_obs

        # bootstrap value if not done
        curiosity_rewards[-1, :] = curiosity_rewards[-2, :]  # The last reward

        # TODO -- 注意！ 这里和 RND 一样没有考虑 dones，没有 episode
        int_returns, int_advantages = self.calculate_off_policy_gae(next_obs, curiosity_rewards, args, device,
                                                                    log_probS_og, logprobs, int_values)
        del int_values
        del extrinsic_rewards
        del logprobs

        # flatten the batch
        b_obs = obs.reshape((-1,) + (dim_x, dim_y))
        b_log_probS_og = log_probS_og.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = int_advantages.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_inds = np.arange(args.batch_size)

        # update stats
        # self.obs_stats.update(b_obs)

        # train novelD
        # mean_i_rewards = curiosity_rewards.mean()
        # max_i_rewards = curiosity_rewards.max()
        # self.pseudo_ucb_nets.train()
        # _, indices = torch.sort(curiosity_rewards[:-1,:].reshape(-1))
        # for epoch in range(args.rnd_update_epochs):  # TODO
        #     nonzero_indices = torch.nonzero(indices[0:args.rnd_train_num * args.num_envs]).squeeze()
        #     # Push the frontier forward,
        #     nonzero_indices += float(args.num_envs)
        #     # print("nonzero_indices.size() ",nonzero_indices.size())
        #     self.pseudo_ucb_nets.train_RNDs(args, nonzero_indices, b_obs, b_actions)
        # del curiosity_rewards, nonzero_indices

        mean_i_rewards = curiosity_rewards.mean()
        max_i_rewards = curiosity_rewards.max()
        self.pseudo_ucb_nets.train()
        for epoch in range(args.rnd_update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = torch.from_numpy(b_inds[start:end]).to(device)
                self.pseudo_ucb_nets.train_RNDs(args, mb_inds.long(), b_obs, b_actions.long(),
                                                train_net_num=args.train_net_num)
        del curiosity_rewards

        # TODO PPO 也要根据 off policy修改
        self.policy_net.train()
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                new_log_probs, entropy, new_int_values = self.get_action_train(b_obs[mb_inds], b_actions[mb_inds])
                logratio = new_log_probs - b_log_probS_og[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_int_values = new_int_values.view(-1)
                v_loss = ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                self.optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.policy_net.parameters(),
                        args.max_grad_norm,
                    )
                self.optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        return mean_i_rewards, max_i_rewards, next_obs
