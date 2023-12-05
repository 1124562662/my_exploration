import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Any, Dict, List, Optional
from torch.distributions.categorical import Categorical
import numpy as np
from BYOL import BYOL,BYOLEncoder
from .Networks.ACNetwork import  ACNetwork


from never_give_up import Within_episode_BYOL_RND,Within_episode_SA_RND,Within_episode_kernel


class IntrinsicAgent(nn.Module):
    def __init__(self,envs,
                 args:argparse.Namespace,
                 device,
                 learning_rate=1e-3,
                 I_action_gumbel_max_tau: float = 1,
                 I_ubc_gumbel_max_tau: float = 1,
                 pseudo_ucb_coef: float = 1,
                 within_episode_method='kernel',
                 within_eposide_coef:float = 1,
                 byol_update_tau:float = 0.01,
                 ):
        super(IntrinsicAgent, self).__init__()
        self.envs = envs
        self.device = device
        self.clip_intrinsic_reward_min = args.clip_intrinsic_reward_min

        self.byol_encoder = BYOLEncoder(in_channels=1, out_size=888) # output size 600
        self.ac_network = ACNetwork(device,envs ,indim=888).to(device)
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=learning_rate)
        self.action_space = envs.action_space.n

        # used for sample action
        self.I_action_gumbel_max_tau = I_action_gumbel_max_tau

        # pseudo count for the (s,a) pairs
        self.pseudo_ucb_coef = pseudo_ucb_coef  # the scale of the ubc count, related to empowerment
        self.I_ubc_gumbel_max_tau = I_ubc_gumbel_max_tau
        self.pseudo_ucb_net = UcbCountNet(env.action_space.n, embed_dim=args.embed_dim, target=False).to(device)
        self.pseudo_ucb_optimizer = optim.Adam(self.pseudo_ucb_net.parameters(), lr=args.pseudo_ucb_optimizer_lr)
        self.pseudo_ucb_target = UcbCountNet(env.action_space.n, embed_dim=args.embed_dim, target=True).to(device)

        # consider within episode novelty.
        # use option (b) first,
        self.within_eposide_coef = within_eposide_coef
        if within_episode_method == 'kernel' or within_episode_method == 'BYOL_RND':
            self.byol = BYOL(obs_dim=600 ,action_dim=self.action_space ,tau=byol_update_tau,num_hidden=3 ,num_units=256,emb_dim=512
                             ,hidden_size=256, num_layers=2 ,action_e_dim=80)
            if within_episode_method == 'kernel':
                self.within_episode_kernel = Within_episode_kernel()
                self.within_episode_kernel.clear()
            elif within_episode_method == 'BYOL_RND':
                self.within_episode_BYOL_RND = Within_episode_BYOL_RND() # obs_embeds_dim,action_embeds_dim, action_num,env_num,
                                                                         # embed_dim=50, num_hidden=2, num_units=50, learning_rate=0.01,)
        elif within_episode_method == 'SA_RND':
            # Attention! this is used with no BYOL model!
            self.within_episode_BYOL_RND = Within_episode_SA_RND() # obs_embeds_dim, action_embeds_dim, action_num, env_num, embed_dim=50, num_hidden=2, num_units=50,
                                                                   # learning_rate=0.01,
        else:
            raise NotImplementedError("within_episode_method in {kernel,BYOL_RND,SA_RND}")


    def get_action(self,
                   obs_,
                   action_given=None,
                   previous_c_hn=None,
                   previous_open_loop_rnn_hiddens=None,
                   byol_predict_horizon = 20,
                   given_ubc_values =None,
                   intrinsic_reward=None):
        obs_ = obs_.to(self.device).to(torch.float32)
        obs_emb = self.byol_encoder(obs_) #TODO 有问题 RND应该用 raw input
        if given_ubc_values is None:
            # pseudo count of UBC
            with torch.no_grad():
                predicted_values = self.pseudo_ucb_net(obs_)  # (env,action_nums,embed_dim)
                target_value = self.pseudo_ucb_target(obs_)  # (env,action_nums,embed_dim)
            ubc_values_ = F.mse_loss(predicted_values.detach(), target_value.detach(),
                                     reduction='none')  # (env,action_nums,embed_dim)
            ubc_values_ = torch.mean(ubc_values_, dim=-1)  # (env,action_nums)
            ubc_values_ = torch.sqrt(ubc_values_)  # sqrt() similar to UCB
            intrinsic_reward_ = ubc_values_.mean(dim=-1)  # (env,)

            # consider within episode similarity
            if self.within_episode_method == 'kernel' or self.within_episode_method == 'BYOL_RND':
                previous_states_nums = previous_open_loop_rnn_hiddens.size(1)
                poh_tmp = previous_open_loop_rnn_hiddens  # (numlayers,previous states nums P,envNum, hidden size H)
                if previous_states_nums > byol_predict_horizon:
                    poh_tmp = poh_tmp[: ,0:byol_predict_horizon ,: ,:]  # (numlayers,byol_predic_horizon,envNum, hidden size H)
                o_pred ,byol_obs = self.byol.get_action_predictions(obs_emb,previous_c_hn=previous_c_hn,o_hiddens= poh_tmp) # (byol_predic_horizon, envNum, action_dim , embedding ),(envNum ,E)

                if self.within_episode_method == 'kernel':
                    # TODO re_initialize_history
                    self.within_episode_kernel.add(byol_obs) # (env num, byol emb), byol encoding of current state
                    o_pred = o_pred.transpose(1,2)  # (byol_predic_horizon, action_dim , envNum, embedding)
                    o_pred_size = o_pred.size()# (byol_predic_horizon, action_dim , envNum, embedding)
                    o_pred = o_pred.reshape \
                        ((-1, o_pred_size[2], o_pred_size[3]))  # (byol_predic_horizon * action_dim , envNum, embedding)

                    predicted_within_episode_rewards = self.within_episode_kernel.get_similarity_kernel(o_pred) # ( env nums,byol_predic_horizon * action_dim)
                    predicted_within_episode_rewards = predicted_within_episode_rewards.reshape((o_pred_size[2],o_pred_size[1],o_pred_size[0]))# ( env nums,action_dim, byol_predic_horizon)
                    predicted_within_episode_rewards = predicted_within_episode_rewards.mean(-1) # ( env nums,action_dim )
                    ubc_values_ = ubc_values_ * predicted_within_episode_rewards  # ( env nums,action_dim )

                elif self.within_episode_method == 'BYOL_RND':
                    # TODO re_initialize_network
                    r_epoch = 2
                    loss_return = self.within_episode_BYOL_RND.get_similarity_and_update_RND(self.byol.action_embedding,o_pred, r_epoch) # (env_num , action_pred_num)
                    ubc_values_ = ubc_values_ * loss_return  # ( env nums,action_dim )

            elif self.within_episode_method == 'SA_RND':
                pass
        else: # during training, those values are given, to guarantee on-policy-ness
            ubc_values_ = given_ubc_values
            intrinsic_reward_ = intrinsic_reward

        policy, critic = self.ac_network(obs_) # TODO -- will ac network use byol encoder?
        for k in range(policy.size()[0]):
            tau = 1.0
            if intrinsic_reward_[k].float() > self.clip_intrinsic_reward_min:
                tau += intrinsic_reward_[k].float()
            policy[k] = F.gumbel_softmax(policy[k], hard=False, tau=tau)

        policy = policy + self.pseudo_ucb_coef * ubc_values_  # (env,action_nums)

        probs = Categorical(probs=policy)
        if action_given is None:
            actions_arg = probs.sample() # (env, ),  instead of torch.argmax(policy, dim=1)
        else:
            actions_arg = action_given # (env, )

        ubc_loss = None # TODO
        # calculate the loss of the ucb network
        # ubc_loss = ubc_values_og[:, actions_arg].mean()  # 128 should be the mean episodic length* envnum
        # ubc_loss.backward()
        return actions_arg, \
               probs.log_prob( actions_arg), \
               probs.entropy(), \
               intrinsic_reward_, \
               ubc_loss, \
               critic, \
               ubc_values_

    @staticmethod
    def obvs_preprocess(obvs):
        obvs = torch.Tensor(obvs).mean(dim=-1)
        obvs = torch.nn.functional.avg_pool2d(obvs, kernel_size=3)
        return obvs.cpu().detach().numpy()  # [env_num, 70, 53]

    @staticmethod
    def update_info_buffer(ep_info_buffer, ep_success_buffer, infos: List[Dict[str, Any]],
                           dones: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """

        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                ep_success_buffer.append(maybe_is_success)



    def rollout_step(self,args,device,dim_x,dim_y,envs,
                     ep_info_buffer, ep_success_buffer,rb,
                     last_ep_c_hn,
                     ):
        obs = torch.zeros((args.num_steps, args.num_envs) + (dim_x, dim_y)).to(device)  # (steps, env nums, dimx,dimy)
        actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
        extrinsic_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        curiosity_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
        episode_ubc_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rnd_intrinsic_reward_for_training = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # attention: this one's shape!
        closed_loop_hn_for_training = torch.zeros((args.num_steps,
                                                   self.byol.num_layers,
                                                   args.num_envs,
                                                   self.byol.hidden_size)).to(
            "cpu")  # (steps, byol num layers, env nums, byol hidden size)

        c_hn = None
        previous_open_loop_rnn_hiddens = None

        previous_intrinsic_rewards = torch.zeros(args.num_envs)  # (num envs,)
        previous_byol_rewards = None


        next_obs = envs.reset()
        next_obs = self.obvs_preprocess(next_obs)
        next_done = torch.zeros(args.num_envs).numpy()
        self.eval()
        for step in range(args.num_steps):
            with torch.no_grad():
                # get action
                actions_arg, action_log_prob, entropy, rnd_intrinsic_reward, _, int_v, ubc_values = self.get_action(
                    torch.tensor(next_obs),
                    previous_c_hn=c_hn,
                    previous_open_loop_rnn_hiddens=previous_open_loop_rnn_hiddens,
                    byol_predict_horizon=args.byol_predict_horizon, )

            logprobs[step] = action_log_prob
            obs[step] = torch.from_numpy(next_obs).to(device)
            dones[step] = torch.from_numpy(next_done).to(device)
            actions[step] = torch.from_numpy(actions_arg).to(device)
            int_values[step] = int_v.flatten()
            episode_ubc_values[step] = ubc_values.detach()  # TODO: use this in training
            rnd_intrinsic_reward_for_training[step] = rnd_intrinsic_reward
            if step > 0:
                # see novelID
                novelD_reward = rnd_intrinsic_reward.clone()
                novelD_reward = novelD_reward - args.novelD_alpha * previous_intrinsic_rewards
                novelD_reward[
                    novelD_reward < args.clip_intrinsic_reward_min] = 0  # clip small rewards to 0, to avoid cumulative small rewards

                # total curiosity rewards
                curiosity_rewards[step - 1] = novelD_reward + previous_byol_rewards
            # update for novelD
            previous_intrinsic_rewards = rnd_intrinsic_reward.clone()

            # env step() call
            real_next_obs, rewards, next_done, infos = envs.step(actions_arg)
            real_next_obs = self.obvs_preprocess(real_next_obs)

            extrinsic_rewards[step] = torch.from_numpy(rewards).to(device)

            # see BYOL
            with torch.no_grad():
                previous_open_loop_rnn_hiddens, previous_byol_rewards, c_hn = self.byol.get_intrinsic_reward(
                    next_obs,
                    actions_arg,
                    real_next_obs,
                    previous_c_hn=c_hn,
                    o_hiddens=previous_open_loop_rnn_hiddens)

            # cut by byol_open_loop_gru_max_horizon
            if previous_open_loop_rnn_hiddens.size(
                    1) > args.byol_open_loop_gru_max_horizon:  # (numlayers, previous states nums P+1 * envNum, hidden size H)
                if step % args.byol_open_loop_gru_max_horizon == 1:
                    previous_open_loop_rnn_hiddens = previous_open_loop_rnn_hiddens[:,
                                                     -args.byol_open_loop_gru_max_horizon:, :]
                    previous_open_loop_rnn_hiddens_2 = previous_open_loop_rnn_hiddens.detach().clone()  # slicing is a view, how to save memory?
                    del previous_open_loop_rnn_hiddens
                    previous_open_loop_rnn_hiddens = previous_open_loop_rnn_hiddens_2
                else:
                    previous_open_loop_rnn_hiddens = previous_open_loop_rnn_hiddens[:,
                                                     -args.byol_open_loop_gru_max_horizon:, :]

            closed_loop_hn_for_training[step] = c_hn.detach().clone().cpu()  # saved for training

            # if intrinsic_reward_statistics_count == 0:
            #     intrinsic_reward_statistics = rnd_intrinsic_reward.mean()
            # else:
            #     intrinsic_reward_statistics = intrinsic_reward_statistics + (
            #             rnd_intrinsic_reward.mean() - intrinsic_reward_statistics) / intrinsic_reward_statistics_count
            # intrinsic_reward_statistics_count += 1

            self.update_info_buffer(ep_info_buffer, ep_success_buffer, infos, next_done)
            real_next_obs = real_next_obs.copy()
            rb.add(next_obs, real_next_obs, actions_arg, rewards, next_done, infos)
            next_obs = real_next_obs

        # bootstrap value if not done
        curiosity_rewards[-1, :] = curiosity_rewards[-2, :]  # The last reward
        with torch.no_grad():
            next_obs = torch.from_numpy(next_obs)
            _, _, _, _, ucb_loss, next_value_int, _ = self.get_action(next_obs)
            next_value_int = next_value_int.reshape(1, -1)
            int_advantages = torch.zeros_like(curiosity_rewards, device=device)
            int_lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                int_nextnonterminal = 1.0
                if t == args.num_steps - 1:
                    int_nextvalues = next_value_int
                else:
                    int_nextvalues = int_values[t + 1]
                int_delta = curiosity_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                int_advantages[t] = int_lastgaelam = (
                        int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
                )
            int_returns = int_advantages + int_values

        # flatten the batch
        b_obs = obs.reshape((-1,) + (dim_x, dim_y))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = int_advantages.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_inds = np.arange(args.batch_size)


        self.ac_network.train()
        self.pseudo_ucb_net.train()
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Train RND
                UcbCountNet.train_RND(device,args,mb_inds,b_obs, b_actions,self.pseudo_ucb_net,self.pseudo_ucb_target)

                _, newlogprob, entropy, _, _, new_int_values, ubc_values = self.get_action(
                    b_obs[mb_inds], action_given=b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
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
                        self.ac_network.parameters(),
                        args.max_grad_norm,
                    )
                self.optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # TODO -- how to train BYOL?
        self.byol.train()
        o_minibatch = 5
        self.byol.train_byol(args, device,
                                b_obs,
                                b_actions,
                                byol_encoder=self.byol_encoder,
                                steps=args.num_steps,
                                c_hn_t=closed_loop_hn_for_training,
                                last_ep_c_hn=last_ep_c_hn,
                                o_minibatch=o_minibatch, )

        last_ep_c_hn = closed_loop_hn_for_training[-1, :, :, :].squeeze()  # ( num_layers, num_envs ,  hidden_size),
        # handle the starting states, set  starting c_hn to all ones
        # TODO：done == true 到底是下一个还是这一个？
        last_ep_c_hn = last_ep_c_hn.transpose(0, 1).reshape((args.num_envs, -1))  # ( env nums, num_layers * byol_hidden)
        last_ep_c_hn[dones[-1, :] == 1,:] = 1.0  # ( env nums, num_layers * byol_hidden), the next state will be start, so its c_hn is all ones
        last_ep_c_hn = last_ep_c_hn.reshape((args.num_envs, self.byol.num_layers, -1)).transpose(0, 1)  # ( num_layers, num_envs ,  hidden_size)

        return last_ep_c_hn