import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Any, Dict, List, Optional
import numpy as np
from .Networks.QNetwork import QNetwork



class ExtrinsicQAgent(nn.Module):
    def __init__(self,
                 envs,
                 device,
                 tau,
                 learning_rate=1e-3,
                 E_action_gumbel_max_tau: float = 1, ):

        super(ExtrinsicQAgent, self).__init__()
        self.envs =envs
        self.device =device
        self.tau=tau
        self.q_network = QNetwork(envs).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.target_network = QNetwork(envs).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.action_space = envs.action_space.n
        self.E_action_gumbel_max_tau = E_action_gumbel_max_tau

    def get_action(self, obs):
        q_values = self.q_network(torch.Tensor(obs).to(self.device).to(torch.float32))  # (env,action_nums)
        actions = F.gumbel_softmax(q_values, hard=True, tau=self.E_action_gumbel_max_tau)
        actions_arg = torch.argmax(actions, dim=1).cpu().detach().numpy()
        return actions_arg, 0

    def update_target_Q_network(self):
        for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_network_param.data.copy_(
                self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data
            )


    def rollout_step(self,args,device,dim_x,dim_y,envs,ep_info_buffer, ep_success_buffer,rb, global_step:int,next_obs):

        next_done = torch.zeros(args.num_envs).numpy()

        for step in range(args.num_steps):
            actions_arg, _ = self.get_action( next_obs)
            real_next_obs, rewards, next_done, infos = envs.step(actions_arg)
            real_next_obs = self.obvs_preprocess(real_next_obs)
            self.update_info_buffer(ep_info_buffer, ep_success_buffer, infos, next_done)
            real_next_obs = real_next_obs.copy()
            rb.add(next_obs, real_next_obs, actions_arg, rewards, next_done, infos)
            next_obs = real_next_obs

        if global_step > args.learning_starts:
            if global_step % args.train_extrinsic_frequency == 0:
                for ep in range(args.train_extrinsic_epoches):
                    data = rb.sample(args.batch_size)
                    with torch.no_grad():
                        target_max, _ = self.target_network(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                    old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                    # optimize the model
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            if global_step % args.e_target_network_frequency == 0:
                for target_network_param, q_network_param in zip(self.target_network.parameters(),
                                                                 self.q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )





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


