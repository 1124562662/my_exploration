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

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

# class Trajectory:
#     def __init__(self,device="cpu",):
#         self.actual_len = 0

class ReplayBuffer(BaseBuffer):
    def __init__(
            self,
            buffer_size: int,
            action_space: int,
            device: Union[th.device, str] = "cpu",
            dim_x: int = 1,
            dim_y: int = 1,
            n_envs: int = 1,
            handle_timeout_termination: bool = True,
            traj_len:int =128,

    ):

        self.buffer_size = buffer_size
        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        self.traj_len = traj_len
        self.observations = torch.zeros((buffer_size, traj_len,dim_x, dim_y),device="cpu")
        self.actual_lens = torch.zeros((buffer_size, traj_len),device="cpu")
        self.novelty_diversity = torch.zeros((buffer_size, traj_len), device="cpu")
        self.actions = torch.zeros((buffer_size, traj_len,self.action_dim), dtype=torch.long,device="cpu")
        self.rewards = torch.zeros((buffer_size,traj_len), device="cpu")

        # min_heap
        if psutil is not None:
            total_memory_usage = self.observations.element_size() + self.actions.element_size()+ self.rewards.element_size()


            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            reward: torch.Tensor,
            done: torch.Tensor,
    ) -> None:
        # add 时候整块加进来，（envs，rollout len）
        # 缓存前n块。
        # 先判断是否准入，如果最大的novelty足够了，可以就加入。 然后依据dones复制进 trajectories里面

        pass




    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """

        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)


        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
