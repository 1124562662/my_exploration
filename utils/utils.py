from typing import Optional, List, Dict, Any

import torch

from typing import Tuple

import numpy as np
from .normalizer import TorchRunningMeanStd


def obvs_preprocess(obvs,device,
                    obs_stats: TorchRunningMeanStd = None):
    obvs = torch.Tensor(obvs).to(device)
    if obs_stats is not None:
        obvs = obs_stats.normalize(obvs)
    obvs = obvs.mean(dim=-1)
    obvs = torch.nn.functional.avg_pool2d(obvs, kernel_size=3)
    return obvs.detach() #.cpu().detach().numpy()  # [env_num, 70, 53]


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

#
# class RunningMeanStd:
#     def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = (),bias_to_current:float =1):
#         """
#         Calulates the running mean and std of a data stream
#         https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
#
#         :param epsilon: helps with arithmetic issues
#         :param shape: the shape of the data stream's output
#         """
#         self.mean = np.zeros(shape, np.float64)
#         self.var = np.ones(shape, np.float64)
#         self.count = epsilon
#         self.bias_to_current = bias_to_current
#
#     def copy(self) -> "RunningMeanStd":
#         """
#         :return: Return a copy of the current object.
#         """
#         new_object = RunningMeanStd(shape=self.mean.shape)
#         new_object.mean = self.mean.copy()
#         new_object.var = self.var.copy()
#         new_object.count = float(self.count)
#         return new_object
#
#     def combine(self, other: "RunningMeanStd") -> None:
#         """
#         Combine stats from another ``RunningMeanStd`` object.
#
#         :param other: The other object to combine with.
#         """
#         self.update_from_moments(other.mean, other.var, other.count)
#
#     def update(self, arr: np.ndarray) -> None:
#         batch_mean = np.mean(arr, axis=0)
#         batch_var = np.var(arr, axis=0)
#         batch_count = arr.shape[0]
#         self.update_from_moments(batch_mean, batch_var, batch_count)
#
#     def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
#         new_batch_count  = batch_count * self.bias_to_current
#         delta = batch_mean - self.mean
#         tot_count = self.count + new_batch_count
#
#         new_mean = self.mean + delta * new_batch_count  / tot_count
#         m_a = self.var * self.count
#         m_b = batch_var * new_batch_count
#         m_2 = m_a + m_b + np.square(delta) * self.count * new_batch_count  / tot_count
#         new_var = m_2 / tot_count
#
#         new_count = batch_count + self.count
#
#         self.mean = new_mean
#         self.var = new_var
#         self.count = new_count
#
