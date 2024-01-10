import argparse
import copy
import math

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import Adam
from torchvision.models.resnet import BasicBlock
import torch.nn.functional as F
from .CNNResNetEncoder import BaseEncoder


# class EMA:
#     def __init__(self, beta, target_model, online_model):
#         super().__init__()
#         self.beta = beta
#         self.target_model = target_model
#         self.online_model = online_model
#
#     def _update_average(self, old, new, beta):
#         if old is None:
#             return new
#         return old * beta + (1 - beta) * new
#
#     def update_moving_average(self, beta=None):
#         if beta is None:
#             beta = self.beta
#         for current_params, ma_params in zip(self.online_model.parameters(), self.target_model.parameters()):
#             old_weight, up_weight = ma_params.detach().clone(), current_params.detach().clone()
#             ma_params.data = self._update_average(old_weight, up_weight, beta)

# class MyRunningMeanStd:
#
#     def __init__(self, shape , device,min_size=20, mean=0, var=1,count=0):
#         self.device = device
#         self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device) + mean
#         self.var = torch.ones(shape, dtype=torch.float32, device=self.device) * var
#
#         self.count = 0 + count
#
#         self.deltas = []
#         self.min_size = min_size
#
#     @torch.no_grad()
#     def update(self, x):
#         x = x.to(self.device)
#         batch_mean = torch.mean(x, dim=0)
#         batch_var = torch.var(x, dim=0)
#
#         # update count and moments
#         if self.count < 10000:
#             n = x.shape[0]
#             self.count += n
#             delta = batch_mean - self.mean
#             self.mean += delta * n / self.count
#             m_a = self.var * (self.count - n)
#             m_b = batch_var * n
#             M2 = m_a + m_b + torch.square(delta) * self.count * n / self.count
#             self.var = M2 / self.count
#         else:
#             pass
#
#     @torch.no_grad()
#     def update_single(self, x):
#         self.deltas.append(x)
#
#         if len(self.deltas) >= self.min_size:
#             batched_x = torch.concat(self.deltas, dim=0)
#             self.update(batched_x)
#
#             del self.deltas[:]
#
#     @torch.no_grad()
#     def normalize(self, x):
#         return (x.to(self.device) - self.mean) / (torch.sqrt(self.var + 1e-8) )
#
#     @torch.no_grad()
#     def normalize_by_var(self,x):
#         return x / (torch.sqrt(self.var + 1e-8))


class RNDBufferEncoder(nn.Module):
    def __init__(self, emb_dim: int, ema_beta: float, action_num: int,
                 distance_p: float = 2,
                 encoder_learning_rate: float = 0.001,
                 device="cuda:0",
                 ):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.distance_p = distance_p
        self.encoder = BaseEncoder(in_channels=1, action_num=action_num, out_size=192,  # 32
                                   emb_dim=emb_dim).to(device)  # TODO 处理下 BN问题，除非大批量训练，不然不用 BN
        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=encoder_learning_rate)
        # TODO add continual learning methods

        # self.encoder_target = copy.deepcopy(self.encoder).to(device)
        # for p in self.encoder_target.parameters():
        #     p.requires_grad = False
        #
        # self.encoder_ema = EMA(ema_beta, self.encoder_target, self.encoder)

        self.center = torch.zeros((emb_dim,), dtype=torch.float).to(device)

    def set_center(self, center):
        assert center.shape[0] == self.emb_dim and len(center.shape) == 1
        self.center = center.detach().clone()

    def forward(self, x,  # (B, channel, dim x ,dim y)
                ):
        assert len(x.shape) == 4 and x.shape[1] == 1
        x = x.to(self.device)
        return self.encoder(x)

    @torch.no_grad()
    def forward_NGU(self,
                    x,  # (B, channel, dim x ,dim y)
                    ):
        assert len(x.shape) == 4 and x.shape[1] == 1
        x = x.to(self.device)
        return self.encoder.forward_hidden(x)

    def train_encoder(self,
                      states: torch.Tensor,  # (B,rollout, dim x, dim y)
                      actions: torch.Tensor,  # (B,rollout,
                      # done: torch.Tensor,
                      rnd_values: torch.Tensor = None,  # (B,rollout) sample based on this
                      minibatch_size: int = 60,
                      epochs: int = 1,
                      ):
        self.train()
        # self.encoder_ema.update_moving_average()
        assert states.size(1) > 1, "states.size(1) > 1"
        assert len(states.shape) == 4 and len(actions.shape) == 2
        states = states.to(self.device)
        actions = actions.to(self.device)
        if rnd_values is not None:
            rnd_values = rnd_values.to(self.device)

        # 对比学习，IDM model
        batch_size = states.size(1) - 1
        if minibatch_size >= batch_size:
            minibatch_size = batch_size

        obs_a = states[:, :-1, :, :]  # (B, rollout-1, dim x, dim y)
        obs_b = states[:, 1:, :, :]  # (B, rollout-1, dim x, dim y)

        minibatch_num = math.ceil(batch_size / minibatch_size)

        new_pos = torch.zeros((self.emb_dim,), dtype=torch.float).to(self.device)
        b_inds = np.arange(batch_size)
        for epoch in range(epochs):
            np.random.shuffle(b_inds)
            for i in range(minibatch_num):
                start = minibatch_size * i
                end = start + minibatch_size
                end = end if end <= batch_size else batch_size
                if start >= end:
                    break
                mb_inds = b_inds[start:end]

                new_pos += 0.5 * self._train_contrasive(obs_a[:, mb_inds, :, :], obs_b[:, mb_inds, :, :],
                                                        actions[:, mb_inds], reverse=False)
                new_pos += 0.5 * self._train_contrasive(obs_b[:, mb_inds, :, :], obs_a[:, mb_inds, :, :],
                                                        actions[:, mb_inds], reverse=True)
        new_pos /= (epochs + minibatch_num)

        #  additionally train encoder on high-rnd-valued states
        if rnd_values is not None:
            _, indices = torch.sort(rnd_values, dim=1)  # (B, rollout)

            indices_a = indices[:, -int(batch_size / 3):].clone()
            indices_b1 = indices_a + 1
            indices_b1[indices_b1 >= states.size(1)] = indices_a
            indices_b2 = indices_a - 1
            indices_b2[indices_b2 < 0] = indices_a

            indices_a = indices_a[0] * indices.shape[0] + indices_a[1]
            indices_b1 = indices_b1[0] * indices.shape[0] + indices_b1[1]
            indices_b2 = indices_b2[0] * indices.shape[0] + indices_b2[1]

            dim_x, dim_y = states.shape[2], states.shape[3]
            obs = states.reshape((-1, dim_x, dim_y))
            acs = actions.reshape(-1)
            for epoch in range(epochs):
                self._train_contrasive(obs[indices_a, :, :].unsqueeze(1), obs[indices_b1, :, :].unsqueeze(1),
                                       acs[indices_a].unsqueeze(1), reverse=False)
                self._train_contrasive(obs[indices_b2, :, :].unsqueeze(1), obs[indices_a, :, :].unsqueeze(1),
                                       acs[indices_b2].unsqueeze(1), reverse=False)

        self.center = (0.95 * self.center + (1 - 0.95) * new_pos).detach()

    def _train_contrasive(self,
                          view_a: torch.Tensor,  # (batch, mini rollout, dim x, dim y), pos_a
                          view_b: torch.Tensor,  # (batch, mini rollout, dim x, dim y), key
                          actions: torch.Tensor,  # (batch, mini rollout)
                          reverse: bool = False,
                          negative_loss_term_prob=0.8,
                          device="cuda:0",
                          ):
        assert len(view_a.shape) == len(view_b.shape) == 4 and len(actions.shape) == 2
        assert view_a.shape[0] == view_b.shape[0] == actions.shape[0] \
               and view_a.shape[1] == view_b.shape[1] == actions.shape[1]

        mb_size, mini_rollout, dim_x, dim_y = view_a.shape
        view_a = view_a.reshape(-1, 1, dim_x, dim_y)
        view_b = view_b.reshape(-1, 1, dim_x, dim_y)

        pos_a, action_emb_a = self.encoder.forward_train(
            view_a)  # (batch * mini rollout, emb),(batch * mini rollout, emb)

        pos_a = torch.nn.functional.normalize(pos_a, p=self.distance_p, dim=1)  # (batch * mini rollout, emb)

        pos_b, action_emb_b = self.encoder.forward_train(view_b)  # (batch * mini rollout, emb)
        pos_b = torch.nn.functional.normalize(pos_b, p=self.distance_p, dim=1)  # (batch * mini rollout, emb)

        loss = -torch.mul(pos_a, pos_b).mean()
        if torch.rand(1).item() < negative_loss_term_prob:
            center_ = torch.nn.functional.normalize(
                self.center.unsqueeze(0).expand(mb_size * mini_rollout, -1).to(device).detach(), p=self.distance_p,
                dim=1)
            loss += 0.3 * torch.mul(pos_a, center_).mean()
            loss += 0.3 * torch.mul(pos_b, center_).mean()

        #  add inverse dynamic model training,让模型不再关注actions无关的东西
        if not reverse:
            in_a = torch.cat([action_emb_a, action_emb_b], dim=1)
        else:
            in_a = torch.cat([action_emb_b, action_emb_a], dim=1)
        a_pred = self.encoder.action_decoder(in_a)
        loss += torch.nn.functional.cross_entropy(input=a_pred, target=actions.reshape(-1))

        self.encoder_optim.zero_grad()
        loss.backward()
        self.encoder_optim.step()

        return (pos_a.mean(0) + pos_b.mean(0)) / 2

    @torch.no_grad()
    def get_diversity(self,
                      states_A: torch.Tensor,  # (a, dim x ,dim y) or (dim x ,dim y)
                      states_B: torch.Tensor,  # (b, dim x ,dim y) or (dim x ,dim y)
                      tau: float = 1,
                      ):

        self.eval()  # such that BN can work for B=1
        # TODO --  use target encoder or the encoder?
        states_A = states_A.to(self.device)
        states_B = states_B.to(self.device)

        if len(states_A.shape) == 2:
            states_A = states_A.unsqueeze(0)  # (1, dim x ,dim y)
        if len(states_B.shape) == 2:
            states_B = states_B.unsqueeze(0)  # (1, dim x ,dim y)

        assert len(states_A.shape) == len(states_B.shape) == 3

        states_A = states_A.unsqueeze(1)  # (a, 1, dim x ,dim y) add channel
        states_B = states_B.unsqueeze(1)  # (b, 1, dim x ,dim y) add channel

        a = torch.nn.functional.normalize(self.encoder(states_A), p=self.distance_p, dim=1)  # (a, emb)
        b = torch.nn.functional.normalize(self.encoder(states_B), p=self.distance_p, dim=1)  # (b, emb)
        res = torch.matmul(a, b.t())  # (a,b)

        # centering
        res = torch.sigmoid(res / tau)  # (a,b)
        return res  # (a,b) # TODO -- diversities standardization


if __name__ == "__main__":
    emb = 80
    acsd = 18
    e = RNDBufferEncoder(emb, 0.3, acsd)
    for iii in range(10000):
        x = torch.rand((10, 128, 44, 33), device="cuda:0")
        actions_ = torch.randint(high=acsd, size=(10, 128), device="cuda:0")
