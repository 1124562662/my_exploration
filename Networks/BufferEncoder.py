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
from CNNResNetEncoder import BaseEncoder


class EMA:
    def __init__(self, beta, target_model, online_model):
        super().__init__()
        self.beta = beta
        self.target_model = target_model
        self.online_model = online_model

    def _update_average(self, old, new,beta):
        if old is None:
            return new
        return old * beta + (1 - beta) * new

    def update_moving_average(self,beta=None):
        if beta is None:
            beta = self.beta
        for current_params, ma_params in zip(self.online_model.parameters(), self.target_model.parameters()):
            old_weight, up_weight = self.target_model.data, self.online_model.data
            ma_params.data = self._update_average(old_weight, up_weight,beta)


class RNDBufferEncoder(nn.Module):
    def __init__(self, emb_dim: int, ema_beta: float, action_num: int, encoder_learning_rate: float = 0.001):
        super().__init__()
        self.encoder = BaseEncoder(in_channels=1, action_num=action_num, out_size=192,
                                   emb_dim=emb_dim)  # TODO -- 如何保证这个是准确的？在最最一开始就训练encoder
        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=encoder_learning_rate)

        self.encoder_target = copy.deepcopy(self.encoder)
        for p in self.encoder_target.parameters():
            p.requires_grad = False

        self.encoder_ema = EMA(ema_beta, self.encoder_target, self.encoder)

    def forward(self, x):
        return self.encoder(x)

    def train_encoder(self,
                      states: torch.Tensor,  # (B,rollout, dim x, dim y)
                      actions: torch.Tensor,  # (B,rollout,
                      # done: torch.Tensor,
                      intrinsic_rewards: torch.Tensor = None,  # sample based on this
                      minibatch_num: int = 20,
                      epochs: int = 1,
                      K: int = 6,
                      ):
        # 对比学习，IDM model
        batch_size = states.size(0) - 1

        obs_a = states[:,-1, :, :]
        obs_b = states[:,1:, :, :]

        minibatch_size = math.ceil(batch_size / minibatch_num)

        b_inds = np.arange(batch_size)
        for epoch in range(epochs):
            np.random.shuffle(b_inds)
            for i in range(minibatch_num):
                start = minibatch_size * i
                end = start + minibatch_size
                end = end if end <= batch_size else batch_size
                mb_inds_a = b_inds[start:end]
                mb_inds_b = mb_inds_a + 1

                self._train_contrasive(obs_a[:,mb_inds_a,:,:], obs_b[:,mb_inds_b,:,:], actions[:,mb_inds_a], K, reverse=False)
                self._train_contrasive(obs_b[:,mb_inds_b,:,:], obs_a[:,mb_inds_a,:,:], actions[:,mb_inds_a], K, reverse=True)




        # todo additionally train encoder on high-rnd-valued states
        self.encoder_ema.update_moving_average()



    def _train_contrasive(self,
                          view_a: torch.Tensor,  # (batch, mini rollout, dim x, dim y), query
                          view_b: torch.Tensor,  # (batch, mini rollout, dim x, dim y), key
                          actions: torch.Tensor,  # (batch, mini rollout)
                          K: int,
                          theta: float = 1e-7,
                          tau: float = 1.3,
                          reverse: bool = False,
                          ):
        mb_size,mini_rollout,dim_x,dim_y = view_a.shape
        view_a = view_a.reshape(-1,dim_x, dim_y)
        view_b = view_b.reshape(-1, dim_x, dim_y)

        query, action_emb_a = self.encoder.forward_train(view_a)  # (batch * mini rollout, emb),(batch * mini rollout, emb)
        query = torch.nn.functional.normalize(query, dim=1)  # (batch * mini rollout, emb)

        pos_key, action_emb_b = self.encoder.forward_train(view_b) # (batch * mini rollout, emb)
        pos_key = torch.nn.functional.normalize(pos_key, dim=1).detach()  # (batch * mini rollout, emb)

        with torch.no_grad():
            neg_indices = torch.randint(0, high=self.buffer_size, size=(K,))
            neg_key = self.observations.reshape((-1, self.dim_x, self.dim_y))[neg_indices, :, :]  # (K, dim x, dim y)
            neg_key = torch.nn.functional.normalize(self.encoder_target(neg_key), dim=1).detach()  # (K, emb)

        positive = torch.sigmoid(torch.mul(query, pos_key).mean(1) / tau)  # (batch * mini rollout)
        pos_key = pos_key.unsqueeze(1).expand(-1, K, -1)  # (batch * mini rollout, K, emb)
        neg_key = neg_key.unsqueeze(0).expand(mb_size * mini_rollout, -1, -1)  # (batch * mini rollout, K, emb)

        negative = torch.sigmoid(torch.mul(pos_key, neg_key) / tau).sum(2).sum(1)  # (batch * mini rollout)
        negative += positive
        negative /= (K + 1)
        loss = -torch.log(torch.div(positive, negative + theta))
        loss = loss.mean()

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
