import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from TensorLinear import create_3d_mlp

class UcbCountParallelNet(nn.Module):
    def __init__(self, args:argparse.Namespace, env, embed_dim: int, target: bool = False,
                 filter_size:int =5,
                 ):
        super(UcbCountParallelNet, self).__init__()
        in_dim = 9999 #TODO
        self.action_num = env.action_space.n
        self.embed_dim = embed_dim
        self.target = target
        self.filter_size = filter_size
        # depthwise convolution
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=self.action_num, out_channels= filter_size * self.action_num,
                                             kernel_size=3, stride=1, padding=1,groups=self.action_num),
                                  nn.LeakyReLU(0.5),
                                  )
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels= filter_size * self.action_num, out_channels= self.action_num,
                                             kernel_size=3, stride=1, padding=1, groups= self.action_num),
                                   nn.LeakyReLU(0.5),
                                   )
        num_hidden = 1
        if not target:
            num_hidden += 1
            self.optimizer =  optim.Adam(self.pseudo_ucb_net.parameters(), lr=args.pseudo_ucb_optimizer_lr)
        self.models = create_3d_mlp(in_dim,num_hidden,500,embed_dim,self.action_num) #input size is (channel,in_dim,*)


    def forward(self,
                obs, # (envs, dim_x,dim_y)
                ):
        obs = torch.tensor(obs).to(torch.float32)
        obs = obs.unsqueeze(1).expand(-1,self.action_num,-1,-1) # (envs, action channels,  dim_x,dim_y)
        obs = self.conv1(obs) # (envs, action channels*filter_size,  h,w)
        obs = self.conv2(obs) # (envs, action channels,  h,w)
        obs = obs.reshape((obs.size(0), self.action_num, -1)) # (envs, action channels,  in_dim)
        obs = obs.permute(1,2,0)  # ( action channels,  in_dim,  envs)
        res = self.models(obs) # (channel --action num,embed_dim, envs)
        res = res.permute(2,0,1) # (env,action_nums,embed_dim)
        return res  # (env,action_nums,embed_dim)

    def train_RND(self,device,args,
                  mb_inds, #(mini_batch_size,)
                  b_obs, #(N, dim_x,dim_y)
                  b_actions, #(N,)
                  pseudo_ucb_target:nn.Module,):
        N = b_actions.size(0)
        action_bind = b_actions + torch.arange(0,end=N*self.action_num,step=N)

        for i in range(args.minibatch_size):
            y_i = self(b_obs[mb_inds]).reshape((-1,self.embed_dim))  # (N * action_nums,embed_dim)
            y_i = y_i[action_bind].clone()
            with torch.no_grad():
                t_i = self(b_obs[mb_inds]).reshape((-1, self.embed_dim))  # (N * action_nums,embed_dim)
                t_i = t_i[action_bind].clone()
            loss = F.mse_loss(y_i,t_i.detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()