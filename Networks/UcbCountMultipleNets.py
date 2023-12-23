import random

from .UcbCountOneNet import UcbCountOneNet
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class UcbNets(nn.Module):
    def __init__(self, device,
                 args: argparse.Namespace,
                 action_num: int,
                 dim_x: int,
                 dim_y: int,
                 net_num: int = 4,
                 filter_size: int = 40,
                 ae_sin_size=128,
                 hidden_units=512,
                 in_dim = 3834,
                 ):
        super(UcbNets, self).__init__()
        self.device=device
        self.embed_dim = args.embed_dim
        self.net_num = net_num
        self.target = UcbCountOneNet(device, args, action_num, dim_x, dim_y, self.embed_dim, None, None, True,
                                     filter_size,
                                     ae_sin_size, hidden_units,in_dim=in_dim).to(device)
        self.nets = [
            UcbCountOneNet(device, args, action_num, dim_x, dim_y, self.embed_dim, self.target.action_embeddings_sin,
                           self.target.action_embeddings_randn, False, filter_size, ae_sin_size, hidden_units, in_dim= in_dim).to(
                device) for i in
            range(net_num)]

    @torch.no_grad()
    def forward(self,
                obs,  # (envs, dim_x,dim_y)
                ):

        if torch.cuda.is_available():
            obs = obs.to(self.device)
            streams = [torch.cuda.Stream() for _ in self.nets]
            res = [None for i in range(len(self.nets))]
            for stream in streams:
                stream.synchronize()
            for idx, (module, stream) in enumerate(zip(self.nets, streams)):
                # module.to(self.device)
                with torch.cuda.stream(stream):
                    res[idx] = module(obs)
            for stream in streams:
                stream.synchronize()

            res = torch.stack(res)  # (net_num,env,action_nums,embed_dim)
            self.target.to(self.device)
            tar = self.target(obs).unsqueeze(0).expand(self.net_num, -1, -1, -1)  # (net_num ,env,action_nums,embed_dim)
            loss = F.mse_loss(res, tar, reduction='none').mean(0).mean(2)  # (env,action_nums)
            return loss  # (env,action_nums)
        else:
            raise NotImplementedError("cuda only")



    def train_RNDs(self, args,
                   mb_inds,  # (mini_batch_size,)
                   b_obs,  # (N, dim_x,dim_y)
                   b_actions,  # (N,)
                   train_net_num: int = 2,
                   ):
        assert train_net_num < self.net_num, "train_net_num<self.net_num"
        b_obs = b_obs.to(self.device)
        b_actions =b_actions.to(self.device)

        train_nets = random.sample(self.nets, train_net_num)
        if torch.cuda.is_available():
            with torch.no_grad():
                t_embs = self.target.forward_with_action_indices(b_obs, b_actions)  # (N , embeds)

            streams = [torch.cuda.Stream() for _ in train_nets]
            for stream in streams:
                stream.synchronize()
            for module, stream in zip(train_nets, streams):
                # module = module.to(device)
                # mb_inds = mb_inds.to(device)
                # b_obs = b_obs.to(device)
                # b_actions = b_actions.to(device)
                with torch.cuda.stream(stream):
                    module.train_RND(args, mb_inds, b_obs, b_actions, t_embs=t_embs)
            for stream in streams:
                stream.synchronize()
        else:
            raise NotImplementedError("cuda only")






    # @torch.no_grad()
    # def forward_with_actions(self,
    #                          b_obs,  # (N, dim_x,dim_y)
    #                          b_actions,  # (N,)
    #                          ):
    #     if torch.cuda.is_available():
    #         device = "cuda:0"
    #         with torch.no_grad():
    #             t_embs = self.target.forward_with_action_indices(b_obs, b_actions).to(device)  # (N , embeds)
    #         streams = [torch.cuda.Stream() for _ in self.nets]
    #         res = torch.zeros((self.net_num, b_obs.shape[0])).to(device)  # (net num, N)
    #         for stream in streams:
    #             stream.synchronize()
    #         for idx, (module, stream) in enumerate(zip(self.nets, streams)):
    #             # module = module.to(device)
    #             # mb_inds = mb_inds.to(device)
    #             # b_obs = b_obs.to(device)
    #             # b_actions = b_actions.to(device)
    #             with torch.cuda.stream(stream):
    #                 y_embs = module.forward_with_action_indices(b_obs, b_actions).to(device)  # (N , embeds)
    #                 res[idx] = F.mse_loss(y_embs, t_embs, reduction='none').mean(1)  # (N,)
    #         for stream in streams:
    #             stream.synchronize()
    #         return res.mean(0)  # (N,)
    #
    #     else:
    #         raise NotImplementedError("cuda only")














# if __name__ == "__main__":
