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
                 ):
        super(UcbNets, self).__init__()
        self.embed_dim = args.embed_dim
        self.net_num = net_num
        self.target = UcbCountOneNet(device, args, action_num, dim_x, dim_y, self.embed_dim, None, None,True, filter_size,
                                     ae_sin_size, hidden_units).to(device)
        self.nets = [
            UcbCountOneNet(device, args, action_num, dim_x, dim_y, self.embed_dim, self.target.action_embeddings_sin,
                           self.target.action_embeddings_randn, False, filter_size, ae_sin_size, hidden_units).to(device) for i in
            range(net_num)]

    @torch.no_grad()
    def forward(self,
                obs,  # (envs, dim_x,dim_y)
                ):

        if torch.cuda.is_available():
            device = "cuda:0"
            streams = [torch.cuda.Stream() for _ in self.nets]
            res = [None for i in range(len(self.nets))]
            for stream in streams:
                stream.synchronize()
            for idx, (module, stream) in enumerate(zip(self.nets, streams)):
                module.to(device)
                obs.to(device)
                with torch.cuda.stream(stream):
                    res[idx] = module(obs)

            for stream in streams:
                stream.synchronize()

            res = torch.stack(res)  # (net_num,env,action_nums,embed_dim)
            self.target.to(device)
            tar = self.target(obs).unsqueeze(0).expand(self.net_num, -1, -1, -1)  # (net_num ,env,action_nums,embed_dim)
            loss = F.mse_loss(res, tar, reduction='none').mean(0).mean(2)  # (env,action_nums)
            return loss  # (env,action_nums)
        else:
            raise NotImplementedError("cuda only")

    def train_RNDs(self, args,
                   mb_inds,  # (mini_batch_size,)
                   b_obs,  # (N, dim_x,dim_y)
                   b_actions,  # (N,)
                   ):

        if torch.cuda.is_available():
            device = "cuda:0"
            streams = [torch.cuda.Stream() for _ in self.nets]
            with torch.no_grad():
                t_embs = self.target.forward_with_action_indices(b_obs, b_actions).to(device)  # (N , embeds)
            for stream in streams:
                stream.synchronize()
            for module, stream in zip(self.nets, streams):
                # module = module.to(device)
                # mb_inds = mb_inds.to(device)
                # b_obs = b_obs.to(device)
                # b_actions = b_actions.to(device)
                with torch.cuda.stream(stream):
                    module.train_RND(device, args, mb_inds, b_obs, b_actions, t_embs=t_embs)
            for stream in streams:
                stream.synchronize()
        else:
            raise NotImplementedError("cuda only")

# if __name__ == "__main__":
