import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from exploration_on_policy.utils.positional_embedding import positionalencoding1d, positionalencoding2d


def create_mlp(inp_size, num_hidden, num_units, out_size):
    layers = [nn.Linear(inp_size, num_units), nn.ReLU()]
    for _ in range(num_hidden - 1):
        layers.append(nn.Linear(num_units, num_units))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(num_units, out_size))
    return nn.Sequential(*layers)


class UcbCountOneNet(nn.Module):
    def __init__(self, device,
                 args: argparse.Namespace,
                 action_num: int,
                 dim_x: int,
                 dim_y: int,
                 embed_dim: int,
                 action_embeddings_sin: nn.Embedding = None,  # pass from target
                 action_embeddings_randn: nn.Embedding = None,  # pass from target
                 target: bool = False,
                 filter_size: int = 40,
                 ae_sin_size=128,
                 hidden_units=512,
                 ):
        super(UcbCountOneNet, self).__init__()
        in_dim = 3834  # TODO
        self.action_num = action_num  # env.action_space.n
        self.embed_dim = embed_dim
        self.target = target
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.filter_size = filter_size
        self.ae_sin_size = ae_sin_size
        # depthwise convolution
        self.conv = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=filter_size,
                                            kernel_size=3, stride=1, padding=1, ),
                                  nn.LeakyReLU(0.3),
                                  nn.Conv2d(in_channels=filter_size, out_channels=filter_size,
                                            kernel_size=3, stride=1, padding=1, ),
                                  nn.LeakyReLU(0.3),
                                  nn.Conv2d(in_channels=filter_size, out_channels=1,
                                            kernel_size=2, stride=1, padding=1, ),
                                  nn.LeakyReLU(0.3),
                                  )

        num_hidden = 2
        if not target:
            num_hidden += 2
        self.models = create_mlp(in_dim + ae_sin_size, num_hidden, hidden_units, embed_dim)

        if not target:
            self.optimizer = optim.Adam(self.parameters(), lr=args.pseudo_ucb_optimizer_lr)
            self.action_embeddings_sin = action_embeddings_sin.to(device)
            self.action_embeddings_randn = action_embeddings_randn.to(device)
        else:
            action1 = 2 * positionalencoding1d(ae_sin_size, self.action_num).to(device)  # (actions, 200)
            self.action_embeddings_sin = nn.Embedding.from_pretrained(action1).to(device)

            amend = self.action_num % 4
            action2 = (positionalencoding2d(self.action_num + amend, dim_x, dim_y)[amend:,:,:]).reshape((self.action_num,-1)).to(device)
            self.action_embeddings_randn = nn.Embedding.from_pretrained( action2).to(device)

    @torch.no_grad()
    def forward(self,
                obs,  # (envs, dim_x,dim_y)
                ):
        obs = obs.detach().to(torch.float32)
        envs = obs.size(0)
        device = obs.device
        obs = obs.unsqueeze(1).expand(-1, self.action_num, -1, -1)  # (envs, action channels,  dim_x,dim_y)
        ae_randn = self.action_embeddings_randn(torch.arange(self.action_num).to(device)).reshape(
            (-1, self.dim_x, self.dim_y))  # (actions, dim x, dim y)
        ae_randn = ae_randn.unsqueeze(0).expand(envs, -1, -1, -1)  # (env, actions, dim x, dim y)
        ae_randn = ae_randn.detach()
        obs = torch.cat((obs.unsqueeze(2), ae_randn.unsqueeze(2)), dim=2)  # (envs, action channels, 2, dim_x,dim_y)
        obs = obs.reshape((-1, 2, self.dim_x, self.dim_y))  # (envs * action channels, 2, dim_x,dim_y)

        y = self.conv(obs).squeeze().reshape((envs, self.action_num, -1))  # (envs, action channels, in_dim)

        # print(y.size(),"im_dim is", y.size(2))

        ae_sin = self.action_embeddings_sin(torch.arange(self.action_num).to(device)).unsqueeze(0).expand(envs, -1,
                                                                                                          -1)  # (envs, action channels, ae_sin size)
        ae_sin = ae_sin.detach()
        y = torch.cat((y, ae_sin), dim=2)  # (envs, action channels, in_dim + ae_sin size)
        res = self.models(y)  # (envs, action channels, embeds)
        return res  # (env,action_nums,embed_dim)

    def forward_with_action_indices(self,
                                    obs,  # (N, dim x, dim y)
                                    action_indices,  # (N,)
                                    ):
        N = obs.size(0)
        ae_randn = self.action_embeddings_randn(action_indices).reshape(
            (-1, self.dim_x, self.dim_y))  # (N, dim x, dim y)
        ae_randn = ae_randn.detach()
        obs = torch.cat((obs.unsqueeze(1), ae_randn.unsqueeze(1)), dim=1)  # (N, 2, dim_x,dim_y)
        y = self.conv(obs).squeeze().reshape((N, -1))  # (N, in_dim)
        ae_sin = self.action_embeddings_sin(action_indices)  # (N, ae_sin size)
        ae_sin = ae_sin.detach()
        y = torch.cat((y, ae_sin), dim=1).to(obs.device)  # (N, in_dim + ae_sin size)
        res = self.models(y)  # (N , embeds)
        return res  # (N , embeds)

    def train_RND(self, device, args,
                  mb_inds,  # (mini_batch_size,)
                  b_obs,  # (N, dim_x,dim_y)
                  b_actions,  # (N,)
                  t_embs=None,  # (N , embeds)
                  pseudo_ucb_target: nn.Module = None, ):
        if t_embs is None and pseudo_ucb_target is not None:
            with torch.no_grad():
                t_embs = pseudo_ucb_target.forward_with_action_indices(b_obs, b_actions)  # (N , embeds)
        for i in range(args.minibatch_size):
            y_embs = self.forward_with_action_indices(b_obs[mb_inds, :, :], b_actions[mb_inds])  # (M , embeds)
            t_embs_view = t_embs[mb_inds, :]  # (M , embeds)
            loss = F.mse_loss(y_embs, t_embs_view.detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.pseudo_ucb_optimizer_lr = 0.01
    args.minibatch_size = 100
    actions = 6
    dim_x = 33
    dim_y = 37
    embed_dim = 7
    envs = 15
    in_dim = 1292
    device = torch.device("cuda:0")
    target = UcbCountOneNet(device, args, actions, dim_x, dim_y, embed_dim, None, None, True)
    nets = [UcbCountOneNet(device, args, actions, dim_x, dim_y, embed_dim, target.action_embeddings_sin,
                           target.action_embeddings_randn, False) for i in range(14)]

    obs = torch.ones((envs, dim_x, dim_y)).to(device)  # (envs, dim_x,dim_y)

    if torch.cuda.is_available():

        streams = [torch.cuda.Stream() for _ in nets]
        res = []
        for stream in streams:
            stream.synchronize()
        for idx, (module, stream) in enumerate(zip(nets, streams)):
            module.to(device)
            obs.to(device)
            with torch.cuda.stream(stream):
                output = module(obs)
                res.append(output)

        for stream in streams:
            stream.synchronize()
        print(torch.cuda.memory_summary(device))
        # print(res)
    else:
        pass
    # res = net1(obs)  # (env,action_nums,embed_dim)
    # print(res.size())

    torch.cuda.empty_cache()

    mbs = 27
    N = 177
    mb_inds = torch.randint(high=N, size=(mbs,))  # (mini_batch_size,)
    b_obs = torch.rand((N, dim_x, dim_y))  # (N, dim_x,dim_y)
    b_actions = torch.randint(high=actions, size=(N,))  # (N,)
    #
    # net1.train_RND("cpu", args, mb_inds, b_obs, b_actions, target)
    if torch.cuda.is_available():

        streams = [torch.cuda.Stream() for _ in nets]
        res = []
        for stream in streams:
            stream.synchronize()
        for module, stream in zip(nets, streams):
            module = module.to(device)
            mb_inds = mb_inds.to(device)
            b_obs = b_obs.to(device)
            b_actions = b_actions.to(device)
            target = target.to(device)
            with torch.cuda.stream(stream):
                module.train_RND(device, args, mb_inds, b_obs, b_actions, pseudo_ucb_target=target)

        for stream in streams:
            stream.synchronize()
        print(torch.cuda.memory_summary(device))
        # print(res)
