import argparse


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ACNetwork(nn.Module):
    def __init__(self,device, env, indim, add_one=False):
        super().__init__()
        # print(dir(env))
        last_dim = env.action_space.n
        if add_one:
            last_dim += 1
        # self.conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1),
        #                           nn.LeakyReLU(0.7),
        #                           nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=1, padding=1),
        #                           nn.LeakyReLU(0.7),
        #                           )
        # 7668 indim
        self.network1 = nn.Sequential(
            nn.Linear(indim, 520),
            nn.LeakyReLU(),
            nn.Linear(520, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 200),
        )
        self.network2 = nn.Sequential(
            nn.Linear(200, 200),
            nn.LeakyReLU(),
        )
        self.critic = nn.Sequential(
            nn.Linear(200, 200),
            nn.LeakyReLU(0.5),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.6),
            nn.Linear(200, 1))
        self.actor = nn.Sequential(nn.Linear(200, 200),
                                   nn.LeakyReLU(0.3),
                                   nn.Linear(200, 200),
                                   nn.LeakyReLU(0.3),
                                   nn.Linear(200, 200),
                                   nn.LeakyReLU(0.5),
                                   nn.Linear(200, last_dim),
                                   )


        self.magical_number = (torch.sqrt(torch.tensor(2)) / 2).to(device)

    def _get_feature(self, x):
        x = x.to(torch.float32)
        # x = x.view((x.size(0), 1, x.size(1), -1))
        # x = self.conv(x)
        # x = x.view(x.size(0), -1)
        x_ = self.network1(x)
        x2 = self.magical_number * self.network2(x_) + self.magical_number * x_
        return x2

    def forward(self, x):
        x2 = self._get_feature(x)
        return self.actor(x2), self.critic(x2)

    def get_action(self, x):
        x2 = self._get_feature(x)
        return self.actor(x2)  # (env,action_nums)

    def get_critic(self, x):
        x2 = self._get_feature(x)
        return self.critic(x2)  # (env )
