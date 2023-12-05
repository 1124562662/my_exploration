import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, env, add_one=False):
        super().__init__()
        # print(dir(env))
        last_dim = env.action_space.n
        if add_one:
            last_dim += 1
        self.conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(0.7),
                                  nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=1, padding=1),
                                  nn.LeakyReLU(0.7),
                                  )
        self.network1 = nn.Sequential(
            nn.Linear(7668, 520),
            nn.LeakyReLU(),
            nn.Linear(520, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 200),
        )
        self.network2 = nn.Sequential(
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 200),
        )
        self.decoder = nn.Sequential(nn.Linear(200, last_dim))
        device = torch.device("cuda" if torch.cuda.is_available()   else "cpu")
        self.magical_number = (torch.sqrt(torch.tensor(2)) / 2).to(device)

    def forward(self, x):
        x = torch.tensor(x).to(torch.float32)
        x = x.view((x.size(0), 1, x.size(1), -1))
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x_ = self.network1(x)
        x2 = self.magical_number * self.network2(x_) + self.magical_number * x_
        return self.decoder(x2)  # (env,action_nums)