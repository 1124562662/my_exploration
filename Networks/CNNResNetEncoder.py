import argparse
import math
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.models.resnet import BasicBlock
import torch.nn.functional as F


class ResnetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding="same"),
            nn.MaxPool2d((3, 3), stride=2),
        )
        self.residual_blocks = nn.Sequential(
            BasicBlock(out_channels, out_channels),
            BasicBlock(out_channels, out_channels),
        )
        self.norm_out = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        x = self.inp(x)
        x = self.residual_blocks(x)
        x = self.norm_out(x)
        return x


class ResnetMlp(nn.Module):
    def __init__(self, num_units, ):
        super().__init__()
        self.L = nn.Linear(num_units, num_units)
        self.activation = nn.LeakyReLU(0.1)
        self.bn = torch.nn.BatchNorm1d(num_units)

    def forward(self,
                x,  # (batch,num_units)
                ):
        out = self.L(x)
        out = self.activation(out)
        out = self.bn(x + out)
        return out


def create_resnet_mlps(inp_size, num_hidden, num_units, out_size):
    layers = [nn.Linear(inp_size, num_units), nn.LeakyReLU(0.1)]
    for _ in range(num_hidden - 1):
        layers.append(ResnetMlp(num_units))
    layers.append(nn.Linear(num_units, out_size))
    return nn.Sequential(*layers)


class BaseEncoder(nn.Module):
    def __init__(self, in_channels, action_num, out_size, emb_dim=10, ):
        super().__init__()
        self.in_channels = in_channels
        self.resnet_units = nn.Sequential(
            ResnetUnit(in_channels, 32),
            ResnetUnit(32, 32),
            ResnetUnit(32, 32),
            ResnetUnit(32, 32),
        )

        self.mlps1 = create_resnet_mlps(out_size, 2, out_size, out_size)
        self.idm_head = create_resnet_mlps(out_size, 1, 80, emb_dim)
        self.mlps2 = create_resnet_mlps(out_size, 2, out_size, out_size)
        self.out = nn.Linear(out_size, emb_dim)

        self.action_decoder = nn.Linear(emb_dim * 2, action_num)

    def forward(self, x,  # (B,C,dim x , dim y)
                ):
        assert x.shape[1] == 1 and len(x.shape) == 4, "BaseEncoder"
        # if self.in_channels == 1 and len(x.size()) == 3:
        #     x = x.unsqueeze(1)
        x = self.resnet_units(x)
        x = x.flatten(start_dim=1)
        # print("resnet forward", x.size())
        x = self.mlps1(x)
        x = self.mlps2(x)
        x = self.out(x)
        return x

    def forward_train(self, x,  # (B,C,dim x , dim y)
                      ):
        if self.in_channels == 1 and len(x.size()) == 3:
            x = x.unsqueeze(1)
        x = self.resnet_units(x)
        x = x.flatten(start_dim=1)
        # print("resnet forward", x.size())
        x = self.mlps1(x)
        hiddens = self.idm_head(x)
        x = self.mlps2(x)
        x = self.out(x)
        return x, hiddens


if __name__ == "__main__":
    B = 34
    C = 1
    x, y = 44, 66
    emb_dim = 17
    out_size = 96
    action_num = 19
    be = BaseEncoder(C, action_num, out_size, emb_dim)

    x = torch.randn((B, C, x, y))
    out = be(x)
    print(out.size())
    out, h = be.forward_train(x)
    print(out.shape, h.shape)
    out, h2 = be.forward_train(x / 2)
    aa = be.action_decoder(torch.cat([h, h2], dim=1))
    print(aa.shape)
