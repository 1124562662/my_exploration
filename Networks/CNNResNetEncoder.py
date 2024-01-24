import argparse
import math
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
# from torchvision.models.resnet import
import torch.nn.functional as F
from torchvision.models.resnet import conv3x3

from exploration_on_policy.utils.BatchRenorm import BatchRenorm1d, BatchRenorm2d

magic_num = math.sqrt(2) / 2


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchRenorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.001)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchRenorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = magic_num * out + magic_num * identity
        out = self.relu(out)

        return out


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
        # self.norm_out = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        x = self.inp(x)
        x = self.residual_blocks(x)
        # x = self.norm_out(x)
        return x


class ResnetMlp(nn.Module):
    def __init__(self, num_units, ):
        super().__init__()
        self.L = nn.Linear(num_units, num_units)
        self.bn = BatchRenorm1d(num_features=num_units)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self,
                x,  # (batch,num_units)
                ):
        out = self.L(x)
        out = self.activation(out)
        out = self.bn(x + out)
        return out


def create_resnet_mlps(inp_size, num_hidden, num_units, out_size):
    layers = [nn.Linear(inp_size, num_units), nn.LeakyReLU(0.4)]
    for _ in range(num_hidden - 1):
        layers.append(ResnetMlp(num_units))
    layers.append(nn.Linear(num_units, out_size))
    return nn.Sequential(*layers)


class BaseEncoder(nn.Module):
    def __init__(self, in_channels, action_num, out_size, emb_dim=10, ):
        super().__init__()
        self.in_channels = in_channels
        self.resnet_units = nn.Sequential(
            ResnetUnit(in_channels, 128),
            ResnetUnit(128, 128),
            ResnetUnit(128, 128),
            ResnetUnit(128, 32),
        )
        self.num_units = out_size + 200

        self.mlps1 = create_resnet_mlps(out_size, 5, self.num_units, self.num_units)
        self.idm_head = create_resnet_mlps(self.num_units, 2, 80, emb_dim)
        self.mlps2 = create_resnet_mlps(self.num_units, 10, self.num_units, self.num_units)
        self.out = nn.Linear(self.num_units, emb_dim)

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
        assert x.shape[1] == 1 and len(x.shape) == 4, "BaseEncoder"
        self.train()

        x = self.resnet_units(x)
        x = x.flatten(start_dim=1)
        # print("resnet forward", x.size())
        x = self.mlps1(x)

        hiddens = self.idm_head(x)
        x = self.mlps2(x)
        x = self.out(x)
        return x, hiddens

    # @torch.no_grad()
    # def forward_hidden(self, x,  # (B,C,dim x , dim y)
    #                    ):
    #     assert x.shape[1] == 1 and len(x.shape) == 4, "BaseEncoder"
    #     x = self.resnet_units(x)
    #     x = x.flatten(start_dim=1)
    #     # print("resnet forward", x.size())
    #     x = self.mlps1(x)
    #     hiddens = self.idm_head(x)
    #     return hiddens


if __name__ == "__main__":
    B = 34
    C = 1
    x, y = 44, 66
    emb_dim = 78
    out_size = 96
    action_num = 19
    be = BaseEncoder(C, action_num, out_size, emb_dim).to("cuda:0")

    x = torch.randn((B, C, x, y)).to("cuda:0")
    out = be(x)
    print(out.size())
    out, h = be.forward_train(x)
    print(out.shape, h.shape)
    out, h2 = be.forward_train(x / 2)
    aa = be.action_decoder(torch.cat([h, h2], dim=1))
    print(aa.shape)
