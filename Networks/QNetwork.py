import argparse
import math
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models.resnet import conv3x3

magic_num = math.sqrt(2)/2
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
        # self.bn1 = norm_layer(planes)
        self.Lrelu = nn.LeakyReLU(0.1,inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.Lrelu(out)
        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = magic_num*identity + magic_num*out
        out = self.Lrelu(out)

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
        self.activation = nn.LeakyReLU(0.1)
        # self.bn = torch.nn.BatchNorm1d(num_units)

    def forward(self,
                x,  # (batch,num_units)
                ):
        out = self.L(x)
        out = self.activation(out)
        out = magic_num * x + magic_num * out
        return out

def create_resnet_mlps(inp_size, num_hidden, num_units, out_size):
    layers = [nn.Linear(inp_size, num_units), nn.LeakyReLU(0.1)]
    for _ in range(num_hidden - 1):
        layers.append(ResnetMlp(num_units))
    layers.append(nn.Linear(num_units, out_size))
    return nn.Sequential(*layers)

###########################################################################################################
class QNetwork(nn.Module):
    def __init__(self,device,
                 last_dim,
                 out_size,
                 in_channels=1):
        super().__init__()
        # print(dir(env))
        self.device = device
        self.last_dim =last_dim #1, not env.action_space.n
        self.in_channels = in_channels
        self.resnet_units = nn.Sequential(
            ResnetUnit(in_channels, 4),
            ResnetUnit(4, 8),
            ResnetUnit(8, 16),
        )

        self.out = create_resnet_mlps(out_size,3,100,last_dim)

    def forward(self,
                x, #(B,C,dim x , dim y)
                ):
        assert len(x.shape) == 4 and x.shape[1] == 1
        x = x.to(torch.float32).to(self.device)
        x = self.resnet_units(x)
        x = x.flatten(start_dim=1)
        # print("ex q resnet forward", x.size())
        x = self.out(x)
        return x # (B, out emb size)