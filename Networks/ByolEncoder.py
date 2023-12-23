import argparse
import math
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.models.resnet import conv3x3
import torch.nn.functional as F


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

        out += identity
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

        # print(self.norm_out.training,[ (a,b) for (a,b) in enumerate(self.norm_out.named_parameters())])
        return x


class BYOLEncoder(nn.Module):
    def __init__(self, in_channels, out_size, emb_dim=10):
        super().__init__()
        self.in_channels = in_channels
        self.resnet_units = nn.Sequential(
            ResnetUnit(in_channels, 32),
            ResnetUnit(32, 32),
            ResnetUnit(32, 32),
            ResnetUnit(32, 32),
            # ResnetUnit(32, 32)
        )

        self.out = nn.Linear(out_size, emb_dim)

    def forward(self, x,  # (B,C, dim x, dim y)
                ):
        assert x.shape[1] == 1 and len(x.shape) == 4, "BYOL Encoder"
        # if self.in_channels == 1 and len(x.size()) == 3:
        #     x = x.unsqueeze(1)

        x = self.resnet_units(x)
        x = x.flatten(start_dim=1)
        # print("resnet forward", x.size())
        x = self.out(x)
        return x  # (B, out emb size)
