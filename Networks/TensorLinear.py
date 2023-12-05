import argparse


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TensorLinear(nn.Module):
    def __init__(self, channel, outd, ind,a=0.1):
        super(TensorLinear, self).__init__()
        self.channel = channel
        self.ind = ind
        self.outd = outd
        self.a = a
        self.weight = torch.nn.Parameter(torch.Tensor(channel,outd,ind))
        self.register_parameter("tensorLinear weight", self.weight)
        self.bias = torch.nn.Parameter(torch.Tensor(channel,outd))
        self.register_parameter("tensorLinear bias", self.bias)
        torch.nn.init.kaiming_normal_(self.weight, a=a)
        torch.nn.init.normal_(self.bias)

    def forward(self, x):
        # x size is (channel,in_dim,*)
        suffix_size = x.size()[2:]
        x = x.reshape((self.channel, self.ind,-1)) # (channel,in_dim,batch_size)
        batch_size = x.size(2)
        bias_view = self.bias.unsqueeze(2).expand(-1,-1,batch_size) # (channel,out_dim,batch_size)
        out = torch.baddbmm(bias_view, self.weight, x) # (channel,out_dim,batch_size)
        out = out.reshape((self.channel,self.outd,) + suffix_size) #(channel,out_dim,*)
        return out #(channel,out_dim,*)

    def reset_channel(self,channel_idx:int,):
        torch.nn.init.kaiming_normal_(self.weight[channel_idx], a=self.a)
        torch.nn.init.normal_(self.bias[channel_idx])


def create_3d_mlp(inp_size, num_hidden, num_units, out_size,channels):
    a = 0.1
    layers = [TensorLinear(channel=channels,outd=num_units,ind=inp_size,a=a), nn.LeakyReLU(a)]
    for _ in range(num_hidden - 1):
        layers.append(TensorLinear(channel=channels,outd=num_units,ind=num_units,a=a))
        layers.append(nn.LeakyReLU(a))
    layers.append(TensorLinear(channel=channels,outd=out_size,ind=num_units,a=a))
    return nn.Sequential(*layers)



def test_tensornet():
    channel, outd, ind = 4,2,3
    x = torch.randn((channel,ind,10,11))
    ll = TensorLinear(channel, outd, ind)
    out = ll(x)
    print(out.size())

    print(ll.weight[0])
    ll.reset_channel(0)
    print('--------------------------')
    print(ll.weight[0])
    out2 = ll(x)
    print('--------------------------' )
    print((out[0] - out2[0]).mean())
    print((out[1]-out2[1]).mean())


