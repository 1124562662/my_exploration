import math
import torch


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe = 0.7 * pe + 0.3 * torch.randn_like(pe)
    pe[torch.bitwise_and(pe < 0.01, pe > -0.01)] = 0.01
    return pe


def positionalencoding2d_t(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))

    pos_w = torch.randint(low=-width, high=width, size=(height, width))
    pos_h = torch.randint(low=-height, high=height, size=(height, width))

    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term)
    pe[torch.bitwise_and(pe < 0.1, pe > -0.1)] = 0.01
    return pe


if __name__ == "__main__":
    ww = positionalencoding2d(13, 7, 5)
    print(ww)
    print(torch.var(ww, dim=0))
    print(torch.var(torch.randn(4, 7, 5), dim=0))
    print("==========")
    # print(positionalencoding2d(4, 7, 5).permute())
    # print(positionalencoding2d(4, 7, 5).size())
