import torch
from torch import nn
from torch.nn import functional as F

from utils import N_CLASSES


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


def concat(xs):
    return torch.cat(xs, 1)


class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class UNetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class UNet(nn.Module):
    module = UNetModule
    filter_factors = [1, 2, 4, 8, 16]

    def __init__(self, filters_base: int=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        filter_sizes = [filters_base * s for s in self.filter_factors]
        self.down, self.up = [], []
        for i, nf in enumerate(filter_sizes):
            low_nf = 3 if i == 0 else filter_sizes[i - 1]
            self.down.append(self.module(low_nf, nf))
            setattr(self, 'down_{}'.format(i), self.down[-1])
            if i != 0:
                self.up.append(self.module(low_nf + nf, low_nf))
                setattr(self, 'conv_up_{}'.format(i), self.up[-1])
        self.conv_final = nn.Conv2d(filter_sizes[0], N_CLASSES, 1)

    def forward(self, x):
        xs = []
        for i, down in enumerate(self.down):
            x_in = x if i == 0 else self.pool(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        for i, (x_skip, up) in reversed(list(enumerate(zip(xs[:-1], self.up)))):
            x_out = up(concat([self.upsample(x_out), x_skip]))

        x_out = self.conv_final(x_out)
        return F.log_softmax(x_out)
