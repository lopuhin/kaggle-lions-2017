import torch
from torch import nn
from torch.nn import functional as F

import utils


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

    def __init__(self,
                 input_channels: int=3,
                 filters_base: int=32,
                 filter_factors=(1, 2, 4, 8, 16)):
        super().__init__()
        filter_sizes = [filters_base * s for s in filter_factors]
        self.down, self.up = [], []
        for i, nf in enumerate(filter_sizes):
            low_nf = input_channels if i == 0 else filter_sizes[i - 1]
            self.down.append(self.module(low_nf, nf))
            setattr(self, 'down_{}'.format(i), self.down[-1])
            if i != 0:
                self.up.append(self.module(low_nf + nf, low_nf))
                setattr(self, 'conv_up_{}'.format(i), self.up[-1])
        bottom_s = 4
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.UpsamplingNearest2d(scale_factor=2)
        upsample_bottom = nn.UpsamplingNearest2d(scale_factor=bottom_s)
        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers = [upsample] * len(self.up)
        self.upsamplers[-1] = upsample_bottom
        self.conv_final = nn.Conv2d(filter_sizes[0], utils.N_CLASSES + 1, 1)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        for x_skip, upsample, up in reversed(
                list(zip(xs[:-1], self.upsamplers, self.up))):
            x_out = upsample(x_out)
            x_out = up(concat([x_out, x_skip]))

        x_out = self.conv_final(x_out)
        return F.log_softmax(x_out)


class UNetWithHead(nn.Module):
    filters_base = 32
    unet_filters_base = 128
    unet_filter_factors = [1, 2, 4]

    def __init__(self):
        super().__init__()
        b = self.filters_base
        self.head = nn.Sequential(
            Conv3BN(3, b),
            Conv3BN(b, b),
            nn.MaxPool2d(2, 2),
            Conv3BN(b, b * 2),
            Conv3BN(b * 2, b * 2),
            nn.MaxPool2d(2, 2),
        )
        self.unet = UNet(
            input_channels=64,
            filters_base=self.unet_filters_base,
            filter_factors=self.unet_filter_factors,
        )

    def forward(self, x):
        x = self.head(x)
        return self.unet(x)


class Loss:
    def __init__(self, dice_weight=1.0, bg_weight=1.0):
        if bg_weight != 1.0:
            nll_weight = torch.ones(utils.N_CLASSES + 1)
            nll_weight[utils.N_CLASSES] = bg_weight
            nll_weight = nll_weight.cuda()
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss2d(weight=nll_weight)
        self.dice_weight = dice_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        if self.dice_weight:
            cls_weight = self.dice_weight / utils.N_CLASSES
            eps = 1e-5
            for cls in range(utils.N_CLASSES):
                dice_target = (targets == cls).float()
                dice_output = outputs[:, cls].exp()
                intersection = (dice_output * dice_target).sum()
                # union without intersection
                uwi = dice_output.sum() + dice_target.sum() + eps
                loss += (1 - intersection / uwi) * cls_weight
            loss /= (1 + self.dice_weight)
        return loss
