#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F, init
from torchvision.models import vgg16

import utils


class SSPD(nn.Module):
    """ Single Shot Point Detector.
    """
    def __init__(self):
        super().__init__()
        self.vgg = vgg16(pretrained=True).features
        self.conv1_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.multipoints = [MultiPoint(512) for _ in range(3)]
        self.l2_norm = L2Norm(512, 20)

    def forward(self, x):
        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)
        xs = [self.l2_norm(x)]
        # apply the rest of vgg
        for k in range(24, 31):
            x = self.vgg[k](x)
        xs.append(x)
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, kernel_size=2)
        xs.append(x)
        assert len(xs) == len(self.multipoints)
        loc_preds, conf_preds = [], []
        for mp, x in zip(self.multipoints, xs):
            loc, conf = mp(x)
            loc_preds.append(loc)
            conf_preds.append(conf)
        return loc_preds, conf_preds


class SSPDLoss:
    def __init__(self):
        self.loc_loss = nn.MSELoss()
        self.conf_loss = nn.CrossEntropyLoss()

    def __call__(self, outputs, targets):
        loc_preds, conf_preds = outputs
        loc_targets, conf_targets = targets
        assert len(loc_preds) == len(loc_targets)
        assert len(conf_preds) == len(conf_targets)
        loc_loss = sum(self.loc_loss(loc_pred, loc_target)
                       for loc_pred, loc_target in zip(loc_preds, loc_targets))
        conf_loss = sum(self.conf_loss(conf_pred, conf_target)
                        for conf_pred, conf_target in zip(conf_preds, conf_targets))
        return loc_loss + conf_loss


class MultiPoint(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.loc_conv = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)
        self.conf_conv = nn.Conv2d(
            in_channels, utils.N_CLASSES + 1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.loc_conv(x), self.conf_conv(x)


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(1).sqrt()+self.eps
        x /= norm.expand_as(x)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class PointDataset(utils.BasePatchDataset):
    def new_x_y(self, patch, points):
        """ Sample (x, y) pair.
        """
        patch_size = patch.shape[0]
        target_locs = []
        target_confs = []
        for scale in [8, 16, 32]:
            assert patch_size % scale == 0
            s = patch_size // scale
            target_loc = np.zeros((2, s, s), dtype=np.float32)
            target_conf = np.zeros((s, s), dtype=np.int64)
            target_conf[:] = utils.N_CLASSES
            for cls, (x, y) in points:
                if 0 <= x < s and 0 <= y < s:
                    x, y = x / scale, y / scale
                    ix, iy = x // s, y // s
                    target_conf[iy, ix] = cls
                    target_loc[0, iy, ix] = x - ix
                    target_loc[1, iy, ix] = y - iy
            target_locs.append(target_loc)
            target_confs.append(target_conf)
        return (self.transform(patch),
                ([torch.from_numpy(tl) for tl in target_locs],
                 [torch.from_numpy(tc) for tc in target_confs]))


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('root', help='checkpoint root')
    arg('--batch-size', type=int, default=32)
    arg('--patch-size', type=int, default=256)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=2)
    arg('--fold', type=int, default=1)
    arg('--n-folds', type=int, default=5)
    arg('--stratified', action='store_true')
    arg('--mode', choices=[
        'train', 'validation', 'predict_valid', 'predict_test', 'predict_all_valid'],
        default='train')
    arg('--clean', action='store_true')
    arg('--epoch-size', type=int)
    arg('--limit', type=int, help='Use only N images for train/valid')
    arg('--min-scale', type=float, default=1)
    arg('--max-scale', type=float, default=1)
    arg('--test-scale', type=float, default=0.5)
    args = parser.parse_args()

    coords = utils.load_coords()
    train_paths, valid_paths = utils.train_valid_split(args, coords)
    root = Path(args.root)
    model = SSPD()
    model = utils.cuda(model)
    criterion = SSPDLoss()

    if args.mode == 'train':
        kwargs = dict(min_scale=args.min_scale, max_scale=args.max_scale)
        train_loader, valid_loader = (
            utils.make_loader(
                PointDataset, args, train_paths, coords, **kwargs),
            utils.make_loader(
                PointDataset, args, valid_paths, coords,
                deterministic=True, **kwargs))
        if root.exists() and args.clean:
            shutil.rmtree(str(root))
        root.mkdir(exist_ok=True)
        root.joinpath('params.json').write_text(
            json.dumps(vars(args), indent=True, sort_keys=True))
        utils.train(args, model, criterion,
                    train_loader=train_loader, valid_loader=valid_loader)

if __name__ == '__main__':
    main()
