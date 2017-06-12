#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import skimage.exposure
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
        self.multipoints = nn.ModuleList(
            [MultiPoint(512 if i == 0 else 1024) for i in range(3)])
        self.l2_norm = L2Norm(512, 20)

    def forward(self, x):
        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)
        xs = [self.l2_norm(x)]
        x = F.max_pool2d(x, kernel_size=2, stride=1, padding=1)[:, :, :-1, :-1]
        # apply the rest of vgg
        for k in range(24, 30):  # skip maxpool - applied above
            x = self.vgg[k](x)
        xs.append(x)
        x = F.max_pool2d(x, kernel_size=2, stride=1, padding=1)[:, :, :-1, :-1]
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        xs.append(x)
        assert len(xs) == len(self.multipoints)
        loc_preds, conf_preds = [], []
        for i, (mp, x) in enumerate(zip(self.multipoints, xs)):
            x = x if i == 0 else torch.cat([x, xs[i - 1]], 1)
            loc, conf = mp(x)
            loc_preds.append(loc)
            conf_preds.append(conf)
        return loc_preds, conf_preds


class SSPDLoss:
    def __init__(self):
        self.loc_loss = nn.MSELoss(size_average=False)
        self.conf_loss = nn.CrossEntropyLoss()
        non_bg_weight = torch.ones(utils.N_CLASSES + 1)
        non_bg_weight[utils.N_CLASSES] = 0.1
        non_bg_weight = non_bg_weight.cuda()
        self.non_bg_conf_loss = nn.CrossEntropyLoss(
            size_average=False, weight=non_bg_weight)

    def __call__(self, outputs, targets):
        loc_preds, conf_preds = outputs
        loc_target, conf_target = targets
        non_bg = conf_target != utils.N_CLASSES
        n_non_bg = int(non_bg.sum().data[0])
        if n_non_bg == 0:
            loc_loss = non_bg_conf_loss = 0
        else:
            non_bg_loc = non_bg.unsqueeze(1).expand_as(loc_preds[0]).float()
            # do not need loc predictions from non-bg classes
            loc_loss = sum(self.loc_loss(loc_pred * non_bg_loc, loc_target)
                           for loc_pred in loc_preds) / n_non_bg
            non_bg_conf_loss = sum(self.non_bg_conf_loss(conf_pred, conf_target)
                                   for conf_pred in conf_preds) / n_non_bg
        conf_loss = sum(self.conf_loss(conf_pred, conf_target)
                        for conf_pred in conf_preds)
        return conf_loss + loc_loss + non_bg_conf_loss


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
        scale = 8
        assert patch_size % scale == 0
        s = patch_size // scale
        target_loc = np.zeros((2, s, s), dtype=np.float32)
        target_conf = np.zeros((s, s), dtype=np.int64)
        target_conf[:] = utils.N_CLASSES
        random.shuffle(points)
        for cls, (x, y) in points:
            if 0 <= x < patch_size and 0 <= y < patch_size:
                x, y = x / scale, y / scale
                ix, iy = int(x), int(y)
                target_conf[iy, ix] = cls
                target_loc[0, iy, ix] = x - ix
                target_loc[1, iy, ix] = y - iy
        return (self.transform(patch),
                (torch.from_numpy(target_loc), torch.from_numpy(target_conf)))


def save_predictions(root: Path, n: float, inputs, targets, outputs):
    batch_size = inputs.size(0)
    to_numpy = lambda x: x.data.cpu().numpy()
    inputs_data = to_numpy(inputs).transpose([0, 2, 3, 1])
    target_locs, target_confs = map(to_numpy, targets)
    all_pred_locs = [to_numpy(x) for x in outputs[0]]
    all_pred_confs = [to_numpy(F.softmax(x)) for x in outputs[1]]
    s = target_locs.shape[3]
    patch_size = inputs_data.shape[1]
    nneg = lambda x: max(0, x)
    m = 4
    for i in range(batch_size):
        prefix = str(root.joinpath('{}-{}'.format(str(n).zfill(6), str(i).zfill(2))))
        utils.save_image(
            '{}-input.jpg'.format(prefix),
            skimage.exposure.rescale_intensity(inputs_data[i], out_range=(0, 1)))
        target_img = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
        for ix in range(s):
            for iy in range(s):
                cls = target_confs[i, iy, ix]
                if cls != utils.N_CLASSES:
                    x, y = target_locs[i, :, iy, ix]
                    x, y = [int(patch_size / s * v) for v in [ix + x, iy + y]]
                    target_img[nneg(y - m) : y + m, nneg(x - m) : x + m, :] = (
                        utils.CLS_COLORS[cls])
        utils.save_image('{}-target.jpg'.format(prefix), target_img)
        output_img = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
        for pred_locs, pred_confs in zip(all_pred_locs, all_pred_confs):
            for ix in range(s):
                for iy in range(s):
                    probs = pred_confs[i, :, iy, ix]
                    if probs[utils.N_CLASSES] < 0.99:
                        x, y = pred_locs[i, :, iy, ix]
                        x, y = [int(patch_size / s * v) for v in [ix + x, iy + y]]
                        color = np.zeros(3, dtype=np.float)
                        for cls, prob in enumerate(probs[:utils.N_CLASSES]):
                            color += np.array(utils.CLS_COLORS[cls]) * prob
                        output_img[
                            nneg(y - m) : y + m, nneg(x - m) : x + m, :] += color
        utils.save_image('{}-output.jpg'.format(prefix), np.clip(output_img, 0, 1))


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
                    train_loader=train_loader, valid_loader=valid_loader,
                    save_predictions=save_predictions)

if __name__ == '__main__':
    main()
