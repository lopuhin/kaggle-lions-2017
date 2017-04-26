#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import random
import shutil
from typing import List

import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.affinity import rotate
import skimage.transform
import skimage.io
import torch
from torch import nn

from utils import (
    N_CLASSES, cuda, load_coords, train_valid_split, train, validation,
    BaseDataset, make_loader,
)
from unet_models import UNet


class SegmentationDataset(BaseDataset):
    def __init__(self,
                 img_paths: List[Path],
                 coords: pd.DataFrame,
                 transform,
                 size: int,
                 mark_r: int=8,
                 debug: bool=False,
                 deterministic: bool=False,
                 ):
        super().__init__(img_paths, coords)
        self.patch_size = size
        self.mark_r = mark_r
        self.transform = transform
        self.debug = debug
        self.deterministic = deterministic

    def __getitem__(self, idx):
        random.seed(idx if self.deterministic else None)
        img_id = self.img_ids[idx % len(self.img_ids)]
        img = self.imgs[img_id]
        max_y, max_x = img.shape[:2]
        s = self.patch_size
        b = int(np.ceil(np.sqrt(2) * s / 2))
        x, y = (random.randint(b, max_x - (b + s)),
                random.randint(b, max_y - (b + s)))
        patch = img[y: y + 2 * b + s, x: x + 2 * b + s]
        try:
            coords = self.coords.loc[[img_id]]
        except KeyError:
            coords = []
        # TODO - sample a point close to some lion with non-zero prob
        # TODO - check that this loop is not too slow, can be vectorized
        any_lions = False
        angle = random.random() * 360
        patch = skimage.transform.rotate(patch, angle, preserve_range=True)
        mask = np.zeros(patch.shape[:2], dtype=np.int64)
        mask[:] = N_CLASSES
        nneg = lambda x: max(0, x)
        c = b + s // 2
        for i in range(len(coords)):
            item = coords.iloc[i]
            ix, iy = item.col - x, item.row - y
            if (0 <= ix <= 2 * b + s) and (0 <= iy <= 2 * b + s):
                p = rotate(Point(ix, iy), -angle, origin=(c, c))
                ix, iy = int(p.x), int(p.y)
                mask[nneg(iy - self.mark_r): nneg(iy + self.mark_r),
                     nneg(ix - self.mark_r): nneg(ix + self.mark_r)] = item.cls
                any_lions = True
        patch = patch[b:, b:][:s, :s]
        mask = mask[b:, b:][:s, :s]
        if random.random() < 0.5:
            patch = np.flip(patch, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        assert patch.shape == (s, s, 3), patch.shape
        assert mask.shape == (s, s), mask.shape
        if self.debug and any_lions:
            for cls in range(N_CLASSES):
                skimage.io.imsave('mask-{}.jpg'.format(cls),
                                  (mask == cls).astype(np.float32))
            skimage.io.imsave('patch.jpg', patch / 255)
        return self.transform(patch), torch.from_numpy(mask)

    def __len__(self):
        patch_area = self.patch_size ** 2
        return int(sum(img.shape[0] * img.shape[1] / patch_area
                       for img in self.imgs.values()))


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
    arg('--mode', choices=['train', 'validation', 'predict'],
        default='train')
    arg('--clean', action='store_true')
    arg('--epoch-size', type=int)
    arg('--limit', type=int, help='Use only N images for train/valid')
    args = parser.parse_args()

    coords = load_coords()
    train_paths, valid_paths = train_valid_split(args)
    train_loader, valid_loader = (
        make_loader(SegmentationDataset, args, train_paths, coords),
        make_loader(SegmentationDataset, args, valid_paths, coords,
                    deterministic=True))

    root = Path(args.root)
    model = UNet()
    model = cuda(model)
    criterion = nn.NLLLoss2d()
    if args.mode == 'train':
        if root.exists() and args.clean:
            shutil.rmtree(str(root))
        root.mkdir(exist_ok=True)
        root.joinpath('params.json').write_text(json.dumps(vars(args)))
        train(args, model, criterion,
              train_loader=train_loader, valid_loader=valid_loader)
    elif args.mode == 'validation':
        validation(model, criterion, valid_loader)
    elif args.mode == 'predict':
        # TODO
        # predict(model, valid_loader, out_path=root)
        parser.error('TODO')
    else:
        parser.error('Unexpected mode {}'.format(args.mode))


if __name__ == '__main__':
    main()
