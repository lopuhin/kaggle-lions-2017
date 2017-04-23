#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import random
import shutil
from typing import List

import pandas as pd
import numpy as np
import skimage.transform
import skimage.io
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from utils import (
    N_CLASSES, cuda, load_coords, train_valid_split, train, validation,
    BaseDataset
)
from unet_models import UNet


class SegmentationDataset(BaseDataset):
    def __init__(self,
                 img_paths: List[Path],
                 coords: pd.DataFrame,
                 transform,
                 patch_size: int,
                 mask_r: int=16,
                 debug: bool=False,
                 deterministic: bool=False,
                 ):
        super().__init__(img_paths, coords)
        self.patch_size = patch_size
        self.mark_r = mask_r
        self.transform = transform
        self.debug = debug
        self.deterministic = deterministic
        self.to_tensor = ToTensor()

    def __getitem__(self, idx):
        if self.deterministic:
            random.seed(idx)
        img_id = self.img_ids[idx % len(self.img_ids)]
        img = self.imgs[img_id]
        max_y, max_x = img.shape[:2]
        s = self.patch_size
        b = int(np.ceil(np.sqrt(2) * s / 2))
        x, y = (random.randint(b, max_x - (b + s)),
                random.randint(b, max_y - (b + s)))
        patch = img[y: y + 2 * b + s, x: x + 2 * b + s]
        mask = np.zeros_like(patch, dtype=np.int32)
        mask[:] = N_CLASSES
        coords = self.coords.loc[img_id]
        # TODO - check that this loop is not too slow
        any_lions = False
        for i in range(len(coords)):
            item = coords.iloc[i]
            ix, iy = item.col - x, item.row - y
            if (0 >= ix <= 2 * b + s) and (0 >= iy <= 2 * b + s):
                mask[max(0, iy - self.mark_r): iy + self.mark_r,
                     max(0, ix - self.mark_r): ix + self.mark_r] = item.cls
                any_lions = True
        angle = random.random() * 360
        patch = skimage.transform.rotate(patch, angle, preserve_range=True)
        mask = skimage.transform.rotate(mask, angle, preserve_range=True)
        patch = patch[b:, b:][:s, :s]
        mask = mask[b:, b:][:s, :s]
        if random.random() < 0.5:
            patch = np.flip(patch, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        assert patch.shape == mask.shape == (s, s, 3), (patch.shape, mask.shape)
        if any_lions and self.debug:
            # TODO - check
            skimage.io.imsave('patch.jpg', patch)
            skimage.io.imsave('mask.jpg', mask * 255 / N_CLASSES)
        return self.transform(patch), self.to_tensor(mask)

    def __len__(self):
        patch_area = self.patch_size ** 2
        return int(sum(img.shape[0] * img.shape[1] / patch_area
                       for img in self.imgs.values()))


def make_loader(args, paths: List[Path], coords: pd.DataFrame,
                deterministic: bool=False) -> DataLoader:
    transform = Compose([
        ToTensor(),
        # TODO - check actual values
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = SegmentationDataset(
        img_paths=paths,
        coords=coords,
        patch_size=args.patch_size,
        transform=transform,
        deterministic=deterministic,
    )
    return DataLoader(
        dataset=dataset,
        shuffle=True,
        num_workers=args.workers,
        batch_size=args.batch_size,
    )


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('root', help='checkpoint root')
    arg('--batch-size', type=int, default=4)
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
    args = parser.parse_args()

    coords = load_coords()
    train_paths, valid_paths = train_valid_split(args)
    train_loader, valid_loader = (
        make_loader(args, train_paths, coords),
        make_loader(args, valid_paths, coords, deterministic=True))

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
