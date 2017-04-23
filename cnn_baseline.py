#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import random
import shutil
from typing import List

import pandas as pd
import numpy as np
import skimage.io
import skimage.transform
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
import torchvision.models
import tqdm

from utils import (
    N_CLASSES, variable, cuda, load_coords, train_valid_split,
    train, validation, BaseDataset,
)
from models import BaselineCNN


class PatchDataset(BaseDataset):
    def __init__(self,
                 img_paths: List[Path],
                 coords: pd.DataFrame,
                 transform,
                 neg_ratio: float=1,
                 size: int=96,
                 rotate: bool=True,
                 deterministic: bool=False,
                 debug: bool=False,
                 ):
        super().__init__(img_paths, coords)
        self.neg_ratio = neg_ratio
        assert size % 2 == 0
        self.size = size
        self.transform = transform
        self.rotate = rotate
        self.deterministic = deterministic
        self.debug = debug

    def __getitem__(self, idx):
        r = self.size // 2
        if self.rotate:
            r = int(np.ceil(r * np.sqrt(2)))
        if self.deterministic:
            random.seed(idx)
        if idx < len(self.coords):  # positive
            item = self.coords.iloc[idx]
            y, x = int(item.row), int(item.col)
            shift = 16
            y += random.randint(-shift, shift)
            x += random.randint(-shift, shift)
            target = int(item.cls)
            img = self.imgs[item.name]
            max_y, max_x = img.shape[:2]
            x, y = max(r, min(x, max_x - r)), max(r, min(y, max_y - r))
        else:  # negative
            img = random.choice(list(self.imgs.values()))
            target = N_CLASSES
            max_y, max_x = img.shape[:2]
            # can accidentally be close to a positive class
            x, y = (random.randint(r, max_x - r),
                    random.randint(r, max_y - r))
        patch = img[y - r: y + r, x - r: x + r]
        if self.rotate:
            angle = random.random() * 360
            patch = skimage.transform.rotate(patch, angle, preserve_range=True)
            b = int(r - self.size // 2)
            patch = patch[b:, b:][:self.size, :self.size]
        if random.random() < 0.5:
           patch = np.flip(patch, axis=1).copy()
        if self.debug:
            skimage.io.imsave('patch-{}.jpg'.format(target), patch)
        assert patch.shape == (self.size, self.size, 3), patch.shape
        return self.transform(patch), target

    def __len__(self):
        return len(self.coords) * int(1 + self.neg_ratio)


def predict(model: nn.Module, loader: DataLoader, out_path: Path):
    model.eval()
    model.is_fcn = True
    dataset = loader.dataset  # type: PatchDataset
    # TODO - this likely needs splitting into patches
    for img_id, img in tqdm.tqdm(dataset.imgs.items()):
        img = dataset.transform(img)  # type: torch.FloatTensor
        img = img.expand(1, *img.size())
        img = variable(img, volatile=True)
        output = model(img)[0].data.numpy()
        print(output.shape)
        np.save(str(out_path.joinpath(str(img_id))), output)


def make_loader(args, paths: List[Path], coords: pd.DataFrame,
                deterministic: bool=False) -> DataLoader:
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.44, 0.46, 0.46], std=[0.16, 0.15, 0.15]),
    ])
    dataset = PatchDataset(
        img_paths=paths,
        coords=coords,
        size=args.patch_size,
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
    arg('--batch-size', type=int, default=16)
    arg('--patch-size', type=int, default=96)
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
        make_loader(args, train_paths, coords),
        make_loader(args, valid_paths, coords, deterministic=True))

    root = Path(args.root)
    # model = BaselineCNN(patch_size=args.patch_size)
    model = torchvision.models.resnet18(num_classes=N_CLASSES + 1)
    model.avgpool = nn.AvgPool2d(args.patch_size // 32)
    model = cuda(model)
    criterion = nn.CrossEntropyLoss()
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
        predict(model, valid_loader, out_path=root)
    else:
        parser.error('Unexpected mode {}'.format(args.mode))


if __name__ == '__main__':
    main()
