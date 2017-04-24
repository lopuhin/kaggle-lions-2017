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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
import torchvision.models
import tqdm

from utils import (
    N_CLASSES, variable, cuda, load_coords, train_valid_split,
    train, validation, BaseDataset, load_best_model, load_image,
)


class PatchDataset(BaseDataset):
    def __init__(self,
                 img_paths: List[Path],
                 coords: pd.DataFrame,
                 transform,
                 neg_ratio: float=1,  # or 5
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


def make_resnet_fcn(model: torchvision.models.ResNet):
    fc_bias = variable(model.fc.state_dict()['bias'])
    fc_weight = variable(model.fc.state_dict()['weight'])
    fc_weight = fc_weight.unsqueeze(2).unsqueeze(3)

    def forward(x):
        self = model
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = F.conv2d(x, fc_weight, fc_bias)
        return x

    model.forward = forward


def predict(model: nn.Module, img_paths: List[Path], out_path: Path):
    make_resnet_fcn(model)
    model.eval()
    for img_path in tqdm.tqdm(img_paths):
        img = load_image(img_path, cache=False)
        inputs = img_transform(img)
        inputs = inputs.unsqueeze(0)
        inputs = variable(inputs, volatile=True)
        outputs = F.softmax(model(inputs))
        output = outputs[0].data.cpu().numpy()
        np.save(str(out_path / '{}-pred.npy'.format(img_path.stem)),
                output)


img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.44, 0.46, 0.46], std=[0.16, 0.15, 0.15]),
])


def make_loader(args, paths: List[Path], coords: pd.DataFrame,
                deterministic: bool=False) -> DataLoader:
    dataset = PatchDataset(
        img_paths=paths,
        coords=coords,
        size=args.patch_size,
        transform=img_transform,
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
    arg('--mode', choices=['train', 'validation', 'predict_valid'],
        default='train')
    arg('--clean', action='store_true')
    arg('--epoch-size', type=int)
    arg('--limit', type=int, help='Use only N images for train/valid')
    args = parser.parse_args()

    root = Path(args.root)
    coords = load_coords()
    train_paths, valid_paths = train_valid_split(args)
    # model = BaselineCNN(patch_size=args.patch_size)
    model = torchvision.models.resnet18(num_classes=N_CLASSES + 1)
    model.avgpool = nn.AvgPool2d(args.patch_size // 32, stride=1)
    model = cuda(model)
    criterion = nn.CrossEntropyLoss()
    if args.mode == 'train':
        train_loader, valid_loader = (
            make_loader(args, train_paths, coords),
            make_loader(args, valid_paths, coords, deterministic=True))
        if root.exists() and args.clean:
            shutil.rmtree(str(root))
        root.mkdir(exist_ok=True)
        root.joinpath('params.json').write_text(json.dumps(vars(args)))
        train(args, model, criterion,
              train_loader=train_loader, valid_loader=valid_loader)
    elif args.mode == 'validation':
        valid_loader = (
            make_loader(args, valid_paths, coords, deterministic=True))
        load_best_model(model, root)
        validation(model, criterion, valid_loader)
    elif args.mode == 'predict_valid':
        load_best_model(model, root)
        predict(model, valid_paths, out_path=root)
    else:
        parser.error('Unexpected mode {}'.format(args.mode))


if __name__ == '__main__':
    main()
