#!/usr/bin/env python3
import argparse
from pathlib import Path
import random

import pandas as pd
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import tqdm

from utils import N_CLASSES, load_image


class PatchDataset(Dataset):
    def __init__(self, img_path: Path, coords: pd.DataFrame, transform,
                 neg_ratio=1,
                 size=96,
                 ):
        img_paths = list(img_path.glob('*.jpg'))
        self.img_ids = [
            int(p.name.split('.')[0]) for p in img_paths]
        self.imgs = {img_id: load_image(p)
                     for img_id, p in zip(self.img_ids, img_paths)}
        self.coords = coords.loc[self.img_ids]
        self.neg_ratio = neg_ratio
        assert size % 2 == 0
        self.size = size
        self.transform = transform

    def __getitem__(self, idx):
        r = self.size // 2
        if idx < len(self.coords):  # positive
            item = self.coords.iloc[idx]
            y, x = item.row, item.col
            target = int(item.cls)
            img = self.imgs[item.name]
            max_y, max_x = img.shape[:2]
            # FIXME - something more proper?
            x, y = max(r, min(x, max_x - r)), max(r, min(y, max_y - r))
        else:  # negative
            img = random.choice(list(self.imgs.values()))
            target = N_CLASSES
            max_y, max_x = img.shape[:2]
            # FIXME - can accidentally be close to positive class
            x, y = (random.randint(r, max_x - r),
                    random.randint(r, max_y - r))
        patch = img[y - r: y + r, x - r: x + r]
        return self.transform(patch), target

    def __len__(self):
        return len(self.coords) * int(1 + self.neg_ratio)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--batch-size', type=int, default=16)
    arg('--patch-size', type=int, default=96)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=2)
    args = parser.parse_args()

    coords = pd.read_csv('./data/coords-threeplusone.csv', index_col=0)
    dataset = PatchDataset(
        img_path=Path('./data/small/Train/'),
        coords=coords,
        size=args.patch_size,
        transform=ToTensor(),
    )

    assert args.patch_size % 8 == 0
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.AvgPool2d(args.patch_size // 8, stride=1),
        nn.Conv2d(64, N_CLASSES + 1, 1),
    )
    loader = DataLoader(
        dataset=dataset,
        shuffle=True,
        num_workers=args.workers,
        batch_size=args.batch_size,
    )

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for n_epoch in range(args.n_epochs):
        model.train()
        losses = []
        tq = tqdm.tqdm(loader)
        for inputs, targets in tq:
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            (args.batch_size * loss).backward()
            optimizer.step()
            losses.append(loss.data[0])
            tq.set_postfix(loss=np.mean(losses[-100:]))


if __name__ == '__main__':
    main()
