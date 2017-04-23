#!/usr/bin/env python3
import argparse
from itertools import islice
import json
from pathlib import Path
import random
import shutil
from typing import Dict, List

import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
import skimage.transform
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import tqdm

from utils import N_CLASSES, load_image, write_event, variable, cuda
from models import BaselineCNN


class PatchDataset(Dataset):
    def __init__(self,
                 img_paths: List[Path],
                 coords: pd.DataFrame,
                 transform,
                 neg_ratio: float=1,
                 size: int=96,
                 rotate: bool=True,
                 deterministic: bool=False,
                 ):
        self.img_ids = [
            int(p.name.split('.')[0]) for p in img_paths]
        self.imgs = {img_id: load_image(p)
                     for img_id, p in zip(self.img_ids, img_paths)}
        self.coords = coords.loc[self.img_ids]
        self.neg_ratio = neg_ratio
        assert size % 2 == 0
        self.size = size
        self.transform = transform
        self.rotate = rotate
        self.deterministic = deterministic

    def __getitem__(self, idx):
        r = self.size // 2
        if self.rotate:
            r = int(np.ceil(r * np.sqrt(2)))
        if self.deterministic:
            random.seed(idx)
        if idx < len(self.coords):  # positive
            item = self.coords.iloc[idx]
            y, x = item.row, item.col
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
            patch = skimage.transform.rotate(patch, angle)
            b = int(r - self.size // 2)
            patch = patch[b:, b:][:self.size, :self.size]
        if random.random() < 0.5:
           patch = np.flip(patch, axis=1).copy()
        assert patch.shape == (self.size, self.size, 3), patch.shape
        return self.transform(patch), target

    def __len__(self):
        return len(self.coords) * int(1 + self.neg_ratio)


def train(args, model: nn.Module, criterion, *, train_loader, valid_loader):
    optimizer = Adam(model.parameters(), lr=args.lr)

    model_path = Path(args.root).joinpath('model.pt')
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    report_each = 10
    log = Path(args.root).joinpath('train.log').open('at', encoding='utf8')
    for epoch in range(epoch, args.n_epochs + 1):
        model.train()
        if args.epoch_size:
            epoch_batches = args.epoch_size // args.batch_size
            tq = tqdm.tqdm(islice(train_loader, epoch_batches),
                           total=epoch_batches)
        else:
            tq = tqdm.tqdm(train_loader)
        tq.set_description('Epoch {}'.format(epoch))
        losses = []
        try:
            for i, (inputs, targets) in enumerate(tq):
                inputs, targets = variable(inputs), variable(targets)
                outputs = model(inputs)
                outputs = outputs.view(outputs.size(0), -1)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                (batch_size * loss).backward()
                optimizer.step()
                step += 1
                losses.append(loss.data[0])
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss=mean_loss)
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader)
            write_event(log, step, **valid_metrics)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return


def validation(model: nn.Module, criterion, valid_loader) -> Dict[str, float]:
    model.eval()
    losses = []
    for inputs, targets in valid_loader:
        inputs, targets = variable(inputs, volatile=True), variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
    valid_loss = np.mean(losses)  # type: float
    print('Valid loss: {:.3f}'.format(valid_loss))
    return {'valid_loss': valid_loss}


def make_loader(args, paths: List[Path], coords: pd.DataFrame,
                deterministic: bool=False) -> DataLoader:
    dataset = PatchDataset(
        img_paths=paths,
        coords=coords,
        size=args.patch_size,
        transform=ToTensor(),
        deterministic=deterministic,
    )
    return DataLoader(
        dataset=dataset,
        shuffle=True,
        num_workers=args.workers,
        batch_size=args.batch_size,
    )


def train_valid_split(args):
    coords = pd.read_csv('./data/coords-threeplusone.csv', index_col=0)
    img_paths = np.array(list(Path('./data/Train/').glob('*.jpg')))
    cv_split = ShuffleSplit(n_splits=args.n_folds, random_state=42)
    img_folds = list(cv_split.split(img_paths))
    train_ids, valid_ids = img_folds[args.fold - 1]
    train_paths, valid_paths = img_paths[train_ids], img_paths[valid_ids]
    return (make_loader(args, train_paths, coords),
            make_loader(args, valid_paths, coords, deterministic=True))


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
    arg('--mode', choices=['train', 'validation'], default='train')
    arg('--clean', action='store_true')
    arg('--epoch-size', type=int)
    args = parser.parse_args()

    train_loader, valid_loader = train_valid_split(args)

    root = Path(args.root)
    model = BaselineCNN(patch_size=args.patch_size)
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


if __name__ == '__main__':
    main()
