#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import random

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import tqdm

from utils import N_CLASSES, load_image, write_event
from models import BaselineCNN


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
        # TODO: augmentations: rotation, small shift, flip
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


def train(args, model, loader):
    criterion = nn.CrossEntropyLoss()
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
        losses = []
        tq = tqdm.tqdm(loader)
        tq.set_description('Epoch {}'.format(epoch))
        try:
            for i, (inputs, targets) in enumerate(tq):
                inputs, targets = Variable(inputs), Variable(targets)
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
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('root', help='checkpoint root')
    arg('--batch-size', type=int, default=16)
    arg('--patch-size', type=int, default=96)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=2)
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True)
    root.joinpath('params.json').write_text(json.dumps(vars(args)))

    coords = pd.read_csv('./data/coords-threeplusone.csv', index_col=0)
    dataset = PatchDataset(
        img_path=Path('./data/Train/'),
        coords=coords,
        size=args.patch_size,
        transform=ToTensor(),
    )

    loader = DataLoader(
        dataset=dataset,
        shuffle=True,
        num_workers=args.workers,
        batch_size=args.batch_size,
    )

    model = BaselineCNN(patch_size=args.patch_size)
    train(args, model, loader)


if __name__ == '__main__':
    main()
