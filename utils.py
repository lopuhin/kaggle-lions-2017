import json
from datetime import datetime
import glob
from itertools import islice
from pathlib import Path
from pprint import pprint
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import Dataset
import tqdm


N_CLASSES = 5


cuda_is_available = torch.cuda.is_available()


def variable(x, volatile=False):
    return cuda(Variable(x, volatile=volatile))


def cuda(x):
    return x.cuda() if cuda_is_available else x


def load_image(path: Path, cache: bool=True) -> np.ndarray:
    cached_path = path.parent.joinpath(path.stem + '.npy')  # type: Path
    if cache and cached_path.exists():
        return np.load(str(cached_path))
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if cache:
        with cached_path.open('wb') as f:
            np.save(f, img)
    return img


def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def train_valid_split(args) -> Tuple[List[Path], List[Path]]:
    img_paths = list(Path('./data/Train/').glob('*.jpg'))
    if args.limit and len(img_paths) > args.limit:
        random.seed(42)
        img_paths = random.sample(img_paths, args.limit)
    img_paths = np.array(sorted(img_paths))
    cv_split = ShuffleSplit(n_splits=args.n_folds, random_state=42)
    img_folds = list(cv_split.split(img_paths))
    train_ids, valid_ids = img_folds[args.fold - 1]
    return img_paths[train_ids], img_paths[valid_ids]


def load_coords():
    return pd.read_csv('./data/coords-threeplusone.csv', index_col=0)


class BaseDataset(Dataset):
    def __init__(self,
                 img_paths: List[Path],
                 coords: pd.DataFrame,
                 ):
        self.img_ids = [int(p.name.split('.')[0]) for p in img_paths]
        tq_img_paths = tqdm.tqdm(img_paths, desc='Images')
        self.imgs = {img_id: load_image(p)
                     for img_id, p in zip(self.img_ids, tq_img_paths)}
        tq_img_paths.close()
        self.coords = coords.loc[self.img_ids].dropna()


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

    report_each = 50
    log = Path(args.root).joinpath('train.log').open('at', encoding='utf8')
    for epoch in range(epoch, args.n_epochs + 1):
        model.train()
        tq = tqdm.tqdm(total=(args.epoch_size or
                              len(train_loader) * args.batch_size))
        if args.epoch_size:
            train_loader = islice(
                train_loader, args.epoch_size // args.batch_size)
        tq.set_description('Epoch {}'.format(epoch))
        losses = []
        try:
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = variable(inputs), variable(targets)
                outputs = model(inputs)
                outputs = outputs.view(outputs.size(0), -1)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                (batch_size * loss).backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
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


def plot(*args, ymin=None, ymax=None, xmin=None, xmax=None, params=False):
    """ Use in the notebook like this:
    plot('./runs/oc2', './runs/oc1', 'loss', 'valid_loss')
    """
    paths, keys = [], []
    for x in args:
        if x.startswith('.') or x.startswith('/'):
            if '*' in x:
                paths.extend(glob.glob(x))
            else:
                paths.append(x)
        else:
            keys.append(x)
    plt.figure(figsize=(12, 8))
    keys = keys or ['loss', 'valid_loss']

    ylim_kw = {}
    if ymin is not None:
        ylim_kw['ymin'] = ymin
    if ymax is not None:
        ylim_kw['ymax'] = ymax
    if ylim_kw:
        plt.ylim(**ylim_kw)

    xlim_kw = {}
    if xmin is not None:
        xlim_kw['xmin'] = xmin
    if xmax is not None:
        xlim_kw['xmax'] = xmax
    if xlim_kw:
        plt.xlim(**xlim_kw)
    for path in sorted(paths):
        path = Path(path)
        with path.joinpath('train.log').open() as f:
            events = [json.loads(line) for line in f]
        if params:
            print(path)
            pprint(json.loads(path.joinpath('params.json').read_text()))
        for key in sorted(keys):
            xs, ys = [], []
            for e in events:
                if key in e:
                    xs.append(e['step'])
                    ys.append(e[key])
            if xs:
                plt.plot(xs, ys, label='{}: {}'.format(path, key))
    plt.legend()
