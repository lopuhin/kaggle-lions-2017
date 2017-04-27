import json
from datetime import datetime
import glob
from itertools import islice
from pathlib import Path
from pprint import pprint
import random
import shutil
from typing import Dict, List, Tuple
import warnings

import matplotlib.pyplot as plt

import cv2
import pandas as pd
import numpy as np
import skimage.io
import skimage.exposure
from sklearn.model_selection import ShuffleSplit
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
import tqdm


N_CLASSES = 5


cuda_is_available = torch.cuda.is_available()


def variable(x, volatile=False):
    return cuda(Variable(x, volatile=volatile))


def cuda(x):
    return x.cuda() if cuda_is_available else x


img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.44, 0.46, 0.46], std=[0.16, 0.15, 0.15]),
])


def load_image(path: Path, *, cache: bool) -> np.ndarray:
    cached_path = path.parent.joinpath(path.stem + '.npy')  # type: Path
    if cache and cached_path.exists():
        return np.load(str(cached_path))
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if cache:
        with cached_path.open('wb') as f:
            np.save(f, img)
    return img


def make_loader(dataset_cls: type,
                args, paths: List[Path], coords: pd.DataFrame,
                deterministic: bool=False) -> DataLoader:
    dataset = dataset_cls(
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


def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


DATA_ROOT = Path(__file__).absolute().parent / 'data'


def train_valid_split(args) -> Tuple[List[Path], List[Path]]:
    img_paths = list(DATA_ROOT.joinpath('Train').glob('*.jpg'))
    if args.limit and len(img_paths) > args.limit:
        random.seed(42)
        img_paths = random.sample(img_paths, args.limit)
    img_paths = np.array(sorted(img_paths))
    cv_split = ShuffleSplit(n_splits=args.n_folds, random_state=42)
    img_folds = list(cv_split.split(img_paths))
    train_ids, valid_ids = img_folds[args.fold - 1]
    return img_paths[train_ids], img_paths[valid_ids]


def load_coords():
    return pd.read_csv(str(DATA_ROOT / 'coords-threeplusone.csv'),
                       index_col=0)


class BaseDataset(Dataset):
    def __init__(self,
                 img_paths: List[Path],
                 coords: pd.DataFrame,
                 ):
        self.img_ids = [int(p.name.split('.')[0]) for p in img_paths]
        self.imgs = {img_id: load_image(p, cache=True)
                     for img_id, p in tqdm.tqdm(list(zip(self.img_ids, img_paths)),
                                                desc='Images')}
        self.coords = coords.loc[self.img_ids].dropna()


def train(args, model: nn.Module, criterion, *, train_loader, valid_loader):
    optimizer = Adam(model.parameters(), lr=args.lr)

    root = Path(args.root)
    model_path = root / 'model.pt'
    best_model_path = root / 'best-model.pt'
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0
        best_valid_loss = float('inf')

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss
    }, str(model_path))

    report_each = 10
    root = Path(args.root)
    log = root.joinpath('train.log').open('at', encoding='utf8')
    for epoch in range(epoch, args.n_epochs + 1):
        model.train()
        tq = tqdm.tqdm(total=(args.epoch_size or
                              len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}'.format(epoch))
        losses = []
        tl = train_loader
        if args.epoch_size:
            tl = islice(tl, args.epoch_size // args.batch_size)
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs, targets = variable(inputs), variable(targets)
                outputs = model(inputs)
                is_classifier = len(targets.size()) == 1
                if is_classifier:
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
                tq.set_postfix(loss='{:.3f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
                    if not is_classifier:
                        save_predictions(root, inputs, targets, outputs)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                shutil.copy(str(model_path), str(best_model_path))
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return


def save_predictions(root: Path, inputs, targets, outputs):
    batch_size = inputs.size(0)
    inputs_data = inputs.data.cpu().numpy().transpose([0, 2, 3, 1])
    outputs_data = outputs.data.cpu().numpy()
    targets_data = targets.data.cpu().numpy()
    outputs_probs = np.exp(outputs_data)
    for i in range(batch_size):
        prefix = str(root.joinpath(str(i).zfill(2)))
        save_image(
            '{}-input.jpg'.format(prefix),
            skimage.exposure.rescale_intensity(inputs_data[i], out_range=(0, 1)))
        save_image(
            '{}-output.jpg'.format(prefix), colored_prediction(outputs_probs[i]))
        target_one_hot = np.stack([targets_data[i] == cls for cls in range(N_CLASSES)])
        save_image(
            '{}-target.jpg'.format(prefix),
            colored_prediction(target_one_hot.astype(np.float32)))


def colored_prediction(prediction: np.ndarray) -> np.ndarray:
    colors = [
        [1., 0., 0.],  # red: adult males
        [1., 0., 1.],  # magenta: subadult males
        [0.647, 0.1647, 0.1647],  # brown: adult females
        [0., 0., 1.],  # blue: juveniles
        [0., 1., 0.],  # green: pups
    ]
    h, w = prediction.shape[1:]
    planes = []
    for cls, color in enumerate(colors):
        plane = np.rollaxis(np.array(color * h * w).reshape(h, w, 3), 2)
        plane *= prediction[cls]
        planes.append(plane)
    colored = np.sum(planes, axis=0)
    colored = np.clip(colored, 0, 1)
    return colored.transpose(1, 2, 0)


def save_image(fname, data):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        skimage.io.imsave(fname, data)


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


def load_best_model(model: nn.Module, root: Path) -> None:
    state = torch.load(str(root / 'best-model.pt'))
    model.load_state_dict(state['model'])
    print('Loaded model from epoch {epoch}, step {step:,}'.format(**state))


def plot(*args, ymin=None, ymax=None, xmin=None, xmax=None, params=False,
         max_points=200):
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
                if len(xs) > 2 * max_points:
                    indices = (np.arange(0, len(xs), len(xs) / max_points)
                               .astype(np.int32))
                    xs = np.array(xs)[indices[1:]]
                    ys = [np.mean(ys[idx: indices[i + 1]])
                          for i, idx in enumerate(indices[:-1])]
                plt.plot(xs, ys, label='{}: {}'.format(path, key))
    plt.legend()
