from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
import glob
import gzip
from itertools import islice
import functools
from pathlib import Path
from pprint import pprint
import random
import shutil
from typing import Dict, List, Tuple
import warnings

import matplotlib.pyplot as plt

import cv2
import json_lines
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.affinity import rotate
import skimage.io
import skimage.exposure
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold
import statprof
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
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))


def cuda(x):
    return x.cuda() if cuda_is_available else x


img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.44, 0.46, 0.46], std=[0.16, 0.15, 0.15]),
])


def profile(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        statprof.start()
        try:
            return fn(*args, **kwargs)
        finally:
            statprof.stop()
            statprof.display()
    return wrapped


def load_image(path: Path, *, cache: bool) -> np.ndarray:
    cached_path = path.parent / 'cache' / (path.stem + '.npy')  # type: Path
    if cache and cached_path.exists():
        return np.load(str(cached_path))
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if path.parent.name == 'Train':
        # mask with TrainDotted
        img_dotted = cv2.imread(str(path.parent.parent / 'TrainDotted' / path.name))
        img_dotted = cv2.cvtColor(img_dotted, cv2.COLOR_BGR2RGB)
        img[img_dotted.sum(axis=2) == 0, :] = 0
    if cache:
        with cached_path.open('wb') as f:
            np.save(f, img)
    return img


def load_pred(path: Path) -> np.ndarray:
    with gzip.open(str(path), 'rb') as f:
        return np.load(f).astype(np.float32) / 1000


def labeled_paths() -> List[Path]:
    # https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/30895
    mismatched = pd.read_csv(str(DATA_ROOT / 'MismatchedTrainImages.txt'))
    bad_ids = set(mismatched.train_id)
    # https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/31424
    bad_ids.update([941,  200])
    # FIXME - these are valid but have no coords, get them (esp. 912)!
    # https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/31472#175541
    bad_ids.update([491, 912])
    return [p for p in DATA_ROOT.joinpath('Train').glob('*.jpg')
            if int(p.stem) not in bad_ids]


def downsample(img: np.ndarray, ratio: int=4) -> np.ndarray:
    h, w = img.shape
    h = int(h / ratio)
    w = int(w / ratio)
    return cv2.resize(img, (w, h))


def make_loader(dataset_cls: type,
                args, paths: List[Path], coords: pd.DataFrame,
                deterministic: bool=False, **kwargs) -> DataLoader:
    dataset = dataset_cls(
        img_paths=paths,
        coords=coords,
        size=args.patch_size,
        transform=img_transform,
        deterministic=deterministic,
        **kwargs,
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


def train_valid_split(args, coords) -> Tuple[List[Path], List[Path]]:
    img_paths = labeled_paths()
    if args.limit and len(img_paths) > args.limit:
        random.seed(42)
        img_paths = random.sample(img_paths, args.limit)
    if args.stratified:
        sorted_ids = coords['cls'].groupby(level=0).count().sort_values().index
        idx_by_id = {img_id: idx for idx, img_id in enumerate(sorted_ids)}
        img_paths.sort(key=lambda p: idx_by_id.get(int(p.stem), len(sorted_ids)))
        train, test = [], []
        for i, p in enumerate(img_paths):
            if i % args.n_folds == args.fold - 1:
                test.append(p)
            else:
                train.append(p)
        return train, test
    else:
        img_paths = np.array(sorted(img_paths))
        cv_split = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        img_folds = list(cv_split.split(img_paths))
        train_ids, valid_ids = img_folds[args.fold - 1]
        return img_paths[train_ids], img_paths[valid_ids]


def load_coords():
    return pd.read_csv(str(DATA_ROOT / 'coords-threeplusone-v0.4.csv'),
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
        self.coords_by_img_id = {}
        for img_id in self.img_ids:
            try:
                coords = self.coords.loc[[img_id]]
            except KeyError:
                coords = []
            self.coords_by_img_id[img_id] = coords


class BasePatchDataset(BaseDataset):
    def __init__(self,
                 img_paths: List[Path],
                 coords: pd.DataFrame,
                 transform,
                 size: int,
                 min_scale: float=1.,
                 max_scale: float=1.,
                 oversample: float=0.,
                 deterministic: bool=False,
                 ):
        super().__init__(img_paths, coords)
        self.patch_size = size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.oversample = oversample
        self.transform = transform
        self.deterministic = deterministic

    def __getitem__(self, idx):
        if self.deterministic:
            random.seed(idx)
        while True:
            pp = self.get_patch_points()
            if pp is not None:
                return self.new_x_y(*pp)

    def new_x_y(self, patch, points):
        """ Sample (x, y) pair.
        """
        raise NotImplementedError

    def get_patch_points(self):
        oversample = self.oversample and random.random() < self.oversample
        if oversample:
            item = None
            while item is None or item.name not in self.imgs:
                item = self.coords.iloc[random.randint(0, len(self.coords) - 1)]
            img_id = item.name
        else:
            img_id = random.choice(self.img_ids)
        img = self.imgs[img_id]
        max_y, max_x = img.shape[:2]
        s = self.patch_size
        scale_aug = not (self.min_scale == self.max_scale == 1)
        if scale_aug:
            scale = random.uniform(self.min_scale, self.max_scale)
            s = int(np.round(s / scale))
        else:
            scale = 1
        coords = self.coords_by_img_id[img_id]
        b = int(np.ceil(np.sqrt(2) * s / 2))
        if oversample:
            item = coords.iloc[random.randint(0, len(coords) - 1)]
            x0, y0 = item.col, item.row
            try:
                x = random.randint(max(x0 - s, b), min(x0 + s, max_x - (b + s)))
                y = random.randint(max(y0 - s, b), min(y0 + s, max_y - (b + s)))
            except ValueError:
                oversample = False  # this can happen with large x0 or y0
        if not oversample:
            x = random.randint(b, max_x - (b + s))
            y = random.randint(b, max_y - (b + s))
        patch = img[y - b: y + b + s, x - b: x + b + s]
        angle = random.random() * 360
        patch = rotated(patch, angle)
        patch = patch[b:, b:][:s, :s]
        if (patch == 0).sum() / s**2 > 0.02:
            return None  # masked too much
        if scale_aug:
            patch = cv2.resize(patch, (self.patch_size, self.patch_size))
        assert patch.shape == (self.patch_size, self.patch_size, 3), patch.shape
        points = []
        if len(coords) > 0:
            for cls, col, row in zip(coords.cls, coords.col, coords.row):
                ix, iy = col - x, row - y
                if (-b <= ix <= b + s) and (-b <= iy <= b + s):
                    p = rotate(Point(ix, iy), -angle, origin=(s // 2, s // 2))
                    points.append((cls, (p.x * scale, p.y * scale)))
        return patch, points

    def __len__(self):
        patch_area = self.patch_size ** 2
        return int(sum(img.shape[0] * img.shape[1] / patch_area
                       for img in self.imgs.values()))


def train(args, model: nn.Module, criterion, *, train_loader, valid_loader,
          make_optimizer=None, save_predictions=None, is_classification=False):
    lr = args.lr
    make_optimizer = make_optimizer or (lambda lr: Adam(model.parameters(), lr=lr))
    optimizer = make_optimizer(lr)

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

    def save(ep: int):
        torch.save(
            {'model': model.state_dict(),
             'epoch': ep,
             'step': step,
             'best_valid_loss': best_valid_loss,
             }, str(model_path))
        shutil.copy(str(model_path), str(root / 'model-{}.pt'.format(ep)))

    report_each = 10
    save_prediction_each = report_each * 10
    root = Path(args.root)
    log = root.joinpath('train.log').open('at', encoding='utf8')
    valid_losses = []
    for epoch in range(epoch, args.n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(args.epoch_size or
                              len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        if args.epoch_size:
            tl = islice(tl, args.epoch_size // args.batch_size)
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs, targets = variable(inputs), variable(targets)
                outputs = model(inputs)
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
                    if save_predictions and i % save_prediction_each == 0:
                        p_i = (i // save_prediction_each) % 5
                        save_predictions(root, p_i, inputs, targets, outputs)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader,
                                       is_classification=is_classification)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                shutil.copy(str(model_path), str(best_model_path))
            elif len(valid_losses) > 2 and min(valid_losses[-2:]) > best_valid_loss:
                # two epochs without improvement
                lr /= 5
                optimizer = make_optimizer(lr)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return


def rotated(patch, angle):
    size = patch.shape[:2]
    center = tuple(np.array(size) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(patch, rot_mat, size, flags=cv2.INTER_LINEAR)


def save_image(fname, data):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        skimage.io.imsave(fname, data)


CLS_COLORS = [
    [1., 0., 0.],  # red: adult males
    [1., 0., 1.],  # magenta: subadult males
    [0.647, 0.1647, 0.1647],  # brown: adult females
    [0., 0., 1.],  # blue: juveniles
    [0., 1., 0.],  # green: pups
]
CLS_NAMES = ['male', 'sub_male', 'female', 'juv', 'pup']


def validation(model: nn.Module, criterion, valid_loader,
               is_classification=False) -> Dict[str, float]:
    model.eval()
    losses = []
    all_targets, all_outputs = [], []
    for inputs, targets in valid_loader:
        inputs, targets = variable(inputs, volatile=True), variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
        if is_classification:
            all_targets.extend(targets.data.cpu().numpy())
            all_outputs.extend(outputs.data.cpu().numpy().argmax(axis=1))
    valid_loss = np.mean(losses)  # type: float
    metrics = {'valid_loss': valid_loss}
    print('Valid loss: {:.5f}'.format(valid_loss))
    if is_classification:
        accuracy = accuracy_score(all_targets, all_outputs)
        print('Accuracy: {:.3f}'.format(accuracy))
        print(classification_report(all_targets, all_outputs))
        metrics['accuracy'] = accuracy
    return metrics


def load_best_model(model: nn.Module, root: Path, model_path=None) -> None:
    model_path = model_path or str(root / 'best-model.pt')
    state = torch.load(model_path)
    model.load_state_dict(state['model'])
    print('Loaded model from epoch {epoch}, step {step:,}'.format(**state))


def batches(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def imap_fixed_output_buffer(fn, it, threads: int):
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        max_futures = threads + 1
        for x in it:
            while len(futures) >= max_futures:
                future, futures = futures[0], futures[1:]
                yield future.result()
            futures.append(executor.submit(fn, x))
        for future in futures:
            yield future.result()


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
    for path in paths:
        path = Path(path)
        with json_lines.open(str(path.joinpath('train.log')), broken=True) as f:
            events = list(f)
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
