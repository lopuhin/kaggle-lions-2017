#!/usr/bin/env python3
import argparse
import json
from functools import partial
from pathlib import Path
import random
import shutil
from typing import List

import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import skimage.exposure
from torch import nn
from torchvision import models
from torch.optim import SGD
import tqdm

import utils


class ClassificationDataset(utils.BaseDataset):
    def __init__(self,
                 img_paths: List[Path],
                 coords: pd.DataFrame,
                 transform,
                 size: int,
                 offset: int=6,
                 min_scale: float=1.,
                 max_scale: float=1.,
                 deterministic: bool=False,
                 ):
        super().__init__(img_paths, coords)
        self.patch_size = size
        self.offset = offset
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.transform = transform
        self.deterministic = deterministic

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if self.deterministic:
            random.seed(idx)
        while True:
            xy = self.get_patch_target()
            if xy is not None:
                return xy

    def get_patch_target(self):
        item = None
        skipped = {4}  # pups - too small, not confused
        while item is None or item.name not in self.imgs or item.cls in skipped:
            item = self.coords.iloc[random.randint(0, len(self.coords) - 1)]
        img_id = item.name
        img = self.imgs[img_id]
        max_y, max_x = img.shape[:2]
        s = self.patch_size
        scale_aug = not (self.min_scale == self.max_scale == 1)
        if scale_aug:
            scale = random.uniform(self.min_scale, self.max_scale)
            s = int(np.round(s / scale))
        b = int(np.ceil(np.sqrt(2) * s / 2))
        x0, y0 = item.col, item.row
        off = self.offset
        half = int(round(2 * b + s) / 2)
        try:
            x = random.randint(max(x0 - off, half), min(x0 + off, max_x - half))
            y = random.randint(max(y0 - off, half), min(y0 + off, max_y - half))
        except ValueError:
            return None
        patch = img[y - half: y + half, x - half: x + half]
        angle = random.random() * 360
        patch = utils.rotated(patch, angle)
        patch = patch[b:, b:][:s, :s]
        if (patch == 0).sum() / s**2 > 0.02:
            return None  # masked too much
        if scale_aug:
            patch = cv2.resize(patch, (self.patch_size, self.patch_size))
        assert patch.shape == (self.patch_size, self.patch_size, 3), patch.shape
        return self.transform(patch), int(item.cls)


class VGGModel(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        num_filters = 2048
        s = patch_size // 32
        self.classifier = nn.Sequential(
            nn.Linear(512 * s**2, num_filters),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(num_filters, num_filters),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(num_filters, utils.N_CLASSES - 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def save_predictions(root: Path, n: int, inputs, targets, outputs):
    batch_size = inputs.size(0)
    inputs_data = inputs.data.cpu().numpy().transpose([0, 2, 3, 1])
    # TODO - a histogram with predictions and correct answer
    for i in range(batch_size):
        prefix = str(root.joinpath(
            '{}-{}'.format(str(n).zfill(2), str(i).zfill(2))))
        utils.save_image(
            '{}-input.jpg'.format(prefix),
            skimage.exposure.rescale_intensity(
                inputs_data[i], out_range=(0, 1)))
        probs = [0.1, 0.4, 0.5, 0.01]  # FIXME
        cls_names = utils.CLS_NAMES[:-1]
        xs = range(len(cls_names))

        plt.figure()
        barlist = plt.bar(xs, probs)
        cls = 1  # FIXME
        barlist[1].set_color('r')
        plt.ylim(0, 1)
        plt.xticks(xs, cls_names)
        plt.savefig('{}-output.png'.format(prefix))
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('root', help='checkpoint root')
    arg('--batch-size', type=int, default=32)
    arg('--patch-size', type=int, default=160)
    arg('--offset', type=int, default=6)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=2)
    arg('--fold', type=int, default=1)
    arg('--n-folds', type=int, default=5)
    arg('--stratified', action='store_true')
    arg('--mode', choices=[
        'train', 'valid', 'predict_valid', 'predict_test', 'predict_all_valid'],
        default='train')
    arg('--model-path',
        help='path to model file to use for validation/prediction')
    arg('--clean', action='store_true')
    arg('--epoch-size', type=int)
    arg('--limit', type=int, help='Use only N images for train/valid')
    arg('--min-scale', type=float, default=1)
    arg('--max-scale', type=float, default=1)
    arg('--test-scale', type=float, default=0.5)
    arg('--pred-oddity', type=int, help='set to 0/1 to predict even/odd images')
    args = parser.parse_args()

    coords = utils.load_coords()
    train_paths, valid_paths = utils.train_valid_split(args, coords)
    root = Path(args.root)
    model = VGGModel(args.patch_size)
    model = utils.cuda(model)
    criterion = nn.CrossEntropyLoss()
    loader_kwargs = dict(
        min_scale=args.min_scale, max_scale=args.max_scale, offset=args.offset)
    if args.mode == 'train':
        train_loader, valid_loader = (
            utils.make_loader(ClassificationDataset, args, train_paths, coords,
                              **loader_kwargs),
            utils.make_loader(ClassificationDataset, args, valid_paths, coords,
                              deterministic=True, **loader_kwargs))
        if root.exists() and args.clean:
            shutil.rmtree(str(root))
        root.mkdir(exist_ok=True)
        root.joinpath('params.json').write_text(
            json.dumps(vars(args), indent=True, sort_keys=True))
        utils.train(
            args, model, criterion,
            train_loader=train_loader, valid_loader=valid_loader,
            save_predictions=save_predictions,
            is_classification=True,
            optimizer_cls=partial(SGD, nesterov=True, momentum=0.9),
        )
    elif args.mode == 'valid':
        utils.load_best_model(model, root, args.model_path)
        valid_loader = utils.make_loader(
            ClassificationDataset, args, valid_paths, coords,
            deterministic=True, **loader_kwargs)
        utils.validation(model, criterion,
                         tqdm.tqdm(valid_loader, desc='Validation'),
                         is_classification=True)
    else:
        utils.load_best_model(model, root, args.model_path)
        if args.mode in {'predict_valid', 'predict_all_valid'}:
            if args.mode == 'predict_all_valid':
                # include all paths we did not train on (makes sense only with --limit)
                valid_paths = list(
                    set(valid_paths) | (set(utils.labeled_paths()) - set(train_paths)))
            predict(model, valid_paths, out_path=root,
                    patch_size=args.patch_size, batch_size=args.batch_size,
                    min_scale=args.min_scale, max_scale=args.max_scale,
                    downsampled=args.with_head)
        elif args.mode == 'predict_test':
            out_path = root.joinpath('test')
            out_path.mkdir(exist_ok=True)
            predicted = {p.stem.split('-')[0] for p in out_path.glob('*.npy')}
            test_paths = [p for p in utils.DATA_ROOT.joinpath('Test').glob('*.jpg')
                          if p.stem not in predicted]
            if args.pred_oddity is not None:
                assert args.pred_oddity in {0, 1}
                test_paths = [p for p in test_paths
                              if int(p.stem) % 2 == args.pred_oddity]
            predict(model, test_paths, out_path,
                    patch_size=args.patch_size, batch_size=args.batch_size,
                    test_scale=args.test_scale,
                    is_test=True, downsampled=args.with_head)
        else:
            parser.error('Unexpected mode {}'.format(args.mode))


if __name__ == '__main__':
    main()
