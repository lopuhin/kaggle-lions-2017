#!/usr/bin/env python3
import argparse
import gzip
import json
from pathlib import Path
import random
import shutil
from typing import List

from make_submission import PRED_SCALE

import cv2
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.affinity import rotate
import torch
import tqdm

import utils
from unet_models import UNet, Loss


class SegmentationDataset(utils.BaseDataset):
    def __init__(self,
                 img_paths: List[Path],
                 coords: pd.DataFrame,
                 transform,
                 size: int,
                 mark_r: int=8,
                 min_scale: float=1.,
                 max_scale: float=1.,
                 debug: bool=False,
                 deterministic: bool=False,
                 ):
        super().__init__(img_paths, coords)
        self.patch_size = size
        self.mark_r = mark_r
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.transform = transform
        self.debug = debug
        self.deterministic = deterministic
        self.coords_by_img_id = {}
        for img_id in self.img_ids:
            try:
                coords = self.coords.loc[[img_id]]
            except KeyError:
                coords = []
            self.coords_by_img_id[img_id] = coords

    def __getitem__(self, idx):
        if self.deterministic:
            random.seed(idx)
        while True:
            x = self.new_x_y()
            if x is not None:
                return x

    def new_x_y(self):
        """ Sample (x, y) pair.
        """
        img_id = random.choice(self.img_ids)
        img = self.imgs[img_id]
        max_y, max_x = img.shape[:2]
        s = self.patch_size
        scale_aug = not (self.min_scale == self.max_scale == 1)
        if scale_aug:
            scale = random.uniform(self.min_scale, self.max_scale)
            s = int(np.round(s * scale))
        else:
            scale = 1
        b = int(np.ceil(np.sqrt(2) * s / 2))
        x, y = (random.randint(b, max_x - (b + s)),
                random.randint(b, max_y - (b + s)))
        patch = img[y - b: y + b + s, x - b: x + b + s]
        coords = self.coords_by_img_id[img_id]
        any_lions = False
        angle = random.random() * 360
        patch = utils.rotated(patch, angle)
        patch = patch[b:, b:][:s, :s]
        if (patch == 0).sum() / s**2 > 0.02:
            return None  # masked too much
        if scale_aug:
            patch = cv2.resize(patch, (self.patch_size, self.patch_size))
        assert patch.shape == (self.patch_size, self.patch_size, 3), patch.shape
        mask = np.zeros((self.patch_size, self.patch_size), dtype=np.int64)
        mask[:] = utils.N_CLASSES
        nneg = lambda x: max(0, x)
        for cls, col, row in zip(coords.cls, coords.col, coords.row):
            ix, iy = col - x, row - y
            if (-b <= ix <= b + s) and (-b <= iy <= b + s):
                p = rotate(Point(ix, iy), -angle, origin=(s // 2, s // 2))
                ix, iy = int(p.x / scale), int(p.y / scale)
                mask[nneg(iy - self.mark_r): nneg(iy + self.mark_r),
                     nneg(ix - self.mark_r): nneg(ix + self.mark_r)] = cls
                any_lions = True
        if random.random() < 0.5:
            patch = np.flip(patch, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if self.debug and any_lions:
            for cls in range(utils.N_CLASSES):
                utils.save_image('mask-{}.jpg'.format(cls),
                                 (mask == cls).astype(np.float32))
            utils.save_image('patch.jpg', patch)
        return self.transform(patch), torch.from_numpy(mask)

    def __len__(self):
        patch_area = self.patch_size ** 2
        return int(sum(img.shape[0] * img.shape[1] / patch_area
                       for img in self.imgs.values()))


def predict(model, img_paths: List[Path], out_path: Path, patch_size: int,
            is_test=False, test_scale=1.0, min_scale=None, max_scale=None):
    model.eval()

    def predict(arg):
        (img_path, img_scale), img = arg
        h, w = img.shape[:2]
        if img_scale != 1:
            h = int(h * img_scale)
            w = int(w * img_scale)
            img = cv2.resize(img, (w, h))
        s = patch_size
        step = s - 32  # // 2
        xs = list(range(0, w - s, step)) + [w - s]
        ys = list(range(0, h - s, step)) + [h - s]
        all_xy = [(x, y) for x in xs for y in ys]
        pred_img = np.zeros((utils.N_CLASSES + 1, h, w), dtype=np.float32)
        pred_count = np.zeros((h, w), dtype=np.int32)

        def make_batch(xy_batch_):
            return (xy_batch_, torch.stack([
                utils.img_transform(img[y: y + s, x: x + s]) for x, y in xy_batch_]))

        for xy_batch, inputs in utils.imap_fixed_output_buffer(
                make_batch, tqdm.tqdm(list(utils.batches(all_xy, 64))),
                threads=1):
            outputs = model(utils.variable(inputs, volatile=True))
            outputs_data = np.exp(outputs.data.cpu().numpy())
            for (x, y), pred in zip(xy_batch, outputs_data):
                pred_img[:, y: y + s, x: x + s] += pred
                pred_count[y: y + s, x: x + s] += 1
        pred_img /= np.maximum(pred_count, 1)
        return (img_path, img_scale), pred_img

    if is_test:
        scales = [test_scale]
    else:
        if min_scale and max_scale:
            scales = np.linspace(min_scale, max_scale, 4)
        else:
            scales = [1]
    paths_scales = [(p, s) for p in img_paths for s in scales]

    predictions = map(
        predict,
        utils.imap_fixed_output_buffer(
            lambda x: (x, utils.load_image(x[0], cache=False)),
            tqdm.tqdm(paths_scales),
            threads=2))

    for (img_path, img_scale), pred_img in utils.imap_fixed_output_buffer(
            lambda _: next(predictions), paths_scales, threads=1):
        resized = np.stack([utils.downsample(p, PRED_SCALE) for p in pred_img])
        binarized = (resized * 1000).astype(np.uint16)
        with gzip.open(
                str(out_path / '{}-{:.5f}-pred.npy'.format(
                    img_path.stem, img_scale)),
                'wb', compresslevel=4) as f:
            np.save(f, binarized)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('root', help='checkpoint root')
    arg('--batch-size', type=int, default=32)
    arg('--patch-size', type=int, default=256)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=2)
    arg('--fold', type=int, default=1)
    arg('--bg-weight', type=float, default=1.0, help='background weight')
    arg('--dice-weight', type=float, default=0.0)
    arg('--n-folds', type=int, default=5)
    arg('--stratified', action='store_true')
    arg('--mode', choices=[
        'train', 'validation', 'predict_valid', 'predict_test', 'predict_all_valid'],
        default='train')
    arg('--clean', action='store_true')
    arg('--epoch-size', type=int)
    arg('--limit', type=int, help='Use only N images for train/valid')
    arg('--min-scale', type=float, default=1)
    arg('--max-scale', type=float, default=1)
    arg('--test-scale', type=float, default=0.5)
    args = parser.parse_args()

    coords = utils.load_coords()
    train_paths, valid_paths = utils.train_valid_split(args, coords)
    root = Path(args.root)
    model = UNet()
    model = utils.cuda(model)
    criterion = Loss(dice_weight=args.dice_weight, bg_weight=args.bg_weight)
    if args.mode == 'train':
        kwargs = dict(min_scale=args.min_scale, max_scale=args.max_scale)
        train_loader, valid_loader = (
            utils.make_loader(
                SegmentationDataset, args, train_paths, coords, **kwargs),
            utils.make_loader(
                SegmentationDataset, args, valid_paths, coords,
                deterministic=True, **kwargs))
        if root.exists() and args.clean:
            shutil.rmtree(str(root))
        root.mkdir(exist_ok=True)
        root.joinpath('params.json').write_text(
            json.dumps(vars(args), indent=True, sort_keys=True))
        utils.train(args, model, criterion,
                    train_loader=train_loader, valid_loader=valid_loader)
    else:
        utils.load_best_model(model, root)
        if args.mode == 'validation':
            valid_loader = utils.make_loader(
                SegmentationDataset, args, valid_paths, coords, deterministic=True)
            utils.validation(model, criterion, valid_loader)
        elif args.mode == 'predict_valid':
            predict(model, valid_paths, out_path=root, patch_size=args.patch_size,
                    min_scale=args.min_scale, max_scale=args.max_scale)
        elif args.mode == 'predict_all_valid':
            # include all paths we did not train on (makes sense only with --limit)
            valid_paths = list(
                set(valid_paths) | (set(utils.labeled_paths()) - set(train_paths)))
            predict(model, valid_paths, out_path=root, patch_size=args.patch_size,
                    min_scale=args.min_scale, max_scale=args.max_scale)
        elif args.mode == 'predict_test':
            test_paths = list(utils.DATA_ROOT.joinpath('Test').glob('*.jpg'))
            out_path = root.joinpath('test')
            out_path.mkdir(exist_ok=True)
            predict(model, test_paths, out_path, patch_size=args.patch_size,
                    is_test=True, test_scale=args.test_scale)
        else:
            parser.error('Unexpected mode {}'.format(args.mode))


if __name__ == '__main__':
    main()
