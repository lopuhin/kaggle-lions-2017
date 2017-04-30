#!/usr/bin/env python3
import argparse
import gzip
import json
from pathlib import Path
import random
import shutil
from typing import List

import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.affinity import rotate
import skimage.transform
import skimage.io
import torch
from torch import nn
import tqdm

import utils
from unet_models import UNet


class SegmentationDataset(utils.BaseDataset):
    def __init__(self,
                 img_paths: List[Path],
                 coords: pd.DataFrame,
                 transform,
                 size: int,
                 mark_r: int=8,
                 debug: bool=False,
                 deterministic: bool=False,
                 ):
        super().__init__(img_paths, coords)
        self.patch_size = size
        self.mark_r = mark_r
        self.transform = transform
        self.debug = debug
        self.deterministic = deterministic

    def __getitem__(self, idx):
        random.seed(idx if self.deterministic else None)
        while True:
            x = self.new_x_y()
            if x is not None:
                return x

    def new_x_y(self):
        img_id = random.choice(self.img_ids)
        img = self.imgs[img_id]
        max_y, max_x = img.shape[:2]
        s = self.patch_size
        b = int(np.ceil(np.sqrt(2) * s / 2))
        x, y = (random.randint(b, max_x - (b + s)),
                random.randint(b, max_y - (b + s)))
        patch = img[y: y + 2 * b + s, x: x + 2 * b + s]
        try:
            coords = self.coords.loc[[img_id]]
        except KeyError:
            coords = []
        # TODO - sample a point close to some lion with non-zero prob
        # TODO - check that this loop is not too slow, can be vectorized
        any_lions = False
        angle = random.random() * 360
        patch = skimage.transform.rotate(patch, angle, preserve_range=True)
        mask = np.zeros(patch.shape[:2], dtype=np.int64)
        mask[:] = utils.N_CLASSES
        nneg = lambda x: max(0, x)
        c = b + s // 2
        for i in range(len(coords)):
            item = coords.iloc[i]
            ix, iy = item.col - x, item.row - y
            if (0 <= ix <= 2 * b + s) and (0 <= iy <= 2 * b + s):
                p = rotate(Point(ix, iy), -angle, origin=(c, c))
                ix, iy = int(p.x), int(p.y)
                mask[nneg(iy - self.mark_r): nneg(iy + self.mark_r),
                     nneg(ix - self.mark_r): nneg(ix + self.mark_r)] = item.cls
                any_lions = True
        patch = patch[b:, b:][:s, :s]
        mask = mask[b:, b:][:s, :s]
        if (patch.sum(axis=2) == 0).sum() / s**2 > 0.02:
            return None  # masked too much
        if random.random() < 0.5:
            patch = np.flip(patch, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        assert patch.shape == (s, s, 3), patch.shape
        assert mask.shape == (s, s), mask.shape
        if self.debug and any_lions:
            for cls in range(utils.N_CLASSES):
                skimage.io.imsave('mask-{}.jpg'.format(cls),
                                  (mask == cls).astype(np.float32))
            skimage.io.imsave('patch.jpg', patch / 255)
        return self.transform(patch), torch.from_numpy(mask)

    def __len__(self):
        patch_area = self.patch_size ** 2
        return int(sum(img.shape[0] * img.shape[1] / patch_area
                       for img in self.imgs.values()))


def predict(model, img_paths: List[Path], out_path: Path, patch_size: int):
    model.eval()

    def predict(arg):
        img_path, img = arg
        h, w = img.shape[:2]
        s = patch_size
        step = s + 32  # // 2
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
        return img_path, pred_img

    predictions = map(
        predict,
        utils.imap_fixed_output_buffer(
            lambda p: (p, utils.load_image(p, cache=False)),
            tqdm.tqdm(img_paths),
            threads=2))

    for img_path, pred_img in utils.imap_fixed_output_buffer(
            lambda _: next(predictions), img_paths, threads=1):
        resized = np.stack([utils.downsample(p, 4) for p in pred_img])
        binarized = (resized * 1000).astype(np.uint16)
        with gzip.open(str(out_path / '{}-pred.npy'.format(img_path.stem)), 'wb',
                           compresslevel=4) as f:
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
    arg('--nol-weight', type=float, default=1.0)
    arg('--n-folds', type=int, default=5)
    arg('--mode', choices=[
        'train', 'validation', 'predict_valid', 'predict_test', 'predict_all_valid'],
        default='train')
    arg('--clean', action='store_true')
    arg('--epoch-size', type=int)
    arg('--limit', type=int, help='Use only N images for train/valid')
    args = parser.parse_args()

    coords = utils.load_coords()
    train_paths, valid_paths = utils.train_valid_split(args)
    root = Path(args.root)
    model = UNet()
    model = utils.cuda(model)
    class_weights = torch.ones(utils.N_CLASSES + 1)
    class_weights[utils.N_CLASSES] = args.nol_weight
    criterion = nn.NLLLoss2d(weight=utils.cuda(class_weights))
    if args.mode == 'train':
        train_loader, valid_loader = (
            utils.make_loader(SegmentationDataset, args, train_paths, coords),
            utils.make_loader(SegmentationDataset, args, valid_paths, coords,
                              deterministic=True))
        if root.exists() and args.clean:
            shutil.rmtree(str(root))
        root.mkdir(exist_ok=True)
        root.joinpath('params.json').write_text(json.dumps(vars(args)))
        utils.train(args, model, criterion,
              train_loader=train_loader, valid_loader=valid_loader)
    else:
        utils.load_best_model(model, root)
        if args.mode == 'validation':
            valid_loader = utils.make_loader(
                SegmentationDataset, args, valid_paths, coords, deterministic=True)
            utils.validation(model, criterion, valid_loader)
        elif args.mode == 'predict_valid':
            predict(model, valid_paths, out_path=root, patch_size=args.patch_size)
        elif args.mode == 'predict_all_valid':
            # include all paths we did not train on (makes sense only with --limit)
            valid_paths = list(
                set(valid_paths) | (set(utils.labeled_paths()) - set(train_paths)))
            predict(model, valid_paths, out_path=root, patch_size=args.patch_size)
        elif args.mode == 'predict_test':
            test_paths = list(utils.DATA_ROOT.joinpath('Test').glob('*.jpg'))
            out_path = root.joinpath('test')
            out_path.mkdir(exist_ok=True)
            predict(model, test_paths, out_path, patch_size=args.patch_size)
        else:
            parser.error('Unexpected mode {}'.format(args.mode))


if __name__ == '__main__':
    main()
