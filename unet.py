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
import numpy as np
import skimage.exposure
import torch
import tqdm

import utils
from unet_models import UNet, UNetWithHead, Loss


class SegmentationDataset(utils.BasePatchDataset):
    def __init__(self, *args, mark_r: int=8, debug: bool=False,
                 downscale=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.mark_r = mark_r
        self.downscale = downscale
        self.debug = debug

    def new_x_y(self, patch, points):
        """ Sample (x, y) pair.
        """
        s = self.patch_size
        m = self.mark_r
        if self.downscale:
            s = s // 4
            m = m // 4
        mask = np.zeros((s, s), dtype=np.int64)
        mask[:] = utils.N_CLASSES
        nneg = lambda x: max(0, x)
        for cls, (x, y) in points:
            if self.downscale:
                x, y = x / 4, y / 4
            ix, iy = int(round(x)), int(round(y))
            mask[nneg(iy - m): nneg(iy + m), nneg(ix - m): nneg(ix + m)] = cls
        if random.random() < 0.5:
            patch = np.flip(patch, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if self.debug and points:
            for cls in range(utils.N_CLASSES):
                utils.save_image('mask-{}.jpg'.format(cls),
                                 (mask == cls).astype(np.float32))
            utils.save_image('patch.jpg', patch)
        return self.transform(patch), torch.from_numpy(mask)


def predict(model, img_paths: List[Path], out_path: Path,
            patch_size: int, batch_size: int,
            is_test=False, downsampled=False,
            test_scale=1.0, min_scale=1.0, max_scale=1.0,
            ):
    model.eval()

    def load_image(path):
        if is_test:
            scale = test_scale
        elif min_scale != max_scale:
            random.seed(path.stem)
            scale = round(random.uniform(min_scale, max_scale), 5)
        else:
            scale = min_scale
        img = utils.load_image(path, cache=False)
        h, w = img.shape[:2]
        if scale != 1:
            h = int(h * scale)
            w = int(w * scale)
            img = cv2.resize(img, (w, h))
        return (path, scale), img

    def predict(arg):
        img_meta, img = arg
        h, w = img.shape[:2]
        s = patch_size
        # step = s // 2 - 32
        step = s - 64
        xs = list(range(0, w - s, step)) + [w - s]
        ys = list(range(0, h - s, step)) + [h - s]
        all_xy = [(x, y) for x in xs for y in ys]
        pred_img = np.zeros((utils.N_CLASSES + 1, h, w), dtype=np.float32)
        pred_count = np.zeros((h, w), dtype=np.int32)

        def make_batch(xy_batch_):
            return (xy_batch_, torch.stack([
                utils.img_transform(img[y: y + s, x: x + s]) for x, y in xy_batch_]))

        for xy_batch, inputs in utils.imap_fixed_output_buffer(
                make_batch, tqdm.tqdm(list(utils.batches(all_xy, batch_size))),
                threads=1):
            outputs = model(utils.variable(inputs, volatile=True))
            outputs_data = np.exp(outputs.data.cpu().numpy())
            for (x, y), pred in zip(xy_batch, outputs_data):
                pred_img[:, y: y + s, x: x + s] += pred
                pred_count[y: y + s, x: x + s] += 1
        pred_img /= np.maximum(pred_count, 1)
        return img_meta, pred_img

    loaded = utils.imap_fixed_output_buffer(
        load_image, tqdm.tqdm(img_paths), threads=4)

    prediction_results = utils.imap_fixed_output_buffer(
        predict, loaded, threads=1)

    def save_prediction(arg):
        (img_path, img_scale), pred_img = arg
        if not downsampled:
            pred_img = np.stack([utils.downsample(p, PRED_SCALE) for p in pred_img])
        binarized = (pred_img * 1000).astype(np.uint16)
        with gzip.open(
                str(out_path / '{}-{:.5f}-pred.npy'.format(
                    img_path.stem, img_scale)),
                'wb', compresslevel=4) as f:
            np.save(f, binarized)
        return img_path.stem

    for _ in utils.imap_fixed_output_buffer(
            save_prediction, prediction_results, threads=4):
        pass


def save_predictions(root: Path, n: int, inputs, targets, outputs):
    batch_size = inputs.size(0)
    inputs_data = inputs.data.cpu().numpy().transpose([0, 2, 3, 1])
    outputs_data = outputs.data.cpu().numpy()
    targets_data = targets.data.cpu().numpy()
    outputs_probs = np.exp(outputs_data)
    for i in range(batch_size):
        prefix = str(root.joinpath('{}-{}'.format(str(n).zfill(2), str(i).zfill(2))))
        utils.save_image(
            '{}-input.jpg'.format(prefix),
            skimage.exposure.rescale_intensity(inputs_data[i], out_range=(0, 1)))
        utils.save_image(
            '{}-output.jpg'.format(prefix), colored_prediction(outputs_probs[i]))
        target_one_hot = np.stack(
            [targets_data[i] == cls for cls in range(utils.N_CLASSES)])
        utils.save_image(
            '{}-target.jpg'.format(prefix),
            colored_prediction(target_one_hot.astype(np.float32)))


def colored_prediction(prediction: np.ndarray) -> np.ndarray:
    h, w = prediction.shape[1:]
    planes = []
    for cls, color in enumerate(utils.CLS_COLORS):
        plane = np.rollaxis(np.array(color * h * w).reshape(h, w, 3), 2)
        plane *= prediction[cls]
        planes.append(plane)
    colored = np.sum(planes, axis=0)
    colored = np.clip(colored, 0, 1)
    return colored.transpose(1, 2, 0)


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
    arg('--oversample', type=float, default=0.0,
        help='sample near lion with given probability')
    arg('--with-head', action='store_true')
    arg('--pred-oddity', type=int, help='set to 0/1 to predict even/odd images')
    args = parser.parse_args()

    coords = utils.load_coords()
    train_paths, valid_paths = utils.train_valid_split(args, coords)
    root = Path(args.root)
    model = UNetWithHead() if args.with_head else UNet()
    model = utils.cuda(model)
    criterion = Loss(dice_weight=args.dice_weight, bg_weight=args.bg_weight)
    loader_kwargs = dict(
        min_scale=args.min_scale, max_scale=args.max_scale,
        downscale=args.with_head,
    )
    if args.mode == 'train':
        train_loader, valid_loader = (
            utils.make_loader(
                SegmentationDataset, args, train_paths, coords,
                oversample=args.oversample, **loader_kwargs),
            utils.make_loader(
                SegmentationDataset, args, valid_paths, coords,
                deterministic=True, **loader_kwargs))
        if root.exists() and args.clean:
            shutil.rmtree(str(root))
        root.mkdir(exist_ok=True)
        root.joinpath('params.json').write_text(
            json.dumps(vars(args), indent=True, sort_keys=True))
        utils.train(args, model, criterion,
                    train_loader=train_loader, valid_loader=valid_loader,
                    save_predictions=save_predictions)
    elif args.mode == 'valid':
        utils.load_best_model(model, root, args.model_path)
        valid_loader = utils.make_loader(
            SegmentationDataset, args, valid_paths, coords,
            deterministic=True, **loader_kwargs)
        utils.validation(model, criterion,
                         tqdm.tqdm(valid_loader, desc='Validation'))
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
